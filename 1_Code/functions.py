from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import pandas as pd
import xgboost
from sklearn.linear_model import (
    ElasticNetCV,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, cross_val_predict
import numpy as np

def get_common_features(train_data, test_data, mode="common", target_column=None):
    """
    Identify common or dropped features between train_data and test_data.

    Parameters:
    - train_data: Training data DataFrame.
    - test_data: Testing data DataFrame.
    - mode: "common" to return common features or "dropped" to return dropped features.
    - target_column: (Optional) The target column name in the training data. Default is None.

    Returns:
    - List of common or dropped features.
    """

    # Identify common features
    common_features = train_data.columns.intersection(test_data.columns).tolist()

    if mode == "common":
        # If a target column is specified and it's in the common features, remove it
        if target_column and target_column in common_features:
            common_features.remove(target_column)
        return common_features

    elif mode == "dropped":
        all_features = set(train_data.columns.tolist() + test_data.columns.tolist())
        dropped_features = list(all_features - set(common_features))

        # If a target column is specified and it's in the dropped features, remove it
        if target_column and target_column in dropped_features:
            dropped_features.remove(target_column)
        return dropped_features

    else:
        raise ValueError("Invalid mode. Choose either 'common' or 'dropped'.")

def encode_categorical_columns(data, categorical_columns, encoding_type="label"):
    """
    Encodes categorical columns.

    Parameters:
    - data: DataFrame
    - categorical_columns: List of columns to encode
    - encoding_type: 'label' for label encoding, 'ordinal' for ordinal encoding, 'onehot' for one-hot encoding

    Returns: DataFrame
    """
    if encoding_type == "label":
        for column in categorical_columns:
            encoder = LabelEncoder()
            data[column] = encoder.fit_transform(data[column])
    elif encoding_type == "ordinal":
        encoder = OrdinalEncoder()
        data[categorical_columns] = encoder.fit_transform(data[categorical_columns])
    elif encoding_type == "onehot":
        data = pd.get_dummies(data, columns=categorical_columns, drop_first=False)
    else:
        raise ValueError(f"Invalid encoding_type: {encoding_type}")

    return data

def calculate_exact_age(row):
    age = row["Survey_year"] - row["Birthyear"]
    age += (row["Survey_month"] - row["Birthmonth"]) / 12
    return age

def transform_percentage_columns(data, percentage_columns, percentage_mapping):
    """
    Transforms percentage columns based on given mapping.

    Parameters:
    - data: DataFrame
    - percentage_columns: List of columns to transform
    - percentage_mapping: Dictionary mapping of percentage ranges to values

    Returns: DataFrame
    """
    for column in percentage_columns:
        data[column] = data[column].map(percentage_mapping)
    return data

def impute_maths_combined(row):
    if row["Math"] >= 3:
        return 2
    elif row["Mathlit"] >= 3:
        return 1
    else:
        return 0

def set_education_quality(row):
    if row["Matric"] == 1:
        return 1 if row["Schoolquintile"] >= 4 else 0
    return row.get("education_quality", 0)

def bin_column(data, column_name, bins, labels):
    """
    Bins a column based on given bins and labels.

    Parameters:
    - data: DataFrame
    - column_name: Name of column to bin
    - bins: List of bin edges
    - labels: List of labels for the bins

    Returns: DataFrame
    """
    new_column_name = f"{column_name}_group"
    data[new_column_name] = pd.cut(
        data[column_name], bins=bins, labels=labels, right=False
    )
    return data

def compute_aggregated_target_by_group(
    data, group_column, target_column, operation="mean", test_data=None
):
    """
    Computes the aggregated target by a given group column based on the specified operation.

    Parameters:
    - data: DataFrame (Training data)
    - group_column: Name of column to group by
    - target_column: Name of target column
    - operation: Aggregation operation ('min', 'max', 'mean', 'mode')
    - test_data: Optional DataFrame (Test data)

    Returns: Dictionary
    """

    if test_data is not None:
        data = pd.concat([data, test_data], ignore_index=True)

    if operation == "mean":
        return data.groupby(group_column)[target_column].mean().to_dict()
    elif operation == "min":
        return data.groupby(group_column)[target_column].min().to_dict()
    elif operation == "max":
        return data.groupby(group_column)[target_column].max().to_dict()
    elif operation == "mode":
        # mode() returns a Series, so we get the first value
        return (
            data.groupby(group_column)[target_column]
            .agg(lambda x: x.mode().iloc[0])
            .to_dict()
        )
    else:
        raise ValueError(
            f"Invalid operation: {operation}. Choose 'min', 'max', 'mean', or 'mode'."
        )

def create_interactions(data, interactions):
    """
    Creates interaction terms between two columns.

    Parameters:
    - data: DataFrame
    - interactions: Dictionary where keys and values are column names to create interactions for.

    Returns: DataFrame with added interaction terms.
    """
    for col1, col2 in interactions.items():
        # Check number of unique values in the columns
        if len(data[col1].unique()) > 10 or len(data[col2].unique()) > 10:
            print(
                f"Skipping interaction for {col1} and {col2} due to high cardinality."
            )
            continue

        # One-hot encode the columns
        col1_dummies = pd.get_dummies(data[col1], prefix=col1)
        col2_dummies = pd.get_dummies(data[col2], prefix=col2)

        # Create interaction terms
        for col1_dummy in col1_dummies.columns:
            for col2_dummy in col2_dummies.columns:
                interaction_col_name = f"{col1_dummy}_x_{col2_dummy}"
                data[interaction_col_name] = (
                    col1_dummies[col1_dummy] * col2_dummies[col2_dummy]
                )

    return data

def impute_column_with_regressor(
    train_data, test_data, column_to_impute, excluded_columns=[], regressor=None
):
    """
    Imputes missing values of a specified column using a regressor (default is XGBoost).

    Parameters:
    - train_data: Training data DataFrame.
    - test_data: Testing data DataFrame.
    - column_to_impute: The column for which missing values need to be imputed.
    - excluded_columns: List of columns to be excluded from the feature set for training the imputer.
    - regressor: Regressor instance. If None, default XGBoost regressor will be used.

    Returns:
    - Imputed train_data and test_data DataFrames.
    """
    # Features for prediction (excluding non-numeric columns and excluded columns)

    columns_to_drop = get_common_features(train_data, test_data, "dropped")
    print(columns_to_drop)

    features = (
        train_data.drop(columns=[column_to_impute] + excluded_columns + columns_to_drop)
        .select_dtypes(include=["int64", "float64"])
        .columns
    )

    # If no regressor provided, initialize the default one
    if regressor is None:
        regressor = xgboost.XGBRegressor(
            n_estimators=100, objective="reg:squarederror", random_state=42
        )

    # Impute train_data
    data_known_train = train_data.dropna(subset=[column_to_impute])
    data_unknown_train = train_data[train_data[column_to_impute].isnull()]
    if not data_unknown_train.empty:
        regressor.fit(data_known_train[features], data_known_train[column_to_impute])
        predicted_values_train = regressor.predict(data_unknown_train[features])
        data_unknown_train.loc[:, column_to_impute] = predicted_values_train
        train_data = pd.concat(
            [data_known_train, data_unknown_train], axis=0
        ).sort_index()

    # Impute test_data using the regressor trained on the entire train_data
    data_known_test = test_data.dropna(subset=[column_to_impute])
    data_unknown_test = test_data[test_data[column_to_impute].isnull()]
    if not data_unknown_test.empty:
        regressor.fit(train_data[features], train_data[column_to_impute])
        predicted_values_test = regressor.predict(data_unknown_test[features])
        data_unknown_test.loc[:, column_to_impute] = predicted_values_test
        test_data = pd.concat([data_known_test, data_unknown_test], axis=0).sort_index()

    return train_data, test_data

def impute_missing_value(data, column, strategy="min"):
    """
    Impute missing values for a specified column using either min or mode.

    Parameters:
    - data: DataFrame to impute.
    - column: The column name for which missing values should be imputed.
    - strategy: Either "min" or "mode".

    Returns:
    - DataFrame with imputed values for the specified column.
    """
    if strategy == "min":
        fill_value = data[column].min()
    elif strategy == "mode":
        fill_value = data[column].mode()[0]
    else:
        raise ValueError(f"Invalid strategy: {strategy}. Choose 'min' or 'mode'.")

    data[column].fillna(fill_value, inplace=True)
    return data

def normalize_data(X_train, X_val=None, target_column=None):
    """
    Normalize features using StandardScaler.

    Parameters:
    - X_train: Training features.
    - X_val: Optional validation features.
    - target_column: Name of the target column to exclude from normalization.

    Returns:
    - Scaled training and validation features.
    """
    scaler = StandardScaler()

    # Identify numeric columns in the dataframe
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

    # If target column is specified, remove it from the list of columns to normalize
    if target_column and target_column in numeric_cols:
        numeric_cols = numeric_cols.drop(target_column)

    # Scale only the numeric columns
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

    if X_val is not None:
        X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
        return X_train, X_val

    return X_train

def save_to_csv(data, filename):
    """
    Save DataFrame to CSV.

    Parameters:
    - data: DataFrame to save.
    - filename: The name of the CSV file.
    """
    data.to_csv(filename, index=False)

def drop_features_using_elasticnet(df, target_column):
    # 1. Split the data into training and test sets.
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 2. Normalize the data.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print("NaN values in y_train:", y_train.isnull().sum())
    print("NaN values in y_val:", y_val.isnull().sum())

    # 3. Use ElasticNetCV to determine the optimal alpha and l1_ratio.
    elasticnetcv = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, random_state=42,max_iter=10000,tol=0.0001)
    elasticnetcv.fit(X_train_scaled, y_train)
    
    y_pred_en = elasticnetcv.predict(X_val_scaled)
    mse_en = mean_squared_error(y_val, y_pred_en)

    features_to_drop = X.columns[elasticnetcv.coef_ == 0].tolist()

    print("MSE for Elastic Net:", mse_en, features_to_drop)
    
    return features_to_drop

def compute_vif(data, target_column, threshold=10.0, show=True):
    """
    Compute the Variance Inflation Factor (VIF) for the specified numerical columns and
    return features with VIF greater than the given threshold.

    Parameters:
    - data: DataFrame
    - target_column: Name of the target column.
    - threshold: VIF threshold for filtering features. Default is 10.0.

    Returns:
    - List of features with VIF values greater than the threshold.
    """
    # Drop the target column
    data = data.drop(columns=target_column, errors="ignore")

    # Select only numeric data and drop columns with constant values
    numeric_data = data.select_dtypes(include=[np.number])
    numeric_data = numeric_data.loc[:, numeric_data.var() != 0]

    # Drop rows with missing values for VIF computation
    numeric_data = numeric_data.dropna()

    # Compute VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = numeric_data.columns
    vif_data["VIF"] = [
        variance_inflation_factor(numeric_data.values, i)
        for i in range(numeric_data.shape[1])
    ]

    # Sort by VIF values in descending order and filter by threshold
    vif_data = vif_data.sort_values(by="VIF", ascending=False)
    if show:
        print(vif_data)
    high_vif_features = vif_data[vif_data["VIF"] > threshold]["Feature"].tolist()

    return high_vif_features

def drop_multicollinear_features(data, target_column, threshold=10.0):
    """
    Drop multicollinear features based on VIF.

    Parameters:
    - data: DataFrame
    - target_column: Name of the target column.
    - threshold: VIF threshold for filtering features. Default is 10.0.

    Returns:
    - DataFrame after dropping multicollinear features.
    """
    features_to_drop = []
    while True:
        high_vif_features = compute_vif(data, target_column, threshold)
        if not high_vif_features:
            break

        # Drop the feature with the highest VIF value
        drop_feature = high_vif_features[0]
        features_to_drop.append(drop_feature)
        data = data.drop(columns=[drop_feature])

    return features_to_drop

def print_dataframe_info(df):
    """
    Print basic information about a DataFrame.

    Parameters:
    - df: The input DataFrame.
    """
    print("Data Overview:")
    print("==============")

    # Shape of the dataframe
    print(f"Number of Rows: {df.shape[0]}")
    print(f"Number of Columns: {df.shape[1]}")
    print("\n")

    # Display datatypes and non-null counts
    print("Data Types and Non-Null Counts:")
    print(df.info())
    print("\n")

    # Check for NaN values
    nan_counts = df.isnull().sum()
    columns_with_nan = nan_counts[nan_counts > 0]
    if columns_with_nan.empty:
        print("The DataFrame has no missing values.")
    else:
        print("Columns with Missing Values:")
        print(columns_with_nan)
        print("\n")

        # Percentage of missing values
        print("Percentage of Missing Values:")
        missing_percentage = (columns_with_nan / df.shape[0]) * 100
        print(missing_percentage)
        print("\n")

    # Display first few rows
    print("First Few Rows of the DataFrame:")
    print(df.head())
    print("\n")

    # Display basic statistics
    print("Basic Statistics:")
    print(df.describe())
