from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, PolynomialFeatures
import pandas as pd
import xgboost
from sklearn.linear_model import (
    ElasticNetCV,
)
from sklearn.impute import KNNImputer

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, cross_val_predict
import numpy as np
from sklearn.feature_selection import (VarianceThreshold, SelectKBest, 
                                       f_classif, f_regression, mutual_info_classif, mutual_info_regression)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

def aggregate_by_group(data, target_column, strategy="mean", group_columns=[], second_data=None):
    """
    Aggregate a target column based on the provided strategy and grouping columns.

    Parameters:
    - data: The primary DataFrame to process.
    - target_column: The column to aggregate.
    - strategy: The aggregation strategy; can be 'mean', 'mode', 'min', or 'max'.
    - group_columns: The columns by which to group.
    - second_data: An optional second DataFrame to consider for the aggregation.

    Returns:
    - A DataFrame with the aggregated data.
    """
    
    # If a second DataFrame is provided, concatenate it with the primary one
    if second_data is not None:
        data = pd.concat([data, second_data], ignore_index=True)
    
    if strategy == "mean":
        return data.groupby(group_columns)[target_column].mean().reset_index()
    elif strategy == "mode":
        # Using first() to handle cases where mode returns multiple values
        return data.groupby(group_columns)[target_column].apply(lambda x: x.mode().iloc[0]).reset_index()
    elif strategy == "min":
        return data.groupby(group_columns)[target_column].min().reset_index()
    elif strategy == "max":
        return data.groupby(group_columns)[target_column].max().reset_index()
    else:
        raise ValueError(f"Invalid strategy: {strategy}. Choose from 'mean', 'mode', 'min', or 'max'.")

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

def create_single_interaction(data: pd.DataFrame, col_encode, col_multiply):
    """
    Creates interaction terms between a one-hot encoded column and another column.

    Parameters:
    - data: DataFrame
    - col_encode: Column name to be one-hot encoded.
    - col_multiply: Column name to be multiplied with the encoded columns.

    Returns: DataFrame with added interaction terms.
    """
    
    # Check number of unique values in the col_encode
    if len(data[col_encode].unique()) > 10:
        print(
            f"Skipping interaction for {col_encode} and {col_multiply} due to high cardinality in {col_encode}."
        )
        return data

    # One-hot encode the col_encode
    col_encode_dummies = pd.get_dummies(data[col_encode], prefix=col_encode)

    # Multiply one-hot encoded columns with col_multiply, ensuring NaN values in col_multiply don't get multiplied
    for col_encode_dummy in col_encode_dummies.columns:
        interaction_col_name = f"{col_encode_dummy}_x_{col_multiply}"
        data[interaction_col_name] = np.where(data[col_multiply].isna(), 1, col_encode_dummies[col_encode_dummy] * data[col_multiply])

    return data

def impute_column_with_regressor(
    train_data, test_data, column_to_impute, excluded_columns=[], 
    method='regressor', regressor=None, n_neighbors=5
):
    """
    Imputes missing values of a specified column using a regressor or KNN.

    Parameters:
    - train_data: Training data DataFrame.
    - test_data: Testing data DataFrame.
    - column_to_impute: The column for which missing values need to be imputed.
    - excluded_columns: List of columns to be excluded from the feature set for imputation.
    - method: Method for imputation ('regressor' or 'knn').
    - regressor: Regressor instance. If None and method is 'regressor', default XGBoost regressor will be used.
    - n_neighbors: Number of neighbors to use for KNN imputation.

    Returns:
    - Imputed train_data and test_data DataFrames.
    """
    
    # Dummy function for get_common_features, you might have a different implementation
    def get_common_features(train_data, test_data, mode="dropped"):
        return []
    
    # Features for prediction (excluding non-numeric columns and excluded columns)
    columns_to_drop = get_common_features(train_data, test_data, "dropped")
    features = (
        train_data.drop(columns=[column_to_impute] + excluded_columns + columns_to_drop)
        .select_dtypes(include=["int64", "float64"])
        .columns
    )

    if method == 'regressor':
        # If no regressor provided, initialize the default one
        if regressor is None:
            import xgboost
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

    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=n_neighbors)
        
        # Prepare data for imputation
        train_for_imputation = train_data[features].copy()
        test_for_imputation = test_data[features].copy()
        
        # Fit imputer on known train data and transform both train and test data
        train_imputed = imputer.fit_transform(train_for_imputation)
        test_imputed = imputer.transform(test_for_imputation)
        
        # Update the original dataframes with the imputed values
        train_data.loc[:, features] = train_imputed
        test_data.loc[:, features] = test_imputed

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

def enhanced_feature_selection_drop(df, target_column, problem_type="classification", method="variance_threshold", k_features=10):
    """
    Enhanced feature selection method.
    
    Parameters:
    - df: DataFrame with features and target.
    - target_column: Name of the target column in df.
    - problem_type: "classification" or "regression"
    - method: The feature selection method to use. Choices are:
              "variance_threshold", "univariate_test", "mutual_information", 
              "correlation_coefficient", "tree_importance", "lasso".
    - k_features: Number of top features to select for some methods.
    
    Returns:
    - List of dropped feature names.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # All features
    all_features = set(X.columns)
    
    # Variance Threshold
    if method == "variance_threshold":
        selector = VarianceThreshold()
        selector.fit(X)
        kept_features = X.columns[selector.get_support()]
        return list(all_features - set(kept_features))

    # Univariate Statistical Tests
    if method == "univariate_test":
        if problem_type == "classification":
            test_func = f_classif
        else:
            test_func = f_regression
        selector = SelectKBest(test_func, k=k_features).fit(X, y)
        kept_features = X.columns[selector.get_support()]
        return list(all_features - set(kept_features))

    # Mutual Information
    if method == "mutual_information":
        if problem_type == "classification":
            mi_func = mutual_info_classif
        else:
            mi_func = mutual_info_regression
        selector = SelectKBest(mi_func, k=k_features).fit(X, y)
        kept_features = X.columns[selector.get_support()]
        return list(all_features - set(kept_features))

    # Correlation Coefficient
    if method == "correlation_coefficient":
        corrs = X.corrwith(y)
        kept_features = corrs.nlargest(k_features).index
        return list(all_features - set(kept_features))

    # Feature Importance from Tree-based Models
    if method == "tree_importance":
        if problem_type == "classification":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-k_features:]
        kept_features = X.columns[top_indices]
        return list(all_features - set(kept_features))

    # Lasso for Feature Selection
    if method == "lasso":
        lasso = LassoCV(cv=5)
        lasso.fit(X, y)
        kept_features = X.columns[lasso.coef_ != 0]
        return list(all_features - set(kept_features))

    return []

def evaluate_and_compare_models(base_learners, X, y, n_splits=5):
    from sklearn.model_selection import StratifiedKFold
    
    # Initialize StratifiedKFold
    strat_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store aggregated predictions for all folds
    aggregated_predictions = {name: [] for name, _ in base_learners}

    # Store performance metrics for all models
    performance_metrics = {
        'Model': [],
        'AUC-ROC': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': []
    }

    # Iterate over each fold and train/evaluate models
    for train_index, val_index in strat_kfold.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        for name, model in base_learners:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            # Store metrics
            performance_metrics['Model'].append(name)
            performance_metrics['AUC-ROC'].append(roc_auc_score(y_val, y_pred_proba))
            performance_metrics['Accuracy'].append(accuracy_score(y_val, y_pred))
            performance_metrics['Precision'].append(precision_score(y_val, y_pred))
            performance_metrics['Recall'].append(recall_score(y_val, y_pred))
            performance_metrics['F1 Score'].append(f1_score(y_val, y_pred))

            # Aggregate predictions
            aggregated_predictions[name].extend(y_pred_proba)

    # Convert aggregated predictions to DataFrame
    df_aggregated_predictions = pd.DataFrame(aggregated_predictions)

    # Compute the correlation matrix
    corr_matrix = df_aggregated_predictions.corr()
    print("Model Similarity (Pearson Correlation Coefficients):")
    print(corr_matrix)
    
    # Convert performance metrics to DataFrame for better visualization
    df_performance = pd.DataFrame(performance_metrics)
    print("\nModel Performance Metrics (averaged over k-folds):")
    print(df_performance.groupby("Model").mean())

def create_and_select_interactions(data, target, interaction_pairs, degree=2, alpha=0.05):
    """
    Creates interaction terms and performs feature selection.

    Parameters:
    - data: DataFrame with original features.
    - target: Series with the target variable.
    - interaction_pairs: List of tuple pairs indicating which columns to interact.
    - degree: Degree for polynomial features (only for continuous interactions).
    - alpha: Regularization strength for Lasso.

    Returns: DataFrame with selected interaction terms.
    """
    original_cols = data.columns.tolist()
    
    # Create interaction terms
    for col1, col2 in interaction_pairs:
        if data[col1].dtype == 'object' or data[col2].dtype == 'object':
            # If at least one of the columns is categorical
            cat_col, cont_col = (col1, col2) if data[col1].dtype == 'object' else (col2, col1)
            cat_dummies = pd.get_dummies(data[cat_col], prefix=cat_col)
            for cat_dummy in cat_dummies.columns:
                data[f"{cat_dummy}_x_{cont_col}"] = cat_dummies[cat_dummy] * data[cont_col]
        else:
            # If both columns are continuous, create polynomial interactions
            poly = PolynomialFeatures(degree, include_bias=False)
            poly_data = poly.fit_transform(data[[col1, col2]])
            cols = [f"{col1}_{col2}_deg{d}" for d in range(1, degree + 1)]
            data = pd.concat([data, pd.DataFrame(poly_data, columns=cols, index=data.index)], axis=1)
            
    # Feature Selection using Lasso
    X = data.drop(columns=original_cols)  # Only interaction terms
    y = target

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Lasso regression for feature selection
    lasso = LassoCV(alphas=[alpha], cv=5, max_iter=10000)
    lasso.fit(X_scaled, y)

    # Select features where coefficient is non-zero
    selected_features = X.columns[lasso.coef_ != 0].tolist()

    # Retain only selected interaction terms
    data = data[original_cols + selected_features]

    return data

def apply_feature_hashing(data, column, n_features=8):
    """
    Applies feature hashing to a specified column.

    Parameters:
    - data: DataFrame
    - column: The name of the column to apply feature hashing.
    - n_features: The number of features in the output. Default is 8.

    Returns: DataFrame with the hashed features replacing the original column.
    """
    hasher = FeatureHasher(n_features=n_features, input_type='string')
    hashed_features = hasher.transform(data[column].astype(str))
    hashed_df = pd.DataFrame(hashed_features.toarray(), columns=[f"{column}_hash{idx}" for idx in range(n_features)])
    
    # Concatenate the hashed features with the original data and drop the original column
    data = pd.concat([data, hashed_df], axis=1).drop(column, axis=1)
    
    return data

def flexible_add(data, input1, input2):
    """
    Combine two inputs by adding their values together. The inputs can be two columns or one column and an integer.
    
    Parameters:
    - data: DataFrame containing the columns (if column names are provided as inputs).
    - input1, input2: Column names or an integer.

    Returns:
    - A Series containing the combined result.
    """
    
    # Check if input1 is a column name or an integer
    if isinstance(input1, str):
        val1 = data[input1].fillna(0)
    else:
        val1 = input1

    # Check if input2 is a column name or an integer
    if isinstance(input2, str):
        val2 = data[input2].fillna(0)
    else:
        val2 = input2

    # Combine the values
    combined = val1 + val2
    
    return combined

def set_value_by_group(data, target_column, group_columns, strategy="mean", second_data=None):
    """
    Modify a target column in the primary and optional second DataFrame based on aggregation 
    using provided strategy and grouping columns.

    Parameters:
    - data: The primary DataFrame to modify.
    - target_column: The column to modify.
    - group_columns: The columns by which to group.
    - strategy: The aggregation strategy; can be 'mean', 'mode', 'min', or 'max'.
    - second_data: An optional second DataFrame to modify based on the aggregation.

    Returns:
    - Tuple of Modified primary DataFrame and optionally the modified second DataFrame.
    """
    
    # Aggregate the data
    mapping_df = aggregate_by_group(data, target_column, strategy, group_columns, second_data)
    mapping = mapping_df.set_index(group_columns)[target_column].to_dict()
    
    # Define the function to apply to each row
    def modify_row(row):
        key = tuple(row[col] for col in group_columns)
        return mapping.get(key, row[target_column])

    # Modify the target column for the primary DataFrame
    data[target_column] = data.apply(modify_row, axis=1)
    
    if second_data is not None:
        # Modify the target column for the second DataFrame
        second_data[target_column] = second_data.apply(modify_row, axis=1)
        return data, second_data

    return data,
