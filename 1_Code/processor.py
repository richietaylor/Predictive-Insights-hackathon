import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA


# Function to encode categorical variables
def encode_categorical_columns(data):
    categorical_columns = ["Status", "Geography", "Province"]
    for column in categorical_columns:
        encoder = LabelEncoder()
        data[column] = encoder.fit_transform(data[column])
    return data


def encode_categorical_columns_with_ordinal(data):
    categorical_columns = ["Status", "Geography", "Province"]
    encoder = OrdinalEncoder()
    data[categorical_columns] = encoder.fit_transform(data[categorical_columns])
    return data


def calculate_exact_age(row):
    age = row["Survey_year"] - row["Birthyear"]
    # Adjusting the age based on months
    age += (row["Survey_month"] - row["Birthmonth"]) / 12
    return age


# Function to transform percentage range columns
def transform_percentage_columns(data):
    percentage_columns = ["Math", "Mathlit", "Additional_lang", "Home_lang", "Science"]
    percentage_mapping = {
        "0 - 29 %": 0,
        "30 - 39 %": 1,
        "40 - 49 %": 2,
        "50 - 59 %": 3,
        "60 - 69 %": 4,
        "70 - 79 %": 5,
        "80 - 100 %": 6,
    }
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
    return row.get(
        "education_quality", 0
    )  # default value if 'education_quality' doesn't exist


def bin_age(data):
    age_bins = [18, 25, 35, 45, 55, float("inf")]
    # age_labels = ["18-25", "26-35", "36-45", "46-55", "56+"]
    age_labels = [0, 1, 2, 3, 4]

    data["Age_group"] = pd.cut(
        data["Age"], bins=age_bins, labels=age_labels, right=False
    )
    return data


def bin_tenure(data):
    # Converting years to days for the bins
    tenure_bins = [0, 365, 3 * 365, 5 * 365, 10 * 365, float("inf")]
    # tenure_labels = ["0-1 years", "1-3 years", "3-5 years", "5-10 years", "10+ years"]
    tenure_labels = [0, 1, 2, 3, 4]

    data["Tenure_group"] = pd.cut(
        data["Tenure"], bins=tenure_bins, labels=tenure_labels, right=False
    )
    return data


def compute_mean_target_by_round(train_data):
    return train_data.groupby("Round")["Target"].mean().to_dict()


def encode_categorical_columns_with_onehot(data):
    categorical_columns = ["Status", "Geography", "Province"]
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=False)
    return data


# Function to process the data
def process_data(data, mean_target_by_round=None):
    # Convert "Survey_date" to datetime and extract features
    data["Survey_date"] = pd.to_datetime(data["Survey_date"])
    data["Survey_year"] = data["Survey_date"].dt.year
    data["Survey_month"] = data["Survey_date"].dt.month
    data["Survey_day"] = data["Survey_date"].dt.day

    interactions = {
        "Province": "Geography",
        "Geography": "Status",
        "Status": "Province",
        "Status": "Tenure",
        "Sa_citizen": "Additional_lang",
        # "Geography": "Schoolquintile",
        # "Province": "Schoolquintile",
        # "Diploma": "Tenure",
        # "Degree": "Tenure",
    }
    for x in interactions:
        print(x, "starting")
        province_dummies = pd.get_dummies(data[x], prefix=x)
        geography_dummies = pd.get_dummies(data[interactions[x]], prefix=interactions[x])

        # Create interaction terms
        for province_col in province_dummies.columns:
            for geo_col in geography_dummies.columns:
                interaction_col_name = f"{province_col}_x_{geo_col}"
                data[interaction_col_name] = (
                    province_dummies[province_col] * geography_dummies[geo_col]
                )


    # interactions = {
    #     "Province": ["Geography"],
    #     "Geography": ["Status", "Schoolquintile"],
    #     "Status": ["Province", "Tenure"],
    #     "Sa_citizen": ["Additional_lang"],
    #     "Diploma": ["Tenure"],
    #     "Degree": ["Tenure"],
    # }

    # # We use pandas .get_dummies only once for each unique column in interactions
    # dummies = {col: pd.get_dummies(data[col], prefix=col) for col in set(interactions.keys()).union(*interactions.values())}

    # for col, interact_with in interactions.items():
    #     for interact_col in interact_with:
    #         # Vectorized operation for creating interaction terms
    #         interaction_matrix = dummies[col].values[:, :, None] * dummies[interact_col].values[:, None, :]
            
    #         # Efficient memory usage by converting boolean matrix to uint8
    #         interaction_matrix = interaction_matrix.astype('uint8')
            
    #         # Constructing column names and creating new DataFrame for interactions
    #         columns = [f"{col_name}_x_{interact_col_name}" for col_name in dummies[col].columns for interact_col_name in dummies[interact_col].columns]
    #         interaction_df = pd.DataFrame(interaction_matrix.reshape(len(data), -1), columns=columns, index=data.index)
            
    #         # Concatenating the interaction DataFrame with the original DataFrame
    #         data = pd.concat([data, interaction_df], axis=1)



# TOO SLOW 

#     interactionOne = [
#     "Round", "Status", "Tenure", "Geography", "Province",
#     "Matric", "Degree", "Diploma", "Schoolquintile",
#     "Math", "Mathlit", "Additional_lang", "Home_lang",
#     "Science", "Female", "Sa_citizen", "Birthyear", 
#     "Birthmonth", "Target"
# ]
#     interactionTwo = [
#     "Round", "Status", "Tenure", "Geography", "Province",
#     "Matric", "Degree", "Diploma", "Schoolquintile",
#     "Math", "Mathlit", "Additional_lang", "Home_lang",
#     "Science", "Female", "Sa_citizen", "Birthyear", 
#     "Birthmonth", "Target"
# ]

#     for y in interactionOne:
#         for x in interactionTwo:
#             province_dummies = pd.get_dummies(data[y], prefix=y)
#             geography_dummies = pd.get_dummies(data[x], prefix=x)

#             # Create interaction terms
#             for province_col in province_dummies.columns:
#                 for geo_col in geography_dummies.columns:
#                     interaction_col_name = f"{province_col}_x_{geo_col}"
#                     data[interaction_col_name] = (
#                         province_dummies[province_col] * geography_dummies[geo_col]
#                     )

    # Calculate age
    data["Age"] = data.apply(calculate_exact_age, axis=1)

    # Map "unemployed" status to -1, and all other statuses to 1
    data["Status_mapped"] = data["Status"].apply(
        lambda x: -1 if x == "unemployed" else 1
    )

    # Modify the Tenure value based on the Status
    data["Tenure_Status"] = data.apply(
        lambda row: (
            row["Tenure"] ** 2 if row["Status_mapped"] == -1 else row["Tenure"]
        )
        * row["Status_mapped"],
        axis=1,
    )

    # Transform percentage range columns
    data = transform_percentage_columns(data)

    data['Maths_combined'] = data.apply(impute_maths_combined, axis=1)
    # Count the number of subjects passed
    percentage_columns = ["Math", "Mathlit", "Additional_lang", "Home_lang", "Science"]
    data["Subjects_passed"] = data[percentage_columns].apply(
        lambda x: sum(val >= 5 for val in x), axis=1
    )

    # Calculate the exposure based on subjects
    data["Exposure"] = data[percentage_columns].apply(
        lambda x: sum((val >= 0) * 2 - pd.isna(val) for val in x), axis=1
    )

    # Create 'education_quality' column based on 'Schoolquintile' only if 'matric' is 1
    data["education_quality"] = data.apply(set_education_quality, axis=1)

    # Increase exposure by 10 if they have a degree
    # Assuming the column is named 'Degree' and has a value 'Yes' for those with a degree
    data["Exposure"] = data.apply(
        lambda row: row["Exposure"] + 10
        if row.get("Degree") == "Yes"
        else row["Exposure"],
        axis=1,
    )

    # # Fill missing Tenure values

    # Add the 'downturn' column based on the 'Round' value
    data["downturn"] = data["Round"].apply(lambda x: 1 if x in [1, 3] else 0)

    ###### MAYBE REMOVE

    # 1. Age at First Employment
    data["Age_first_employment"] = data["Age"] - data["Tenure"] / 365.25

    # 2. Interaction Features
    data["Math_Science_interaction"] = data["Math"] * data["Science"]
    # data['Geography_Educ_interaction'] = data['Geography'] * data['educ']

    # Apply ordinal encoding
    data = encode_categorical_columns_with_onehot(data)

    # 3. Province-based Features
    # # These are more suited for train data to avoid data leakage, but we'll add them here for simplicity
    # avg_tenure_per_province = data.groupby("Province")["Tenure"].mean().to_dict()
    # avg_age_per_province = data.groupby("Province")["Age"].mean().to_dict()
    # data["Avg_tenure_province"] = data["Province"].map(avg_tenure_per_province)
    # data["Avg_age_province"] = data["Province"].map(avg_age_per_province)

    # 5. Age Groups
    data["is_young_adult"] = (data["Age"] >= 18) & (data["Age"] <= 25).astype(int)
    data["is_middle_aged"] = (data["Age"] > 25) & (data["Age"] <= 45).astype(int)
    data["is_senior"] = (data["Age"] > 45).astype(int)

    # 6. School and Education Interactions
    for subject in ["Math", "Science"]:
        data[f"{subject}_schoolquintile"] = data[subject] * data["Schoolquintile"]

    # 7. Polarity of Subjects Passed
    data["Subjects_polarity"] = data[
        ["Math", "Science", "Additional_lang", "Home_lang"]
    ].apply(lambda x: sum(val >= 5 for val in x) - sum(val <= 1 for val in x), axis=1)

    # 9. Math-Science Combo
    threshold = 5  # Assuming scores above 5 as high
    data["Math_Science_high_combo"] = (
        (data["Math"] >= threshold) & (data["Science"] >= threshold)
    ).astype(int)

    # 10. Total Achievements
    data["Total_achievements"] = data[
        ["Math", "Science", "Additional_lang", "Home_lang"]
    ].sum(axis=1)

    # data = bin_age(data)
    # data = bin_tenure(data)
    # ########## MABYE REMOVE

    # Polynomial Features
    # data["Math^2"] = data["Math"] ** 2
    # data["Science^2"] = data["Science"] ** 2
    # data["Math_x_Science"] = data["Math"] * data["Science"]  # interaction term
    # data["Additional_lang^2"] = data["Additional_lang"] ** 2
    # Dropping Columns
    data.drop(
        columns=[
            "Survey_date",
        ],
        inplace=True,
    )
    # If mean_target_by_round is provided, map it to the 'Round' column
    if mean_target_by_round:
        data["Mean_Target_By_Round"] = data["Round"].map(mean_target_by_round)
    else:
        # If not provided (i.e., for the training data), compute it directly
        data["Mean_Target_By_Round"] = data.groupby("Round")["Target"].transform("mean")

    return data


# Load the datasets
train_data = pd.read_csv("TrainTest.csv")
test_data = pd.read_csv("TestTest.csv")

import xgboost as xgb

def impute_tenure_with_xgboost(train_data, test_data):
    # Ensure that the test_data doesn't have the 'Target' column
    test_data = test_data.drop(columns=['Target'], errors='ignore')
    
    # Concatenate train and test data
    combined_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
    
    # Split data into sets with known and unknown tenure values
    data_known_tenure = combined_data.dropna(subset=['Tenure'])
    data_unknown_tenure = combined_data[combined_data['Tenure'].isnull()]
    
    # Features for prediction (excluding non-numeric columns and target columns)
    features = combined_data.drop(columns=['Tenure', 'Person_id', 'Target']).select_dtypes(include=["int64", "float64"]).columns
    
    # Train an XGBoost regressor
    xgb_regressor = xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state=42)
    xgb_regressor.fit(data_known_tenure[features], data_known_tenure['Tenure'])
    
    # Predict missing tenure values
    predicted_tenure = xgb_regressor.predict(data_unknown_tenure[features])
    
    # Replace missing values with predicted values using .loc to avoid warnings
    data_unknown_tenure.loc[:, 'Tenure'] = predicted_tenure
    
    # Concatenate data back together
    imputed_data = pd.concat([data_known_tenure, data_unknown_tenure], axis=0).sort_index()
    
    # Separate back into original train and test sets
    imputed_train_data = imputed_data.iloc[:len(train_data)]
    imputed_test_data = imputed_data.iloc[len(train_data):].drop(columns=['Target'], errors='ignore')
    
    return imputed_train_data, imputed_test_data

# Impute tenure values in the train and test data
imputed_train_data, imputed_test_data = impute_tenure_with_xgboost(train_data, test_data)  # Using df_new twice as a placeholder, replace with actual test data when available

# Check if missing values in tenure have been imputed
missing_after_imputation = imputed_train_data['Tenure'].isnull().sum()
print(missing_after_imputation)

train_data = imputed_train_data
test_data = imputed_test_data

# Process both training and test data
mean_target_by_round = compute_mean_target_by_round(train_data)
train_data = process_data(train_data)
test_data = process_data(test_data, mean_target_by_round=mean_target_by_round)

# Handling Missing Values
numerical_columns = train_data.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()
categorical_columns = train_data.select_dtypes(include=["object"]).columns.tolist()
numerical_columns.remove("Target")
categorical_columns.remove("Person_id")

# For numerical columns, fill missing values with the minimum
for col in numerical_columns:
    min_value = train_data[col].min()
    train_data[col].fillna(min_value, inplace=True)
    test_data[col].fillna(min_value, inplace=True)

# For categorical columns, fill missing values with the mode
for col in categorical_columns:
    mode_value = train_data[col].mode()[0]
    train_data[col].fillna(mode_value, inplace=True)
    test_data[col].fillna(mode_value, inplace=True)


# One-Hot Encoding
train_data = pd.get_dummies(train_data, columns=categorical_columns[1:])
test_data = pd.get_dummies(test_data, columns=categorical_columns[1:])
common_features = list(set(train_data.columns) & set(test_data.columns))
common_features.remove("Person_id")
train_data = train_data[["Person_id"] + common_features + ["Target"]]
test_data = test_data[["Person_id"] + common_features]


# Dropping columns with NaN values for VIF computation
filtered_data = train_data[numerical_columns].dropna(axis=1)

vif_data = pd.DataFrame()
vif_data["Feature"] = filtered_data.columns
vif_data["VIF"] = [
    variance_inflation_factor(filtered_data.values, i)
    for i in range(filtered_data.shape[1])
]

print(vif_data.sort_values(by="VIF", ascending=False))


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Splitting the data
X = train_data.drop(columns=["Target", "Person_id"])
y = train_data["Target"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# Normalize the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# # Applying L1 regularization using LassoCV
# lasso = LassoCV(cv=5, random_state=42)
# lasso.fit(X_train_scaled, y_train)

# # Predicting on validation set
# y_pred = lasso.predict(X_val_scaled)
# mse = mean_squared_error(y_val, y_pred)

# # Checking features that have been dropped by Lasso (coefficient = 0)
# dropped_features = X.columns[lasso.coef_ == 0].tolist()


# print("MSE:", mse, dropped_features)

# train_data.drop(
#     columns=dropped_features,
#     inplace=True,
# )
# test_data.drop(
#     columns=dropped_features,
#     inplace=True,
# )

# # Save processed datasets to CSV files
# train_data.to_csv("processed_train.csv", index=False)
# test_data.to_csv("processed_test.csv", index=False)


print("NaN values in y_train:", y_train.isnull().sum())
print("NaN values in y_val:", y_val.isnull().sum())

from sklearn.linear_model import ElasticNetCV

# Applying Elastic Net with cross-validation
elastic_net = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
    cv=5,
    random_state=42,
    max_iter=10000,
    tol=0.0001,
)
elastic_net.fit(X_train_scaled, y_train)

# Predicting on validation set
y_pred_en = elastic_net.predict(X_val_scaled)
mse_en = mean_squared_error(y_val, y_pred_en)

# Checking features that have been dropped by Elastic Net (coefficient = 0)
dropped_features_en = X.columns[elastic_net.coef_ == 0].tolist()

print("MSE for Elastic Net:", mse_en, dropped_features_en)

train_data.drop(columns=dropped_features_en, inplace=True)
test_data.drop(columns=dropped_features_en, inplace=True)

# Save processed datasets to CSV files
train_data.to_csv("processed_train.csv", index=False)
test_data.to_csv("processed_test.csv", index=False)

