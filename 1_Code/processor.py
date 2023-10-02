import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import functions as f


def set_schoolquintile_by_province(row, mapping):
    """
    Sets the value of Schoolquintile based on the mode of the Province.

    Parameters:
    - row: Row of the DataFrame.
    - mapping: Dictionary with Province as key and its corresponding mode of Schoolquintile as value.

    Returns:
    - Updated value for Schoolquintile.
    """
    if row["Matric"] in [0, 1]:
        return int(mapping[row["Province"]])
    else:
        return -1

def impute_maths_combined(row):
    if row["Math"]:
        return row["Math"] + 6
    elif row["Mathlit"]:
        return row["Mathlit"]
    else:
        return 0

def set_education_quality(row):
    if row["Matric"] == 1:
        return 1 if row["Schoolquintile"] >= 4 else 0
    return row.get(
        "education_quality", 0
    )  # default value if 'education_quality' doesn't exist

def set_education_quality(row):
    if row["Matric"] == 1:
        return 1 if row["Schoolquintile"] >= 4 else 0
    return row.get(
        "education_quality", 0
    )  # default value if 'education_quality' doesn't exist

def process_data(data: pd.DataFrame):
    # Convert "Survey_date" to datetime and extract features
    data["Survey_date"] = pd.to_datetime(data["Survey_date"])
    data["Survey_year"] = data["Survey_date"].dt.year
    data["Survey_month"] = data["Survey_date"].dt.month
    data["Survey_day"] = data["Survey_date"].dt.day

    data.drop(columns="Survey_date", inplace=True)

    # data["Schoolquintile"] = data.apply(
    #     lambda row: set_schoolquintile_by_province(
    #         row, mean_school_quintile_by_province
    #     ),
    #     axis=1,
    # )

    # data = f.bin_column(data=data,column_name='Tenure',[])
    

    interactions = {
        # "Province": "Geography",
        "Geography": "Status",
        "Status": "Province",
        # "Female":"Status",
        # "Geography": "Schoolquintile",
        # "Province": "Schoolquintile",
        # "Status": "Schoolquintile",
        # "Diploma": "Tenure",
        # "Degree": "Province",
    }

    # data = f.create_single_interaction(data,'Matric','Tenure')
    # data = f.create_single_interaction(data,'Degree','Tenure')
    # data = f.create_single_interaction(data,'Diploma','Tenure')

    # data = f.create_single_interaction(data,'Province','Matric')

    data = f.create_single_interaction(data,'Status','Tenure')
    # data = f.create_single_interaction(data,'Province','Age')
    # data = f.create_single_interaction(data,'Geography','Age')
    # data = f.create_single_interaction(data,'Status','Age')




    # data = f.create_interactions(data,interactions=interactions)
    # Calculate age
    data["Age"] = data.apply(f.calculate_exact_age, axis=1)
    # data = f.create_single_interaction(data,'Status','Age')
    # data = f.create_single_interaction(data,'Status','Age')

    # 5. Age Groups
    data["is_young_adult"] = (data["Age"] >= 18) & (data["Age"] <= 25).astype(int)
    data["is_middle_aged"] = (data["Age"] > 25) & (data["Age"] <= 30).astype(int)
    data["is_senior"] = (data["Age"] > 30).astype(int)
    data["Age_first_employment"] = data["Age"] - data["Tenure"] / 365.25
    data["Age_first_employment"] = data.apply(lambda row: row["Age_first_employment"] if row["Age_first_employment"] > 18 else 0,axis=1)

    # Transform percentage range columns
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
    data = f.transform_percentage_columns(data, percentage_columns, percentage_mapping)

    data["Maths_combined"] = data.apply(impute_maths_combined, axis=1)
    
    data["Mathadded"] = data["Math"] *2
    data["Maths_combined"] = f.flexible_add(data,'Mathlit','Mathadded')
    
    data.drop(columns=['Mathadded'],inplace=True)

    data["Subjects_passed"] = data[percentage_columns].apply(
        lambda x: sum(val >= 5 for val in x), axis=1
    )
    data["education_quality"] = data.apply(set_education_quality, axis=1)
    thresh = 5
    data["Math_Science_high_combo"] = (
        (data["Math"] >= thresh) & (data["Science"] >= thresh)
    ).astype(int)
    data["Subjects_polarity"] = data[
        ["Math", "Science", "Additional_lang", "Home_lang"]
    ].apply(lambda x: sum(val >= 5 for val in x) - sum(val <= 1 for val in x), axis=1)
    # 6. School and Education Interactions
    for subject in ["Maths_combined"]:
        data[f"{subject}_schoolquintile"] = data[subject] * data["Schoolquintile"]

    data["downturn"] = data["Round"].apply(lambda x: 1 if x in [4,2] else 0)

    # Map "unemployed" status to -1, and all other statuses to 1
    data["Status_mapped"] = data["Status"].apply(
        lambda x: -1 if x == "unemployed" else 1
    )
    data["Tenure"] = data.apply(
        lambda row: 0 if row["Status"] == "studying" else row["Tenure"], axis=1
    )
    # Calculate the exposure based on subjects
    data["Exposure"] = data[percentage_columns].apply(
        lambda x: sum((val >= 0) * 2 - pd.isna(val) for val in x), axis=1
    )
    # Increase exposure by 10 if they have a degree
    # Assuming the column is named 'Degree' and has a value 'Yes' for those with a degree
    data["Exposure"] = data.apply(
        lambda row: row["Exposure"] + 10
        if row.get("Degree") == "Yes"
        else row["Exposure"],
        axis=1,
    )
    # 10. Total Achievements
    data["Total_achievements"] = data[
        ["Math", "Science", "Additional_lang", "Home_lang"]
    ].sum(axis=1)

    data = f.encode_categorical_columns(
        data, ["Status", "Geography", "Province"], encoding_type="onehot"
    )

    # Modify the Tenure value based on the Status
    data["Tenure_Status"] = data.apply(
        lambda row: (
            row["Tenure"]*-1 if row["Status_mapped"] == -1 else row["Tenure"]
        )
        * row["Status_mapped"],
        axis=1,
    )
    # Modify the Tenure value based on the Status
    data["Tenure^2_Status"] = data.apply(
        lambda row: (
            row["Tenure"]**2*-1 if row["Status_mapped"] == -1 else row["Tenure"]**2),
        axis=1)

    # data["Math_Science_interaction"] = data["Math"] * data["Science"]
    return data


# Load the datasets
train_data = pd.read_csv("TrainTest.csv")
test_data = pd.read_csv("TestTest.csv")



# train_data, test_data = f.impute_column_with_model(train_data, test_data, "Tenure",excluded_columns=['Target'],n_neighbors=5)
# train_data = f.impute_missing_value(data=train_data,column='Tenure',strategy='mode')
# test_data = f.impute_missing_value(data=test_data,column='Tenure',strategy='mode')

# train_data,test_data = f.impute_column_with_knn(train_data=train_data,test_data=test_data,column_to_impute='Tenure',excluded_columns=['Target'],n_neighbors=5)
mean_tenure_by_status_province = f.aggregate_by_group(train_data,target_column='Tenure',strategy='mean',group_columns=['Status'],second_data=test_data)
f.print_dataframe_info(mean_tenure_by_status_province)
print(mean_tenure_by_status_province)


mean_school_quintile_by_province = f.aggregate_by_group(train_data,'Schoolquintile','mode',['Province','Geography'],second_data=test_data)
print(mean_school_quintile_by_province)

train_data,test_data = f.set_value_by_group(train_data,'Schoolquintile',strategy='mode',group_columns=['Province','Geography'],second_data=test_data)

# data = f.create_and_select_interactions(train_data,'Target',[('Status','Geography')])

#TODO: Fix function
# train_data = f.create_and_select_interactions(train_data,'Target',[('Status','Province')],alpha=0.001)

# Process the data
train_proc = process_data(train_data)
test_proc = process_data(test_data)



# train_proc = f.apply_feature_hashing(train_proc,'Tenure',8)

# mean_target_by_round = f.aggregate_by_group(data=train_data,target_column='Round',group_columns=['Target'],strategy='mode')

# print(mean_target_by_round)

f.print_dataframe_info(train_proc)

# train_proc["Round"] = train_proc["Round"].map(mean_target_by_round)
# test_proc["Round"] = test_proc["Round"].map(mean_target_by_round)
# Handling Missing Values
numerical_columns = train_proc.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()
categorical_columns = train_proc.select_dtypes(include=["object"]).columns.tolist()
numerical_columns.remove("Target")
categorical_columns.remove("Person_id")

# For numerical columns, fill missing values with the minimum
for col in numerical_columns:
    min_value = train_proc[col].min()
    train_proc[col].fillna(min_value, inplace=True)
    test_proc[col].fillna(min_value, inplace=True)

# For categorical columns, fill missing values with the mode
for col in categorical_columns:
    mode_value = train_proc[col].mode()[0]
    train_proc[col].fillna(mode_value, inplace=True)
    test_proc[col].fillna(mode_value, inplace=True)


# Feature selection
dropped_features = f.drop_features_using_elasticnet(
    train_proc.drop(columns=["Person_id"]), "Target"
)

# dropped_features = f.enhanced_feature_selection_drop(train_proc.drop(columns=['Person_id']),'Target',method='variance_threshold',k_features=30)

# print(dropped_features)

# # Assuming you will do the same for the test set:
# test_proc.drop(columns=dropped_features, inplace=True)
# train_proc.drop(columns=dropped_features, inplace=True)

# train_norm, test_norm = f.normalize_data(X_train=train_proc,X_val=test_proc,target_column='Target')

# # print(f.compute_vif(train_proc, "Target"))

# # f.print_dataframe_info()

f.save_to_csv(train_proc, "processed_train.csv")
f.save_to_csv(test_proc, "processed_test.csv")
