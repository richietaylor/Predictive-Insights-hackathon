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
        "Female":"Degree",
        "Female": "Matric",
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

    # data = f.create_single_interaction(data,'Status','Tenure')
    # data = f.create_single_interaction(data,'Province','Age')
    # data = f.create_single_interaction(data,'Geography','Age')
    # data = f.create_single_interaction(data,'Status','Age')




    data = f.create_interactions(data,interactions=interactions)
    # Calculate age
    data["Age"] = data.apply(f.calculate_exact_age, axis=1)

    # data = f.create_single_interaction(data,'Status','Age')
    data = f.create_single_interaction(data,'Status','Age')
    

    # 5. Age Groups
    data["is_young_adult"] = ((data["Age"] >= 18) & (data["Age"] <= 21)).astype(int)
    data["is_middle_aged"] = ((data["Age"] > 21) & (data["Age"] <= 30)).astype(int)
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
        lambda x: sum(val >= 3 for val in x), axis=1
    )
    data["education_quality"] = data.apply(set_education_quality, axis=1)
    thresh = 4
    data["Math_Science_high_combo"] = (
        (data["Math"] >= thresh) & (data["Science"] >= thresh)
    ).astype(int)
    data["Subjects_polarity"] = data[
        ["Math", "Science", "Additional_lang", "Home_lang"]
    ].apply(lambda x: sum(val >= 4 for val in x) - sum(val <= 2 for val in x), axis=1)
    


    data["downturn"] = data["Round"].apply(lambda x: 1 if x in [3] else 0)

    # Map "unemployed" status to -1, and all other statuses to 1
    data["unemployed"] = data["Status"].apply(
        lambda x: -1 if x == "unemployed" else 1
    )
    data["studying_other"] = data.apply(lambda row: 1 if row["Status"] in ["studying","other","self employed"] else 0,axis=1)
    # data["Tenure"] = data.apply(
    #     lambda row: 0 if row["Status"] == "studying" else row["Tenure"], axis=1
    # )
    # Calculate the exposure based on subjects
    data["Exposure"] = data[percentage_columns].apply(
        lambda x: sum((val >= 0) * 2 - pd.isna(val) for val in x), axis=1
    )
    # Increase exposure by 10 if they have a degree
    # Assuming the column is named 'Degree' and has a value 'Yes' for those with a degree
    data["Exposure"] = data.apply(
        lambda row: row["Exposure"]*5
        if row.get("Degree") == "Yes"
        else row["Exposure"],
        axis=1,
    )
    # 10. Total Achievements
    data["Total_achievements"] = data[
        ["Math", "Science", "Additional_lang", "Home_lang"]
    ].sum(axis=1)

    data = f.encode_categorical_columns(data,['Status'],encoding_type='ordinal')

    data = f.encode_categorical_columns(
        data, ["Geography", "Province"], encoding_type="ordinal"
    )

    # Modify the Tenure value based on the Status
    # data["Tenure_Status"] = data.apply(
    #     lambda row: (
    #         row["Tenure"]*-1 if row["unemployed"] == -1 else row["Tenure"]
    #     )
    #     * row["unemployed"],
    #     axis=1,
    # )
    # Modify the Tenure value based on the Status
    data["Tenure^2_Status"] = data.apply(
        lambda row: (
            row["Tenure"]**2*-1 if row["unemployed"] == -1 else row["Tenure"]**2),
        axis=1)

    # data["Math_Science_interaction"] = data["Math"] * data["Science"]
    return data


# Load the datasets
train_data = pd.read_csv("TrainTest.csv")
test_data = pd.read_csv("TestTest.csv")





# Process the data
train_proc = process_data(train_data)
test_proc = process_data(test_data)

mean_math_schoolquintile = f.aggregate_by_group(data=train_data,second_data=test_data,group_columns=['Schoolquintile'],strategy='mean',target_column='Maths_combined')

train_proc,test_proc = f.aggregate_by_group_v2(data=train_proc,target_column='Maths_combined',group_columns=['Province','Schoolquintile','Geography'],fill_strategy='difference',second_data=test_proc,strategy="mean")
train_proc,test_proc = f.aggregate_by_group_v2(data=train_proc,target_column='Age',group_columns=['Status','Schoolquintile'],fill_strategy='difference',second_data=test_proc,strategy="mean")

# train_proc,test_proc = f.impute_column_with_knn_v2(train_data=train_proc,test_data=test_proc,column_to_impute='Tenure',excluded_columns=['Target'],n_neighbors=10)
# train_proc["imputed_Tenure"] = train_proc.apply(lambda row: -0.1 if row["studying_other"] == 1  else row["imputed_Tenure"],axis=1)
# train_proc,test_proc = f.set_value_by_group(train_data,'Tenure',strategy='mean',group_columns=['Birthyear','Province','Geography'],second_data=test_data)
# train_proc,test_proc = f.set_value_by_group(train_data,'Matric',strategy='num',group_columns=[],num_value=-1,second_data=test_data)
# train_proc,test_proc = f.set_value_by_group(train_data,'Degree',strategy='num',group_columns=[],num_value=-1,second_data=test_data)
# train_proc,test_proc = f.set_value_by_group(train_data,'Diploma',strategy='num',group_columns=[],num_value=-1,second_data=test_data)
# train_proc,test_proc = f.set_value_by_group(train_data,'Schoolquintile',strategy='mean',group_columns=['Province','Geography'],second_data=test_data)

# train_proc['Schoolquintile'] = train_proc.apply(lambda row: None if row["Matric"] == None else row["Schoolquintile"],axis=1)
# test_proc['Schoolquintile'] = test_proc.apply(lambda row: None if row["Matric"] == None else row["Schoolquintile"],axis=1)


# train_proc = f.encode_categorical_columns(train_proc,['Status','Province','Geography'],encoding_type='label')
# test_proc = f.encode_categorical_columns(test_proc,['Status','Province','Geography'],encoding_type='label')


# data = f.create_and_select_interactions(train_data,'Target',[('Status','Geography')])

#TODO: Fix function
# train_data = f.create_and_select_interactions(train_data,'Target',[('Status','Province')],alpha=0.001)




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
# dropped_features = f.drop_features_using_elasticnet(
#     train_proc.drop(columns=["Person_id"]), "Target"
# )

dropped_features = f.enhanced_feature_selection_drop(train_proc.drop(columns=['Person_id']),'Target',method='mutual_information',k_features=30)

# print(dropped_features)

# # Assuming you will do the same for the test set:
test_proc.drop(columns=dropped_features, inplace=True)
train_proc.drop(columns=dropped_features, inplace=True)


pairs = f.find_collinear_pairs(train_proc, 0.9)


f.print_correlation_of_pairs_with_target(collinear_features=pairs,data=train_proc,target_column='Target')
# f.print_feature_target_correlation(train_proc,'Target')
# multicolinear = f.drop_multicollinear_features(train_proc,'Target',100)

# train_proc.drop(columns=multicolinear,inplace=True)
# test_proc.drop(columns=multicolinear,inplace=True)


# train_norm, test_norm = f.normalize_data(X_train=train_proc,X_val=test_proc,target_column='Target')

# # f.print_dataframe_info()

f.save_to_csv(train_proc, "processed_train.csv")
f.save_to_csv(test_proc, "processed_test.csv")
