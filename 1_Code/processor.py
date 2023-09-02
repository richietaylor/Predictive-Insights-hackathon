import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Load the datasets
train_data = pd.read_csv('TrainTest.csv')
test_data = pd.read_csv('TestTest.csv')

# Function to encode categorical variables
def encode_categorical_columns(data):
    categorical_columns = ['Status', 'Geography', 'Province']
    for column in categorical_columns:
        encoder = LabelEncoder()
        data[column] = encoder.fit_transform(data[column])
    return data

def encode_categorical_columns_with_ordinal(data):
    categorical_columns = ['Status', 'Geography', 'Province']
    encoder = OrdinalEncoder()
    data[categorical_columns] = encoder.fit_transform(data[categorical_columns])
    return data


def calculate_exact_age(row):
    age = row['Survey_year'] - row['Birthyear']
    # Adjusting the age based on months
    age += (row['Survey_month'] - row['Birthmonth']) / 12
    return age

# Function to transform percentage range columns
def transform_percentage_columns(data):
    percentage_columns = ['Math', 'Mathlit', 'Additional_lang', 'Home_lang', 'Science']
    percentage_mapping = {'0 - 29 %': 0, '30 - 39 %': 1, '40 - 49 %': 2, '50 - 59 %': 3, '60 - 69 %': 4, '70 - 79 %': 5, '80 - 100 %': 6}
    for column in percentage_columns:
        data[column] = data[column].map(percentage_mapping)
    return data

def set_education_quality(row):
    if row['Matric'] == 1:
        return 1 if row['Schoolquintile'] >= 4 else 0
    return row.get('education_quality', 0)  # default value if 'education_quality' doesn't exist

def bin_age(data):
    age_bins = [18, 25, 35, 45, 55, float('inf')]
    age_labels = ['18-25', '26-35', '36-45', '46-55', '56+']
    data['Age_group'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels, right=False)
    return data

def bin_tenure(data):
    tenure_bins = [0, 1, 3, 5, 10, float('inf')]
    tenure_labels = ['0-1 years', '1-3 years', '3-5 years', '5-10 years', '10+ years']
    data['Tenure_group'] = pd.cut(data['Tenure'], bins=tenure_bins, labels=tenure_labels, right=False)
    return data

def process_data_with_binning(data):
    data = process_data(data)
    data = bin_age(data)
    data = bin_tenure(data)
    return data

# Function to process the data
def process_data(data):
    # Convert "Survey_date" to datetime and extract features
    data['Survey_date'] = pd.to_datetime(data['Survey_date'])
    data['Survey_year'] = data['Survey_date'].dt.year
    data['Survey_month'] = data['Survey_date'].dt.month
    data['Survey_day'] = data['Survey_date'].dt.day

    # Calculate age
    data['Age'] = data.apply(calculate_exact_age, axis=1)
    
    # Map "unemployed" status to -1, and all other statuses to 1
    data['Status_mapped'] = data['Status'].apply(lambda x: -1 if x == 'unemployed' else 1)

    # Create new feature by multiplying 'Tenure' by 'Status_mapped'
    data['Tenure_Status'] = data['Tenure'] * data['Status_mapped']
    
    # Apply ordinal encoding
    data = encode_categorical_columns_with_ordinal(data)

    # Transform percentage range columns
    data = transform_percentage_columns(data)

    # Count the number of subjects passed
    percentage_columns = ['Math', 'Mathlit', 'Additional_lang', 'Home_lang', 'Science']
    data['Subjects_passed'] = data[percentage_columns].apply(lambda x: sum(val >= 5 for val in x), axis=1)

    # Calculate the exposure based on subjects
    data['Exposure'] = data[percentage_columns].apply(lambda x: sum((val >= 0)*2 - pd.isna(val) for val in x), axis=1)

    # Create 'education_quality' column based on 'Schoolquintile' only if 'matric' is 1
    data['education_quality'] = data.apply(set_education_quality, axis=1)

    # Increase exposure by 10 if they have a degree
    # Assuming the column is named 'Degree' and has a value 'Yes' for those with a degree
    data['Exposure'] = data.apply(lambda row: row['Exposure'] + 10 if row.get('Degree') == 'Yes' else row['Exposure'], axis=1)

    # Fill missing Tenure values
    average_tenure_by_province = data.groupby('Province')['Tenure'].mean()
    data['Tenure'] = data.apply(lambda row: average_tenure_by_province[row['Province']] if pd.isnull(row['Tenure']) else row['Tenure'], axis=1)

    # Add the 'downturn' column based on the 'Round' value
    # data['downturn'] = data['Round'].apply(lambda x: 1 if x in [2, 4] else 0)


     # 1. Age at First Employment
    data['Age_first_employment'] = data['Age'] - data['Tenure']/365.25

    # 2. Interaction Features
    data['Math_Science_interaction'] = data['Math'] * data['Science']
    # data['Geography_Educ_interaction'] = data['Geography'] * data['educ']
    
    # 3. Province-based Features
    # These are more suited for train data to avoid data leakage, but we'll add them here for simplicity
    avg_tenure_per_province = data.groupby('Province')['Tenure'].mean().to_dict()
    avg_age_per_province = data.groupby('Province')['Age'].mean().to_dict()
    data['Avg_tenure_province'] = data['Province'].map(avg_tenure_per_province)
    data['Avg_age_province'] = data['Province'].map(avg_age_per_province)
    
    # 5. Age Groups
    data['is_young_adult'] = (data['Age'] >= 18) & (data['Age'] <= 25).astype(int)
    data['is_middle_aged'] = (data['Age'] > 25) & (data['Age'] <= 45).astype(int)
    data['is_senior'] = (data['Age'] > 45).astype(int)
    
    # 6. School and Education Interactions
    for subject in ['Math', 'Science']:
        data[f'{subject}_schoolquintile'] = data[subject] * data['Schoolquintile']
    
    # 7. Polarity of Subjects Passed
    data['Subjects_polarity'] = data[['Math', 'Science', 'Additional_lang', 'Home_lang']].apply(lambda x: sum(val >= 5 for val in x) - sum(val <= 1 for val in x), axis=1)

    # 9. Math-Science Combo
    threshold = 5  # Assuming scores above 5 as high
    data['Math_Science_high_combo'] = ((data['Math'] >= threshold) & (data['Science'] >= threshold)).astype(int)
    
    # 10. Total Achievements
    data['Total_achievements'] = data[['Math', 'Science', 'Additional_lang', 'Home_lang']].sum(axis=1)


    # Dropping Columns
    data.drop(columns=['Status_mapped'],inplace=True)

    return data


# Process both training and test data

train_data = process_data_with_binning(train_data)
test_data = process_data_with_binning(test_data)



# Univariate Analysis
features = train_data.columns.difference(['Person_id', 'Survey_date', 'Target'])
# Adjusting the layout to accommodate all the features
num_features = len(features)
num_rows = int(np.ceil(num_features / 5))

plt.figure(figsize=(20, num_rows * 3))
for i, feature in enumerate(features, 1):
    plt.subplot(num_rows, 5, i)
    if train_data[feature].dtype == 'float64' or train_data[feature].dtype == 'int64':
        sns.histplot(train_data[feature], bins=30)
    else:
        sns.countplot(data=train_data, x=feature)
    plt.title(feature)
    plt.xticks(rotation=45)
    plt.tight_layout()

plt.show()

# Correlation Coefficients for continuous variables
correlation_with_target = train_data[features].corrwith(train_data['Target'])
correlation_with_target.sort_values(ascending=False)

# Computing the VIF (Variance Inflation Factor) to check for multicollinearity
# We will only consider the numerical columns for this

# Filtering numeric columns
numeric_cols = train_data.select_dtypes(include=['float64', 'int64']).columns

# Dropping columns with NaN values for VIF computation
filtered_data = train_data[numeric_cols].dropna(axis=1)

vif_data = pd.DataFrame()
vif_data["Feature"] = filtered_data.columns
vif_data["VIF"] = [variance_inflation_factor(filtered_data.values, i) for i in range(filtered_data.shape[1])]

vif_data.sort_values(by="VIF", ascending=False)

# Save processed datasets to CSV files
train_data.to_csv('processed_train.csv', index=False)
test_data.to_csv('processed_test.csv', index=False)
