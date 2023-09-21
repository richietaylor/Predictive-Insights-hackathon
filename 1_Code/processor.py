import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA

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
    data['downturn'] = data['Round'].apply(lambda x: 1 if x in [2, 4] else 0)

    # Dropping Columns
    data.drop(columns=['Tenure', 'Mathlit', 'Additional_lang', 'Survey_year', 'Age', 'Exposure', 'downturn'],inplace=True)

    return data


# Process both training and test data
train_data = process_data(train_data)
test_data = process_data(test_data)

# Save processed datasets to CSV files
train_data.to_csv('processed_train.csv', index=False)
test_data.to_csv('processed_test.csv', index=False)



# # # Univariate Analysis
# features = train_data.columns.difference(['Person_id', 'Survey_date', 'Target'])
# # Adjusting the layout to accommodate all the features
# num_features = len(features)
# num_rows = int(np.ceil(num_features / 5))

# plt.figure(figsize=(20, num_rows * 3))
# for i, feature in enumerate(features, 1):
#     plt.subplot(num_rows, 5, i)
#     if train_data[feature].dtype == 'float64' or train_data[feature].dtype == 'int64':
#         sns.histplot(train_data[feature], bins=30)
#     else:
#         sns.countplot(data=train_data, x=feature)
#     plt.title(feature)
#     plt.xticks(rotation=45)
#     plt.tight_layout()

# plt.show()

# Correlation Coefficients for continuous variables
# correlation_with_target = train_data[features].corrwith(train_data['Target'])
# correlation_with_target.sort_values(ascending=False)

# Computing the VIF (Variance Inflation Factor) to check for multicollinearity
# We will only consider the numerical columns for this

# Filtering numeric columns
# numeric_cols = train_data.select_dtypes(include=['float64', 'int64']).columns

# # Dropping columns with NaN values for VIF computation
# filtered_data = train_data[numeric_cols].dropna(axis=1)

# vif_data = pd.DataFrame()
# vif_data["Feature"] = filtered_data.columns
# vif_data["VIF"] = [variance_inflation_factor(filtered_data.values, i) for i in range(filtered_data.shape[1])]

# print(vif_data.sort_values(by="VIF", ascending=False))


# pca_data = filtered_data.drop(columns=['Target'])
# pca_data = (pca_data - pca_data.mean()) / pca_data.std()


# pca = PCA()

# pca_result = pca.fit_transform(pca_data)

# explained_variance = pca.explained_variance_ratio_
# cumulative_variance = np.cumsum(explained_variance)

# plt.figure(figsize=(10,5))
# plt.bar(range(len(explained_variance)),explained_variance,alpha=0.5,align='center',label='individual explained variance')
# plt.step(range(len(cumulative_variance)),cumulative_variance,where='mid',label='cumulative explained variance')
# plt.xlabel("Principal components")
# plt.ylabel("Explained variance ratio")
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()  




# Handling Missing Values
numerical_columns = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = train_data.select_dtypes(include=['object']).columns.tolist()
numerical_columns.remove('Target')
categorical_columns.remove('Person_id')

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




from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Splitting the data
X = train_data.drop(columns=['Target', 'Person_id', 'Survey_date'])
y = train_data['Target']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# Normalize the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Applying L1 regularization using LassoCV
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train_scaled, y_train)

# Predicting on validation set
y_pred = lasso.predict(X_val_scaled)
mse = mean_squared_error(y_val, y_pred)

# Checking features that have been dropped by Lasso (coefficient = 0)
dropped_features = X.columns[lasso.coef_ == 0].tolist()


print("MSE:", mse, dropped_features)