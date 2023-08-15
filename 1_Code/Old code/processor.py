import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

# Function to process the data
def process_data(data):
    # Convert "Survey_date" to datetime and extract features
    data['Survey_date'] = pd.to_datetime(data['Survey_date'])
    data['Survey_year'] = data['Survey_date'].dt.year
    data['Survey_month'] = data['Survey_date'].dt.month
    data['Survey_day'] = data['Survey_date'].dt.day

    # Calculate age
    # data['Age'] = data['Survey_year'] - data['Birthyear']
    # data['Age'] = data.apply(lambda row: row['Age'] - 1 if row['Survey_month'] < row['Birthmonth'] else row['Age'], axis=1)
    data['Age'] = data.apply(calculate_exact_age, axis=1)
    # Map "unemployed" status to -1, and all other statuses to 1
    data['Status_mapped'] = data['Status'].apply(lambda x: -1 if x == 'unemployed' else 1)

    # Create new feature by multiplying 'Tenure' by 'Status_mapped'
    data['Tenure_Status'] = data['Tenure'] * data['Status_mapped']

    # Dropping Columns Here

    data.drop(columns=['Status_mapped','Birthyear','Birthmonth'],inplace=True)
    
    # Apply categorical encoding
    data = encode_categorical_columns(data)

    # Transform percentage range columns
    data = transform_percentage_columns(data)

    # Fill missing Tenure values
    average_tenure_by_province = data.groupby('Province')['Tenure'].mean()
    data['Tenure'] = data.apply(lambda row: average_tenure_by_province[row['Province']] if pd.isnull(row['Tenure']) else row['Tenure'], axis=1)

    return data

# Process both training and test data
train_data = process_data(train_data)
test_data = process_data(test_data)

# Save processed datasets to CSV files
train_data.to_csv('processed_train.csv', index=False)
test_data.to_csv('processed_test.csv', index=False)
