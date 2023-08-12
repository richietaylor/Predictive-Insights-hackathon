import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Function to convert percentage to pass/fail
def convert_to_pass_fail(value):
    if pd.isnull(value):
        return 0
    elif any(percentage in value for percentage in ['50 - 59 %', '60 - 69 %', '70 - 79 %', '80 - 100 %']):
        return 1
    else:
        return 0

# Load training data
train_data_path = "Train.csv"  # Adjust the path as needed
train_data = pd.read_csv(train_data_path)

# Determine if someone passed Math or Mathslit for training data
train_data['Math_Pass'] = train_data['Math'].apply(convert_to_pass_fail) * 2
train_data['Mathslit_Pass'] = train_data['Mathlit'].apply(convert_to_pass_fail)
train_data['Math_Status'] = train_data['Math_Pass'] + train_data['Mathslit_Pass']
train_data.drop(columns=['Math', 'Mathlit', 'Math_Pass', 'Mathslit_Pass','Schoolquintile'], inplace=True)

# Combine "Additional_lang" and "Home_lang" into a single pass/fail column
train_data['Language_Pass'] = train_data[['Additional_lang', 'Home_lang']].applymap(convert_to_pass_fail).max(axis=1)

# Set tenure to 0 if employed for training data
train_data.loc[train_data['Status'] == 'Employed', 'Tenure'] = 0

# Data Cleaning for Training Data
train_data.drop(columns=['Additional_lang', 'Home_lang', 'Science'], inplace=True)
train_data['Degree'].fillna(0, inplace=True)
train_data['Diploma'].fillna(0, inplace=True)
train_data['Matric'].fillna(0, inplace=True)
tenure_imputer = SimpleImputer(strategy='median')
train_data['Tenure'] = tenure_imputer.fit_transform(train_data['Tenure'].values.reshape(-1, 1))
train_data = pd.get_dummies(train_data, columns=['Geography', 'Province'], drop_first=True)
train_data['Survey_date'] = pd.to_datetime(train_data['Survey_date'])
train_data['Age'] = train_data['Survey_date'].dt.year - train_data['Birthyear']

# Filter rows based on age threshold for training data
age_threshold = 18
train_data = train_data[train_data['Age'] >= age_threshold]

# Save the processed training data to CSV
processed_file_path = "processed.csv"
train_data.to_csv(processed_file_path, index=False)

# Continue with training the Random Forest Classifier
train_data.drop(columns=['Person_id', 'Survey_date', 'Birthyear'], inplace=True)
X_train = train_data.drop(columns=['Status', 'Target'])
y_train = train_data['Target']
random_forest_model = RandomForestClassifier(random_state=42, n_jobs=-1)
random_forest_model.fit(X_train, y_train)

# Load test data
test_data_path = "Test.csv"  # Adjust the path as needed
test_data = pd.read_csv(test_data_path)

# Repeat the process for the test data
test_data['Math_Pass'] = test_data['Math'].apply(convert_to_pass_fail) * 2
test_data['Mathslit_Pass'] = test_data['Mathlit'].apply(convert_to_pass_fail)
test_data['Math_Status'] = test_data['Math_Pass'] + test_data['Mathslit_Pass']
test_data.drop(columns=['Math', 'Mathlit', 'Math_Pass', 'Mathslit_Pass'], inplace=True)

person_id_test = test_data['Person_id'].copy()

# Combine "Additional_lang" and "Home_lang" into a single pass/fail column for test data
test_data['Language_Pass'] = test_data[['Additional_lang', 'Home_lang']].applymap(convert_to_pass_fail).max(axis=1)

# Set tenure to 0 if employed for test data
test_data.loc[test_data['Status'] == 'Employed', 'Tenure'] = 0

# Data Cleaning for Test Data (similar to training data)
test_data.drop(columns=['Additional_lang', 'Home_lang', 'Science','Schoolquintile'], inplace=True)
test_data['Degree'].fillna(0, inplace=True)
test_data['Diploma'].fillna(0, inplace=True)
test_data['Matric'].fillna(0, inplace=True)
test_data['Tenure'] = tenure_imputer.transform(test_data['Tenure'].values.reshape(-1, 1))
test_data = pd.get_dummies(test_data, columns=['Geography', 'Province'], drop_first=True)
test_data['Survey_date'] = pd.to_datetime(test_data['Survey_date'])
test_data['Age'] = test_data['Survey_date'].dt.year - test_data['Birthyear']

# Filter rows based on age threshold for test data
test_data = test_data[test_data['Age'] >= age_threshold]

# Predict on test data
test_data.drop(columns=['Person_id', 'Survey_date', 'Birthyear'], inplace=True)
X_test = test_data.drop(columns=['Status'])
test_predictions = random_forest_model.predict(X_test)

# Save predictions to CSV
predictions_df = pd.DataFrame({'Person_id': person_id_test, 'Predicted_Unemployment': test_predictions})
predictions_file_path = "predictions.csv"
predictions_df.to_csv(predictions_file_path, index=False)
