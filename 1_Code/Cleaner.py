import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(train_file, test_file):
    # Load the training data
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Handling Missing Values for Numerical Columns
    train_data.fillna(train_data.min(numeric_only=True), inplace=True)
    test_data.fillna(train_data.min(numeric_only=True), inplace=True)

    # Encoding Categorical Variables
    label_encoders = {}
    for column in train_data.select_dtypes(include=['object']).columns:
        if column not in ['Person_id', 'Survey_date']:
            le = LabelEncoder()
            train_data[column] = le.fit_transform(train_data[column])
            label_encoders[column] = le

    # Handling Missing Values for Categorical Columns and Encoding in the test data
    for column, le in label_encoders.items():
        mode_value = train_data[column].mode()[0]
        test_data[column].fillna(mode_value, inplace=True)
        test_data[column] = test_data[column].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # Removing unnecessary columns
    train_data.drop(columns=['Person_id', 'Survey_date'], inplace=True)
    test_data.drop(columns=['Person_id', 'Survey_date'], inplace=True)

    # Save the cleaned data to CSV files
    train_data.to_csv('train_clean.csv', index=False)
    test_data.to_csv('test_clean.csv', index=False)

# Call the function with the paths to the raw train and test data
preprocess_data('Train.csv', 'Test.csv')
