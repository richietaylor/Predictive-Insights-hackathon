import pandas as pd
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# Load the training data
train_data = pd.read_csv('Train.csv')

# Handling Missing Values for Numerical Columns
train_data.fillna(train_data.mean(numeric_only=True), inplace=True)

# Encoding Categorical Variables
label_encoders = {}
for column in train_data.select_dtypes(include=['object']).columns:
    if column not in ['Person_id', 'Survey_date']:
        le = LabelEncoder()
        train_data[column] = le.fit_transform(train_data[column])
        label_encoders[column] = le

# Removing unnecessary columns
train_data.drop(columns=['Person_id', 'Survey_date'], inplace=True)

# Separating the dependent and independent variables
X_train = train_data.drop(columns=['Target'])
y_train = train_data['Target']

# Creating the LightGBM model
lightgbm_model = lgb.LGBMClassifier(random_state=42)

# Fitting the LightGBM model to the training data
lightgbm_model.fit(X_train, y_train)

# Load the test data
test_data = pd.read_csv('Test.csv')

# Handling Missing Values for Numerical Columns in the test data
test_data.fillna(train_data.mean(numeric_only=True), inplace=True)

# Handling Missing Values for Categorical Columns and Encoding in the test data
for column, le in label_encoders.items():
    mode_value = train_data[column].mode()[0]
    test_data[column].fillna(mode_value, inplace=True)
    test_data[column] = test_data[column].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

# Removing unnecessary columns from the test data
test_data_processed = test_data.drop(columns=['Person_id', 'Survey_date'])

# Making predictions on the test data (predicting probabilities for the positive class)
y_test_prob_pred_lgb = lightgbm_model.predict_proba(test_data_processed)[:, 1]

# Creating a DataFrame with "person_id" and the predicted probability of unemployment
predictions_prob_df = pd.DataFrame({
    'Person_id': test_data['Person_id'],
    'Probability_Unemployed': y_test_prob_pred_lgb
})

# Path to save the CSV file
predictions_csv_path = 'predictions.csv'

# Saving the DataFrame to a CSV file
predictions_prob_df.to_csv(predictions_csv_path, index=False)

print(f"Predictions saved to {predictions_csv_path}")
