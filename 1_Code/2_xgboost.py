import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score

# Load the training data
train_data = pd.read_csv('Train.csv')

# Handling Missing Values for Numerical Columns
train_data.fillna(train_data.min(numeric_only=True), inplace=True)

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

# Adding a constant to the independent variables (intercept term)
X_train_const = sm.add_constant(X_train)

# Creating the XGBoost model
xgboost_model = xgb.XGBClassifier(random_state=42)

# Performing 5-fold cross-validation
cross_val_scores = cross_val_score(xgboost_model, X_train_const, y_train, cv=5)
print(f'5-Fold Cross-Validation Scores: {cross_val_scores}')
print(f'Mean Cross-Validation Score: {cross_val_scores.mean()}')

# Fitting the XGBoost model to the training data
xgboost_model.fit(X_train_const, y_train)

# Load the test data
test_data = pd.read_csv('Test.csv')

# Handling Missing Values for Numerical Columns in the test data
test_data.fillna(train_data.min(numeric_only=True), inplace=True)

# Handling Missing Values for Categorical Columns and Encoding in the test data
for column, le in label_encoders.items():
    mode_value = train_data[column].mode()[0]
    test_data[column].fillna(mode_value, inplace=True)
    test_data[column] = test_data[column].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

# Removing unnecessary columns from the test data
test_data_processed = test_data.drop(columns=['Person_id', 'Survey_date'])

# Adding a constant to the test independent variables (intercept term)
X_test_const = sm.add_constant(test_data_processed)

# Making predictions on the test data (predicting probabilities for the positive class)
y_test_prob_pred_xgb = xgboost_model.predict_proba(X_test_const)[:, 1]

# Creating a DataFrame with "person_id" and the predicted probability of unemployment
predictions_prob_df = pd.DataFrame({
    'Person_id': test_data['Person_id'],
    'Probability_Unemployed': y_test_prob_pred_xgb
})

# Path to save the CSV file
predictions_csv_path = 'predictions2.csv'

# Saving the DataFrame to a CSV file
predictions_prob_df.to_csv(predictions_csv_path, index=False)

print(f"Predictions saved to {predictions_csv_path}")
