import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
<<<<<<< HEAD:1_Code/2_xgboost.py
from sklearn.model_selection import cross_val_score
=======
import statsmodels.api as sm
>>>>>>> ecba2cd93fc5e02b4e77d00773f352ebc0e9df75:1_Code/make_prediction_xgboost.py

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
train_data.drop(columns=['Person_id', 'Survey_date','round','year'], inplace=True)

# Separating the dependent and independent variables
X_train = train_data.drop(columns=['Target'])
y_train = train_data['Target']

<<<<<<< HEAD:1_Code/2_xgboost.py
# Creating the XGBoost model with parameters
xgboost_model = xgb.XGBClassifier(
    objective='count:poisson',
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    random_state=42
)

# Performing 5-fold cross-validation
cross_val_scores = cross_val_score(xgboost_model, X_train, y_train, cv=5)
print(f'5-Fold Cross-Validation Scores: {cross_val_scores}')
print(f'Mean Cross-Validation Score: {cross_val_scores.mean()}')
=======
# Adding a constant to the independent variables (intercept term)
X_train_const = sm.add_constant(X_train)

# Creating the XGBoost model
xgboost_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False)
>>>>>>> ecba2cd93fc5e02b4e77d00773f352ebc0e9df75:1_Code/make_prediction_xgboost.py

# Fitting the XGBoost model to the training data
xgboost_model.fit(X_train, y_train)

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
test_data_processed = test_data.drop(columns=['Person_id', 'Survey_date','round','year'])

# Making predictions on the test data (predicting probabilities for the positive class)
y_test_prob_pred_xgb = xgboost_model.predict_proba(test_data_processed)[:, 1]

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
