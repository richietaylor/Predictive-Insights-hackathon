import pandas as pd
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split

# Load the training data
train_data = pd.read_csv('processed_train.csv')

# Handling Missing Values for Numerical Columns
train_data.fillna(train_data.min(numeric_only=True), inplace=True)

# Identifying categorical columns
cat_columns = train_data.select_dtypes(include=['object']).columns.tolist()
cat_columns.remove('Person_id')
cat_columns.remove('Survey_date')

# Fill missing values in categorical columns with a placeholder string
for column in cat_columns:
    train_data[column].fillna("missing", inplace=True)

# Separating the dependent and independent variables
X_train = train_data.drop(columns=['Person_id', 'Survey_date', 'Target'])
y_train = train_data['Target']

# Creating Pool object for training
train_pool = Pool(data=X_train, label=y_train, cat_features=cat_columns)

# Creating the CatBoost model
catboost_model = CatBoostClassifier(iterations=3000,
                                    learning_rate=0.01,
                                    depth=5,
                                    cat_features=cat_columns,
                                    verbose=200,loss_function='Logloss',thread_count=6,early_stopping_rounds=3)

# Performing cross-validation
cv_params = catboost_model.get_params()
cv_results = cv(train_pool, cv_params, fold_count=5, verbose=200)

print("Cross-validation results:")
print(cv_results)

# Fitting the CatBoost model to the entire training data
catboost_model.fit(train_pool)

# Load the test data
test_data = pd.read_csv('processed_test.csv')

# Preprocessing the test data (similar to the training data)
test_data.fillna(train_data.min(numeric_only=True), inplace=True)

# Fill missing values in categorical columns with a placeholder string in the test data
for column in cat_columns:
    test_data[column].fillna("missing", inplace=True)

# Removing unnecessary columns from the test data
test_data_processed = test_data.drop(columns=['Person_id', 'Survey_date'])

# Making predictions on the test data (predicting probabilities for the positive class)
y_test_prob_pred_catboost = catboost_model.predict_proba(test_data_processed)[:, 1]

# Creating a DataFrame with "Person_id" and the predicted probability of unemployment
predictions_prob_df = pd.DataFrame({
    'Person_id': test_data['Person_id'],
    'Probability_Unemployed': y_test_prob_pred_catboost
})

# Path to save the CSV file
predictions_csv_path = 'catboost.csv'

# Saving the DataFrame to a CSV file
predictions_prob_df.to_csv(predictions_csv_path, index=False)

print(f"Predictions saved to {predictions_csv_path}")
