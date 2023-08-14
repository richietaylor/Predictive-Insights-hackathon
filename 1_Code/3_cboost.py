import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
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

# Splitting the dataset into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

# Creating Pool objects for training and validation
train_pool = Pool(data=X_train_split, label=y_train_split, cat_features=cat_columns)
val_pool = Pool(data=X_val_split, label=y_val_split, cat_features=cat_columns)

# Creating the CatBoost model
catboost_model = CatBoostClassifier(iterations=2000,
                                    learning_rate=0.02,
                                    depth=4,
                                    cat_features=cat_columns,
                                    verbose=200)

# Fitting the CatBoost model to the training data
catboost_model.fit(train_pool, eval_set=val_pool)

# Predicting probabilities for the validation set (positive class)
y_val_prob_pred_catboost = catboost_model.predict_proba(X_val_split)[:, 1]

# Calculating the ROC AUC score for the validation set
roc_auc_val = roc_auc_score(y_val_split, y_val_prob_pred_catboost)
print(f"ROC AUC Score on Validation Set: {roc_auc_val}")

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
predictions_csv_path = 'predictions3.csv'

# Saving the DataFrame to a CSV file
predictions_prob_df.to_csv(predictions_csv_path, index=False)

print(f"Predictions saved to {predictions_csv_path}")
