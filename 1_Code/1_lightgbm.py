import pandas as pd
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

# Load the training data
train_data = pd.read_csv('processed_train.csv')

# Removing unnecessary columns
train_data.drop(columns=['Person_id'], inplace=True)

# Separating the dependent and independent variables
X_train = train_data.drop(columns=['Target'])
y_train = train_data['Target']

# Creating the LightGBM model
lightgbm_model = lgb.LGBMClassifier(
    num_leaves=31,
    min_child_samples=20,
    max_depth=-1,
    learning_rate=0.1,
    n_estimators=150,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42,
    objective='regression'
)

# Performing 5-fold cross-validation
cross_val_scores = cross_val_score(lightgbm_model, X_train, y_train, cv=5)
print(f'5-Fold Cross-Validation Scores: {cross_val_scores}')
print(f'Mean Cross-Validation Score: {cross_val_scores.mean()}')

# Fitting the LightGBM model to the training data
lightgbm_model.fit(X_train, y_train)

# Load the test data
test_data = pd.read_csv('processed_test.csv')

# Removing unnecessary columns from the test data
test_data_processed = test_data.drop(columns=['Person_id'])

# Making predictions on the test data (predicting probabilities for the positive class)
y_test_prob_pred_lgb = lightgbm_model.predict_proba(test_data_processed)[:, 1]

# Creating a DataFrame with "person_id" and the predicted probability of unemployment
predictions_prob_df = pd.DataFrame({
    'Person_id': test_data['Person_id'],
    'Probability_Unemployed': y_test_prob_pred_lgb
})

# Path to save the CSV file
predictions_csv_path = 'lightgbm.csv'

# Saving the DataFrame to a CSV file
predictions_prob_df.to_csv(predictions_csv_path, index=False)

print(f"Predictions saved to {predictions_csv_path}")
