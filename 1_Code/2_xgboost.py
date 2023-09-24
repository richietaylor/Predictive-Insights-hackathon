import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import shap
# Load the training data
train_data = pd.read_csv('processed_train.csv')


# Removing unnecessary columns
train_data.drop(columns=['Person_id'], inplace=True)

# Separating the dependent and independent variables
X_train = train_data.drop(columns=['Target'])
y_train = train_data['Target']

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

# Fitting the XGBoost model to the training data
xgboost_model.fit(X_train, y_train)

# Load the test data
test_data = pd.read_csv('processed_test.csv')


# Removing unnecessary columns from the test data
test_data_processed = test_data.drop(columns=['Person_id'])

# Making predictions on the test data (predicting probabilities for the positive class)
y_test_prob_pred_xgb = xgboost_model.predict_proba(test_data_processed)[:, 1]

# Creating a DataFrame with "person_id" and the predicted probability of unemployment
predictions_prob_df = pd.DataFrame({
    'Person_id': test_data['Person_id'],
    'Probability_Unemployed': y_test_prob_pred_xgb
})

# Path to save the CSV file
predictions_csv_path = 'xgboost.csv'

# Saving the DataFrame to a CSV file
predictions_prob_df.to_csv(predictions_csv_path, index=False)

print(f"Predictions saved to {predictions_csv_path}")


# # Create SHAP explainer
# explainer = shap.TreeExplainer(xgboost_model)

# # Compute SHAP values for the entire training set (or a subset if it's too large)
# shap_values = explainer.shap_values(X_train)

# # Plot summary of SHAP values
# shap.summary_plot(shap_values, X_train)
