from sklearn.linear_model import Lars
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.linear_model import Lars, LogisticRegression

# Load Data
train_data = pd.read_csv('processed_train.csv')
test_data = pd.read_csv('processed_test.csv')

common_features = list(set(train_data.columns) & set(test_data.columns))
common_features.remove('Person_id')
train_data = train_data[['Person_id'] + common_features + ['Target']]
test_data = test_data[['Person_id'] + common_features]

# Splitting the Training Data
X = train_data[common_features]
y = train_data['Target']

# 1. Train the LARS Model
lars_model = Lars()
lars_model.fit(X, y)

# 2. Use the LARS model's predictions as a new feature
X_train_lars = lars_model.predict(X)
X_train_with_lars = pd.concat([X, pd.Series(X_train_lars, name="LARS_output")], axis=1)

X_test_lars = lars_model.predict(test_data[common_features])
X_test_with_lars = pd.concat([test_data[common_features], pd.Series(X_test_lars, name="LARS_output")], axis=1)

# 3. Train a logistic regression model on the new dataset
logistic_model = LogisticRegression(max_iter=10000)
logistic_model.fit(X_train_with_lars, y)

# Predict probabilities on test data
test_probabilities = logistic_model.predict_proba(X_test_with_lars)[:, 1]

predictions_df = pd.DataFrame({
    'Person_id': test_data['Person_id'],
    'Probability_Unemployed': test_probabilities
})

# Saving Predictions to CSV
predictions_file_path = 'lars_logistic.csv'
predictions_df.to_csv(predictions_file_path, index=False)

# For AUC-ROC Score
roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
cross_val_scores = cross_val_score(logistic_model, X_train_with_lars, y, cv=5, scoring=roc_auc_scorer)
mean_cross_val_score = cross_val_scores.mean()

print(f"Mean AUC-ROC Score from Cross-Validation: {mean_cross_val_score:.4f}")
