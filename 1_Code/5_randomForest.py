# Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, make_scorer
import matplotlib.pyplot as plt

# Loading Data
train_data = pd.read_csv('processed_train.csv')
test_data = pd.read_csv('processed_test.csv')

common_features = list(set(train_data.columns) & set(test_data.columns))
common_features.remove('Person_id')
train_data = train_data[['Person_id'] + common_features + ['Target']]
test_data = test_data[['Person_id'] + common_features]

# Splitting the Training Data
X = train_data[common_features]
y = train_data['Target']

# Training the Random Forest Model
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42, criterion='entropy',)

# Performing 5-fold cross-validation
roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
cross_val_scores = cross_val_score(random_forest_model, X, y, cv=5, scoring=roc_auc_scorer)
mean_cross_val_score = cross_val_scores.mean()

# Fitting the model to the entire training data for predictions
random_forest_model.fit(X, y)

# Predicting on Test Data
test_probabilities = random_forest_model.predict_proba(test_data[common_features])[:, 1]

predictions_df = pd.DataFrame({
    'Person_id': test_data['Person_id'],
    'Probability_Unemployed': test_probabilities  # Using the correct column name
})

# Saving Predictions to CSV
predictions_file_path = 'randomforest.csv'
predictions_df.to_csv(predictions_file_path, index=False)

print(f"Mean AUC-ROC Score from Cross-Validation: {mean_cross_val_score:.4f}")
