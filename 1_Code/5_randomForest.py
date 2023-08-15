# Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, make_scorer

# Loading Data
train_data = pd.read_csv('processed_train.csv')
test_data = pd.read_csv('processed_test.csv')

# Handling Missing Values
numerical_columns = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = train_data.select_dtypes(include=['object']).columns.tolist()
numerical_columns.remove('Target')
categorical_columns.remove('Person_id')

# For numerical columns, fill missing values with the minimum
for col in numerical_columns:
    min_value = train_data[col].min()
    train_data[col].fillna(min_value, inplace=True)
    test_data[col].fillna(min_value, inplace=True)

# For categorical columns, fill missing values with the mode
for col in categorical_columns:
    mode_value = train_data[col].mode()[0]
    train_data[col].fillna(mode_value, inplace=True)
    test_data[col].fillna(mode_value, inplace=True)

train_data.drop(columns=['Survey_date'], inplace=True)
test_data.drop(columns=['Survey_date'], inplace=True)

# One-Hot Encoding
train_data = pd.get_dummies(train_data, columns=categorical_columns[1:])
test_data = pd.get_dummies(test_data, columns=categorical_columns[1:])
common_features = list(set(train_data.columns) & set(test_data.columns))
common_features.remove('Person_id')
train_data = train_data[['Person_id'] + common_features + ['Target']]
test_data = test_data[['Person_id'] + common_features]

# Splitting the Training Data
X = train_data[common_features]
y = train_data['Target']

# Training the Random Forest Model
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42, criterion='entropy')

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
predictions_file_path = 'predictions5.csv'
predictions_df.to_csv(predictions_file_path, index=False)

print(f"Mean AUC-ROC Score from Cross-Validation: {mean_cross_val_score:.4f}")
