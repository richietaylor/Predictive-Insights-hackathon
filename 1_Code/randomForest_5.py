
# Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Loading Data
train_data = pd.read_csv('Train.csv')
test_data = pd.read_csv('Test.csv')

# Handling Missing Values
numerical_columns = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = train_data.select_dtypes(include=['object']).columns.tolist()
numerical_columns.remove('Target')
categorical_columns.remove('Person_id')
for col in numerical_columns:
    median_value = train_data[col].median()
    train_data[col].fillna(median_value, inplace=True)
    test_data[col].fillna(median_value, inplace=True)
for col in categorical_columns:
    mode_value = train_data[col].mode()[0]
    train_data[col].fillna(mode_value, inplace=True)
    test_data[col].fillna(mode_value, inplace=True)

# Extracting Date Features
train_data['Survey_date'] = pd.to_datetime(train_data['Survey_date'])
test_data['Survey_date'] = pd.to_datetime(test_data['Survey_date'])
train_data['Survey_year'] = train_data['Survey_date'].dt.year
train_data['Survey_month'] = train_data['Survey_date'].dt.month
train_data['Survey_day'] = train_data['Survey_date'].dt.day
test_data['Survey_year'] = test_data['Survey_date'].dt.year
test_data['Survey_month'] = test_data['Survey_date'].dt.month
test_data['Survey_day'] = test_data['Survey_date'].dt.day
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
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Training the Random Forest Model
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

# Predicting on Validation Set
y_val_prob = random_forest_model.predict_proba(X_val)[:, 1]
auc_roc_val = roc_auc_score(y_val, y_val_prob)

# Predicting on Test Data
test_probabilities = random_forest_model.predict_proba(test_data[common_features])[:, 1]
predictions_df = pd.DataFrame({
    'Person_id': test_data['Person_id'],
    'Probability_of_Being_Employed': test_probabilities
})

# Saving Predictions to CSV
predictions_file_path = 'predictions5.csv'
predictions_df.to_csv(predictions_file_path, index=False)

print("Validation AUC-ROC Score:", auc_roc_val)
