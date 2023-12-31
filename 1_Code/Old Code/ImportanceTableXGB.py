# Importing required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import statsmodels.api as sm

# Repeating the previous steps
train_data = pd.read_csv('processed_train.csv')
train_data.fillna(train_data.mean(numeric_only=True), inplace=True)

label_encoders = {}
for column in train_data.select_dtypes(include=['object']).columns:
    if column not in ['Person_id', 'Survey_date']:
        le = LabelEncoder()
        train_data[column] = le.fit_transform(train_data[column])
        label_encoders[column] = le

train_data.drop(columns=['Person_id', 'Survey_date'], inplace=True)
X_train = train_data.drop(columns=['Target'])
y_train = train_data['Target']
X_train_const = sm.add_constant(X_train)
xgboost_model = xgb.XGBClassifier(random_state=42)
xgboost_model.fit(X_train_const, y_train)

test_data = pd.read_csv("processed_test.csv")
test_data.fillna(train_data.mean(numeric_only=True), inplace=True)

for column, le in label_encoders.items():
    mode_value = train_data[column].mode()[0]
    test_data[column].fillna(mode_value, inplace=True)
    test_data[column] = test_data[column].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

test_data_processed = test_data.drop(columns=['Person_id', 'Survey_date'])
X_test_const = sm.add_constant(test_data_processed)

feature_importances = xgboost_model.feature_importances_
feature_importance_dict = {
    'Feature': X_train_const.columns,
    'Importance': feature_importances
}
feature_importance_df = pd.DataFrame(feature_importance_dict)
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

feature_importance_df.head()
print(feature_importance_df)