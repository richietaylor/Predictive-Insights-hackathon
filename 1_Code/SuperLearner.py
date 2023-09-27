import pandas as pd
from sklearn.ensemble import (
    StackingClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

# Paths to your data
train_data_path = (
    "processed_train.csv"
)
test_data_path = "processed_test.csv"

# Loading the data
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

common_features = list(set(train_data.columns) & set(test_data.columns))
common_features.remove("Person_id")


X = train_data[common_features]
y = train_data["Target"]

# Define base learners
base_learners = [
    ('extra_trees', ExtraTreesClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(objective='binary:logistic',
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    random_state=42,)),
    ('hist_gb', HistGradientBoostingClassifier(max_iter=100, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=6))
]

# Initialize Stacking Classifier
stack_clf = StackingClassifier(
    estimators=base_learners, final_estimator=LogisticRegression(), cv=5,
)

# Train the stacking classifier
stack_clf.fit(X, y)

# Predict on the test data
stacked_predictions = stack_clf.predict_proba(test_data[common_features])[:, 1]

# Save predictions to a CSV file
predictions_df = pd.DataFrame({
    'Person_id': test_data['Person_id'],
    'Probability_Unemployed': stacked_predictions
})
predictions_df.to_csv('stacked.csv', index=False)

# Calculate AUC-ROC using 5-fold cross-validation
roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
mean_auc_roc = cross_val_score(stack_clf, X, y, cv=5, scoring=roc_auc_scorer).mean()
print(f"Mean AUC-ROC Score: {mean_auc_roc:.4f}")
