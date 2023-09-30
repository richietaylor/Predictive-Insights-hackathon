import pandas as pd
from sklearn.ensemble import BaggingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


import time
start_time = time.time()
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

strat_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
X_scaled = scaler.fit_transform(X)

# Only transform the test data using the scaler fitted on the training data
test_data_scaled = scaler.transform(test_data[common_features])

# Define base learners
base_learners = [
    ('extra_trees', ExtraTreesClassifier(n_estimators=100, random_state=42,criterion='entropy',)),
    ('xgb', XGBClassifier(objective='binary:logitraw',
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    random_state=42)),
    # ('log_reg', LogisticRegression(max_iter=1000, random_state=42,C=1,penalty='l1',solver='liblinear')),
    ('lda', LinearDiscriminantAnalysis(shrinkage=None, solver='svd')),
    ('naive_bayes', GaussianNB()),
    ('knn', KNeighborsClassifier(n_neighbors=6,weights='distance')),
    # ('ada_boost', AdaBoostClassifier(n_estimators=200, random_state=42,)),
    ('random_forest', RandomForestClassifier(n_estimators=70, random_state=42, criterion='entropy')),

]
# Initialize Stacking Classifier
# Initialize the Bagging Classifier
bag_clf = BaggingClassifier(
    base_estimator=XGBClassifier(),  # You can change this to any other classifier
    n_estimators=50,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True, 
    bootstrap_features=False,
    n_jobs=-1,
    random_state=42
)

# Train the stacking classifier
bag_clf.fit(X, y)

# Predict on the test data
stacked_predictions = bag_clf.predict_proba(test_data[common_features])[:, 1]

# Save predictions to a CSV file
predictions_df = pd.DataFrame({
    'Person_id': test_data['Person_id'],
    'Probability_Unemployed': stacked_predictions
})
predictions_df.to_csv('stacked.csv', index=False)

# Calculate AUC-ROC using 5-fold cross-validation
roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
mean_auc_roc = cross_val_score(bag_clf, X, y, cv=strat_kfold, scoring=roc_auc_scorer,n_jobs=-1).mean()

print(f"Mean AUC-ROC Score: {mean_auc_roc:.4f}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
