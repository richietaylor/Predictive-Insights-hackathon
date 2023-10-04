import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import random
import numpy as np

random.seed(420)

# Load the data
train_data_path = "processed_train.csv"
train_data = pd.read_csv(train_data_path)
common_features = list(set(train_data.columns) - {"Person_id", "Target"})
X = train_data[common_features]
y = train_data["Target"]

strat_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define Level 1 base learners
base_learners_L1 = [
    ("knn", KNeighborsClassifier(n_neighbors=9, weights="distance", n_jobs=-1)),
    ('xgb', XGBClassifier(objective='binary:logitraw',
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    random_state=42,n_jobs=-1))
    # ... any other Level 1 base learners ...
]
stack_clf_L1 = StackingClassifier(
    estimators=base_learners_L1,
    final_estimator=LogisticRegression(),
    cv=5,
    verbose=2
)

# Define Level 2 base learners
base_learners_L2 = [
    ("stack_L1", stack_clf_L1),
    ('BernoulliNB', BernoulliNB()),
    ('ada_boost', AdaBoostClassifier(n_estimators=100, random_state=42,)),
    # ... any other Level 2 base learners ...
]
stack_clf_L2 = StackingClassifier(
    estimators=base_learners_L2,
    final_estimator=LogisticRegression(penalty='l2'),
    cv=5,
    verbose=2
)

roc_auc_scores = []

# Perform stratified k-fold validation
for train_index, valid_index in strat_kfold.split(X, y):
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    # Train Level 1 stack and get OOF predictions
    stack_clf_L1.fit(X_train, y_train)
    meta_features = stack_clf_L1.predict_proba(X_valid)[:, 1].reshape(-1, 1)

    # Train Level 2 stack on the Level 1 OOF predictions
    stack_clf_L2.fit(meta_features, y_valid)
    
    # Predict on the Level 1 OOF predictions using Level 2 and calculate ROC AUC
    predictions = stack_clf_L2.predict_proba(meta_features)[:, 1]
    roc_auc = roc_auc_score(y_valid, predictions)
    roc_auc_scores.append(roc_auc)

mean_roc_auc = np.mean(roc_auc_scores)
print(f"Mean ROC AUC across folds: {mean_roc_auc:.4f}")