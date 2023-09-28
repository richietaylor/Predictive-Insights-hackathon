import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from functions import evaluate_and_compare_models
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

# Define base learners
base_learners = [
    # ('extra_trees', ExtraTreesClassifier(n_estimators=100, random_state=42,criterion='entropy',n_jobs=-1)),
    ('xgb', XGBClassifier(objective='binary:logitraw',
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    random_state=42,n_jobs=-1)),
    # ('log_reg', LogisticRegression(max_iter=1000, random_state=42,C=1,penalty='l1',solver='liblinear')),
    # ('lda', LinearDiscriminantAnalysis(shrinkage=None, solver='svd')),
    ('naive_bayes', GaussianNB()),
    ('knn', KNeighborsClassifier(n_neighbors=6,weights='distance',n_jobs=-1)),
    ('ada_boost', AdaBoostClassifier(n_estimators=200, random_state=42,)),
    # ('random_forest', RandomForestClassifier(n_estimators=70, random_state=42, criterion='entropy',n_jobs=-1)),

]

evaluate_and_compare_models(base_learners=base_learners,X=X,y=y,n_splits=5)


# Initialize Stacking Classifier
stack_clf = StackingClassifier(
    estimators=base_learners, final_estimator=LogisticRegression(), cv=5,verbose=2,
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


# accuracy_scorer = make_scorer(accuracy_score)
# precision_scorer = make_scorer(precision_score)
# recall_scorer = make_scorer(recall_score)
# f1_scorer = make_scorer(f1_score)
# log_loss_scorer = make_scorer(log_loss, needs_proba=True, greater_is_better=False)

# Calculate scores using cross-validation
# mean_accuracy = cross_val_score(stack_clf, X, y, cv=strat_kfold, scoring=accuracy_scorer, n_jobs=-1).mean()
# mean_precision = cross_val_score(stack_clf, X, y, cv=strat_kfold, scoring=precision_scorer, n_jobs=-1).mean()
# mean_recall = cross_val_score(stack_clf, X, y, cv=strat_kfold, scoring=recall_scorer, n_jobs=-1).mean()
# mean_f1 = cross_val_score(stack_clf, X, y, cv=strat_kfold, scoring=f1_scorer, n_jobs=-1).mean()
# mean_log_loss = cross_val_score(stack_clf, X, y, cv=strat_kfold, scoring=log_loss_scorer, n_jobs=-1).mean()
# print(f"Mean Accuracy Score: {mean_accuracy:.4f}")
# print(f"Mean Precision Score: {mean_precision:.4f}")
# print(f"Mean Recall Score: {mean_recall:.4f}")
# print(f"Mean F1 Score: {mean_f1:.4f}")
# print(f"Mean Log Loss Score: {mean_log_loss:.4f}")

# Calculate AUC-ROC using 5-fold cross-validation
roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True)# Define additional scoring metrics
mean_auc_roc = cross_val_score(stack_clf, X, y, cv=strat_kfold, scoring=roc_auc_scorer,n_jobs=-1).mean()

print(f"Mean AUC-ROC Score: {mean_auc_roc:.4f}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
