import pandas as pd
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from functions import evaluate_and_compare_models
import time
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import random
from sklearn.neural_network import MLPClassifier
from mlpclassifier import SklearnCompatibleMLP
from statsmodels.discrete.discrete_model import Probit

random.seed(4200)

start_time = time.time()
# Paths to your data
train_data_path = "processed_train.csv"
test_data_path = "processed_test.csv"

# Loading the data
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

common_features = list(set(train_data.columns) & set(test_data.columns))
common_features.remove("Person_id")


X = train_data[common_features]
y = train_data["Target"]

strat_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=420)

# Define base learners
base_learners = [
    # ('extra_trees', ExtraTreesClassifier(n_estimators=150, random_state=420,criterion='log_loss',n_jobs=-1)),
    (
        "xgb",
        XGBClassifier(
            objective="binary:logitraw",
            n_estimators=200,
            learning_rate=0.03,
            # max_depth=5,
            # subsample=0.5,
            colsample_bytree=0.8,
            # gamma=0.1,
            random_state=420,
            n_jobs=-1,
        ),
    ),
    # # ('lgb',LGBMClassifier(force_row_wise=True,objective='binary')),
    # ('log_reg', LogisticRegression(max_iter=1000, random_state=420,C=1,penalty='l1',solver='liblinear')),
    # ('lda', LinearDiscriminantAnalysis(shrinkage=None, solver='svd')),
    # ('naive_bayes', GaussianNB()),
    ("BernoulliNB", BernoulliNB()),
    ("knn", KNeighborsClassifier(n_neighbors=26, weights="distance", n_jobs=-1)),
    # ('mlp',MLPClassifier(activation='relu',learning_rate='adaptive',hidden_layer_sizes=[147,50,50],verbose=True,max_iter=500,early_stopping=True)),
    (
        "ada_boost",
        AdaBoostClassifier(
            n_estimators=100,
            random_state=420,
        ),
    ),
    # (
    #     "random_forest",
    #     RandomForestClassifier(
    #         n_estimators=50, random_state=420, criterion="gini", n_jobs=-1
    #     ),
    # ),
]

evaluate_and_compare_models(base_learners=base_learners, X=X, y=y, n_splits=5)

# mlp = SklearnCompatibleMLP(len(base_learners),200,1,100,0.001)


# # Initialize Stacking Classifier
# stack_clf = StackingClassifier(
#     estimators=base_learners, final_estimator=mlp, cv=5,verbose=2,
# )

# Initialize Stacking Classifier
stack_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=MLPClassifier(
        activation="relu",
        learning_rate="adaptive",
        hidden_layer_sizes=[300, 100],
        verbose=True,
        early_stopping=True,
        solver="adam",
        alpha=0.1,
        beta_1=0.5,
        beta_2=0.5,
    ),
    cv=strat_kfold,
    verbose=2,
    n_jobs=-1,
)

# # Initialize Stacking Classifier
# stack_clf = StackingClassifier(
#     estimators=base_learners, final_estimator=MLPClassifier(activation='relu',solver='adam',learning_rate='adaptive',verbose=True,hidden_layer_sizes=[200,100],max_iter=200,), cv=5,verbose=2,
# )

# # Initialize Stacking Classifier
# stack_clf = VotingClassifier(
#     estimators=base_learners,verbose=2,voting='soft',weights=[2,1,1,1]
# )


# Train the stacking classifier
stack_clf.fit(
    X,
    y,
)

# # Retrieve the weights of the final estimator
# weights = stack_clf.final_estimator_.coef_

# print(weights)

# Predict on the test data
stacked_predictions = stack_clf.predict_proba(test_data[common_features])[:, 1]

# Save predictions to a CSV file
predictions_df = pd.DataFrame(
    {"Person_id": test_data["Person_id"], "Probability_Unemployed": stacked_predictions}
)
predictions_df.to_csv("stacked.csv", index=False)

# Calculate AUC-ROC using 5-fold cross-validation
roc_auc_scorer = make_scorer(
    roc_auc_score, needs_proba=True
)  # Define additional scoring metrics
mean_auc_roc = cross_val_score(
    stack_clf, X, y, cv=strat_kfold, scoring=roc_auc_scorer, n_jobs=-1
).mean()

print(f"Mean AUC-ROC Score: {mean_auc_roc:.4f}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
