import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from functions import evaluate_and_compare_models, get_path_variable
import time
import random
from sklearn.neural_network import MLPClassifier
import numpy as np

start_time = time.time()

#Setting Random Seeds
np.random.seed(420)
random.seed(420)

DIR = get_path_variable()
train_data_path = (
    DIR + "3_Data/processed_train.csv"
)
test_data_path = DIR + "3_Data/processed_test.csv"

# Loading the data
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

#Getting common features
common_features = list(set(train_data.columns) & set(test_data.columns))
common_features.remove("Person_id")


X = train_data[common_features]
y = train_data["Target"]

strat_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=420)

# Define base learners
base_learners = [ # This collection of base learners seemed to perform the best. Could experiment more...
    # ('extra_trees', ExtraTreesClassifier(n_estimators=150, random_state=420,criterion='log_loss',n_jobs=-1)),
    ('xgb', XGBClassifier(objective='binary:logitraw',
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    # subsample=0.5,
    colsample_bytree=0.8,
    gamma=0.1,
    random_state=420,n_jobs=-1)),
    # # ('lgb',LGBMClassifier(force_row_wise=True,objective='binary')),
    # # ('log_reg', LogisticRegression(max_iter=1000, random_state=420,C=1,penalty='l1',solver='liblinear')),
    # ('lda', LinearDiscriminantAnalysis(shrinkage=None, solver='svd')),
    ('naive_bayes', GaussianNB()),
    ('BernoulliNB', BernoulliNB()),
    ('knn', KNeighborsClassifier(n_neighbors=21,weights='distance',n_jobs=-1)),
    # ('mlp',MLPClassifier(activation='relu',learning_rate='adaptive',hidden_layer_sizes=[147,50,50],verbose=True,max_iter=500,early_stopping=True)),
    ('ada_boost', AdaBoostClassifier(n_estimators=100, random_state=420,)),
    # ('random_forest', RandomForestClassifier(n_estimators=50, random_state=420, criterion='gini',n_jobs=-1)),

]


# Show model simmilarity and performances. Diverse weak learners > single good model
evaluate_and_compare_models(base_learners=base_learners,X=X,y=y,n_splits=5)

# Initialize Stacking Classifier -- MLP performed best by far. Previously just a logistic regression.
# Passthrough made model perform worse. Maybe needed more training time?
stack_clf = StackingClassifier(
    estimators=base_learners, final_estimator=MLPClassifier(activation='relu',learning_rate='adaptive',hidden_layer_sizes=[200,100],verbose=True,random_state=420),verbose=2,
)

# Train the stacking classifier
stack_clf.fit(X, y,)


# Predict on the test data
print("Predicting on the test set...")
stacked_predictions = stack_clf.predict_proba(test_data[common_features])[:, 1]

# Save predictions to a CSV file
predictions_df = pd.DataFrame({
    'Person_id': test_data['Person_id'],
    'Probability_Unemployed': stacked_predictions
})
print("Done!")
predictions_df.to_csv(DIR + '4_Outputs/stacked.csv', index=False)
print("Saved predictions to :" + (DIR + '4_Outputs/stacked.csv'))

# # Calculate AUC-ROC using 5-fold cross-validation
# roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
# mean_auc_roc = cross_val_score(stack_clf, X, y, cv=strat_kfold, scoring=roc_auc_scorer,n_jobs=-1).mean()

# print(f"Mean AUC-ROC Score: {mean_auc_roc:.4f}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
