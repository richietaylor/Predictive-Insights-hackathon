from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score ,make_scorer
import pandas as pd

# Load the data
train_data = pd.read_csv('processed_train.csv')
test_data = pd.read_csv('processed_test.csv')

common_features = list(set(train_data.columns) & set(test_data.columns))
common_features.remove('Person_id')
train_data = train_data[['Person_id'] + common_features + ['Target']]
test_data = test_data[['Person_id'] + common_features]

# Prepare the data
X = train_data[common_features]
y = train_data['Target']

# Initialize a base classifier
base_classifier = DecisionTreeClassifier(max_depth=1)

# Initialize AdaBoostClassifier
ada_classifier = AdaBoostClassifier(base_classifier, n_estimators=100, random_state=42)

# Evaluate performance using 5-fold cross-validation
roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
cross_val_scores = cross_val_score(ada_classifier, X, y, cv=5, scoring=roc_auc_scorer)
mean_cross_val_score = cross_val_scores.mean()

print(f"Mean ROC AUC Score from Cross-Validation: {mean_cross_val_score:.4f}")
