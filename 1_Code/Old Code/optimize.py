import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# List of paths to the CSV files containing the predictions
predictions_files = [
    'lightgbm.csv',
    'xgboost.csv',
    'catboost.csv',
    'randomforest.csv',
    'histboosting.csv'
]

# Initial guess (equal weights)
initial_weights = [1./len(predictions_files)] * len(predictions_files)

# Loading Data
train_data = pd.read_csv('processed_train.csv')
X = train_data.drop(columns=['Target', 'Person_id'])
y = train_data['Target']

# Reorder predictions to match the order of the training data
def reorder_predictions(file_path, train_data):
    predictions = pd.read_csv(file_path)
    merged = train_data.merge(predictions, on='Person_id', how='left')
    return merged['Probability_Unemployed'].values

# Objective function for optimization
def objective(weights: list, X_val_indices=None) -> float:
    """Compute the negative AUC-ROC score of the weighted ensemble."""
    final_prediction = 0
    for weight, file_path in zip(weights, predictions_files):
        predictions = reorder_predictions(file_path, train_data)
        if X_val_indices is not None:
            predictions = predictions[X_val_indices]
        final_prediction += predictions * weight

    # Normalize by dividing by the sum of weights
    final_prediction /= sum(weights)
    
    if X_val_indices is not None:
        y_true = y.iloc[X_val_indices]
    else:
        y_true = y

    score = roc_auc_score(y_true, final_prediction)
    
    # We return the negative value because we want to maximize AUC-ROC score
    return -score

# Constraints: weights are between 0 and 1 and they sum up to 1
cons = ({'type': 'eq', 'fun': lambda w: sum(w) - 1})
bounds = [(0, 1) for _ in range(len(predictions_files))]

# Performing 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_indices, val_indices in kf.split(X):
    result = minimize(objective, initial_weights, args=(val_indices,), method='SLSQP', bounds=bounds, constraints=cons)
    optimal_weights_fold = result.x
    fold_score = -objective(optimal_weights_fold, val_indices)
    cv_scores.append(fold_score)

print(f'5-Fold Cross-Validation AUC-ROC Scores: {cv_scores}')
print(f'Mean Cross-Validation AUC-ROC Score: {sum(cv_scores)/len(cv_scores)}')

# Finding the optimal weights using the entire dataset
result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=cons)
optimal_weights = result.x
print(f"Optimal weights for the entire dataset are: {optimal_weights}")

# Optionally, save the weights to a file
with open('optimal_weights.txt', 'w') as f:
    for weight, file in zip(optimal_weights, predictions_files):
        f.write(f"{file}: {weight}\n")
