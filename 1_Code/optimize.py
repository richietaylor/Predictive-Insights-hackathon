import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
# List of paths to the CSV files containing the predictions
predictions_files = [
    'lightgbm.csv',
    'xgboost.csv',
    'catboost.csv',
    'randomforest.csv',
    'histboosting.csv'

    # Add more files here as needed
]

# List of weights for each CSV file (should be the same length as predictions_files)
weights = [
    1.0, 
    1.0, 
    1.0, 
    1.0,
    1.0
    
]

# Loading Data
train_data = pd.read_csv('processed_train.csv')
test_data = pd.read_csv('processed_test.csv')

common_features = list(set(train_data.columns) & set(test_data.columns))
common_features.remove('Person_id')
train_data = train_data[['Person_id'] + common_features + ['Target']]
test_data = test_data[['Person_id'] + common_features]

# Splitting the Training Data
X = train_data[common_features]
y = train_data['Target']

# Splitting the Training Data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Objective function to optimize
def objective(weights: list) -> float:
    """Compute the negative AUC-ROC score of the weighted ensemble."""
    final_prediction = 0

    for weight, file_path in zip(weights, predictions_files):
        predictions = pd.read_csv(file_path)
        # print(f"Length of predictions from {file_path}: {len(predictions)}")
        final_prediction += predictions['Probability_Unemployed'] * weight

    # Normalize by dividing by the sum of weights
    final_prediction /= sum(weights)
    print(y_val.shape)
    print(final_prediction.shape)
    score = roc_auc_score(y_val, final_prediction)
    
    # We return the negative value because we want to maximize AUC-ROC score
    return -score

# Constraints: weights are between 0 and 1 and they sum up to 1
cons = ({'type': 'eq', 'fun': lambda w: sum(w) - 1})
bounds = [(0, 1) for _ in range(len(predictions_files))]

# Initial guess (equal weights)
initial_weights = [1./len(predictions_files)] * len(predictions_files)

result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=cons)

optimal_weights = result.x
print(f"Optimal weights are: {optimal_weights}")

# Merge predictions using optimal weights
predictions_ensemble_df = pd.read_csv(predictions_files[0])
weighted_probabilities = predictions_ensemble_df['Probability_Unemployed'] * optimal_weights[0]

for idx, file_path in enumerate(predictions_files[1:]):
    predictions = pd.read_csv(file_path)
    weighted_probabilities += predictions['Probability_Unemployed'] * optimal_weights[idx + 1]

# Normalize by the sum of optimal weights (it should be close to 1 due to constraints but just to be sure)
weighted_probabilities /= sum(optimal_weights)

# Update the 'Probability_Unemployed' column with the weighted probabilities
predictions_ensemble_df['Probability_Unemployed'] = weighted_probabilities

# Save the result to a new CSV file
predictions_ensemble_df.to_csv('ensemble_optimal_weights.csv', index=False)