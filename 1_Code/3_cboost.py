import pandas as pd
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split
import numpy as np

# Load the training data
train_data = pd.read_csv('processed_train.csv')


# Identifying categorical columns
cat_columns = train_data.select_dtypes(include=['object']).columns.tolist()
cat_columns.remove('Person_id')

# Separating the dependent and independent variables
X_train = train_data.drop(columns=['Person_id', 'Target'])
y_train = train_data['Target']

# Creating Pool object for training
train_pool = Pool(data=X_train, label=y_train, cat_features=cat_columns)

# Creating the CatBoost model
catboost_model = CatBoostClassifier(iterations=3000,
                                    learning_rate=0.01,
                                    depth=5,
                                    cat_features=cat_columns,
                                    verbose=200,loss_function='Logloss',thread_count=6,early_stopping_rounds=5)

# Performing cross-validation
cv_params = catboost_model.get_params()
cv_results = cv(train_pool, cv_params, fold_count=5, verbose=200)

print("Cross-validation results:")
print(cv_results)

# Fitting the CatBoost model to the entire training data
catboost_model.fit(train_pool)

# Load the test data
test_data = pd.read_csv('processed_test.csv')


# Removing unnecessary columns from the test data
test_data_processed = test_data.drop(columns=['Person_id'])

# Making predictions on the test data (predicting probabilities for the positive class)
y_test_prob_pred_catboost = catboost_model.predict_proba(test_data_processed)[:, 1]

# Creating a DataFrame with "Person_id" and the predicted probability of unemployment
predictions_prob_df = pd.DataFrame({
    'Person_id': test_data['Person_id'],
    'Probability_Unemployed': y_test_prob_pred_catboost
})

# Path to save the CSV file
predictions_csv_path = 'catboost.csv'

# Saving the DataFrame to a CSV file
predictions_prob_df.to_csv(predictions_csv_path, index=False)

print(f"Predictions saved to {predictions_csv_path}")

import os
import pandas as pd

current_script = os.path.abspath(__file__)
current_filename = os.path.basename(current_script)

def save_predictions(predictions, data, file_path):
    """
    Save model predictions to a specified file.
    
    Parameters:
    - predictions: Model's predictions.
    - data: Original data used for making predictions.
    - file_path: Path to save the predictions.
    """
    # Square the predictions
    predictions_squared = np.abs(np.log(np.abs(predictions+0.00001)))
    
    # Check if the file exists
    if os.path.exists(file_path):
        # If it exists, read the file
        predictions_df = pd.read_csv(file_path)
    else:
        # If it doesn't exist, create a new DataFrame with 'Person_id'
        predictions_df = pd.DataFrame({'Person_id': data['Person_id']})

    # Add the squared predictions as a new column
    predictions_df[current_filename] = predictions_squared

    # Save the updated DataFrame back to the file
    predictions_df.to_csv(file_path, index=False)

# For training data
y_train_prob_pred_lgb = catboost_model.predict_proba(X_train)[:, 1]
save_predictions(y_train_prob_pred_lgb, train_data, 'processed_train_advanced.csv')

# For test data
y_test_prob_pred_lgb = catboost_model.predict_proba(test_data_processed)[:, 1]
save_predictions(y_test_prob_pred_lgb, test_data, 'processed_test_advanced.csv')

