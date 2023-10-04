import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold

# Load the training data
train_data = pd.read_csv('processed_train.csv')

# Removing unnecessary columns
train_data.drop(columns=['Person_id'], inplace=True)

# Separating the dependent and independent variables
X_train = train_data.drop(columns=['Target'])
y_train = train_data['Target'].astype(float)


# Check columns that have dtype 'object'
object_cols = X_train.select_dtypes(include='object').columns

if len(object_cols) > 0:
    # Print the non-numeric columns for investigation
    print("Non-numeric columns detected:")
    for col in object_cols:
        print(f"Column '{col}':")
        print(X_train[col].unique())
        print("------")
    # Convert or drop based on your requirements. For demonstration, let's try converting to numeric:
    for col in object_cols:
        try:
            X_train[col] = pd.to_numeric(X_train[col])
        except ValueError:
            # If conversion to numeric fails, drop the column (or handle it differently if needed)
            print(f"Column '{col}' could not be converted to numeric. Dropping it.")
            X_train.drop(columns=[col], inplace=True)

# Define the number of splits
n_splits = 5

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store the performance metric (e.g., accuracy) for each fold
performance = []

for train_index, val_index in skf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    # Fit the Probit model
    probit_model = sm.Probit(y_train_fold, sm.add_constant(X_train_fold))
    probit_results = probit_model.fit(disp=0)  # suppress iteration output
    
    # Predict probabilities on the validation set
    y_val_prob_pred_probit = probit_results.predict(sm.add_constant(X_val_fold))
    
    # Convert probabilities to class labels (assuming threshold of 0.5)
    y_val_pred = (y_val_prob_pred_probit > 0.5).astype(int)
    
    # Calculate accuracy (or any other metric) and append to the performance list
    accuracy = (y_val_pred == y_val_fold).mean()
    performance.append(accuracy)

# Print average performance across all folds
print(f"Average Accuracy: {np.mean(performance):.4f}")

# Optional: Refit the model on the entire training set
probit_model = sm.Probit(y_train, sm.add_constant(X_train))
probit_results = probit_model.fit(disp=0)  # suppress iteration output

# Load the test data
test_data = pd.read_csv('processed_test.csv')

# Removing unnecessary columns from the test data
test_data_processed = test_data.drop(columns=['Person_id'])

# Predicting probabilities using the Probit model
X_test = test_data_processed.astype(float)
y_test_prob_pred_probit = probit_results.predict(sm.add_constant(X_test))

# Creating a DataFrame with "person_id" and the predicted probability of unemployment
predictions_prob_df = pd.DataFrame({
    'Person_id': test_data['Person_id'],
    'Probability_Unemployed': y_test_prob_pred_probit
})

# Path to save the CSV file
predictions_csv_path = 'probit.csv'

# Saving the DataFrame to a CSV file
predictions_prob_df.to_csv(predictions_csv_path, index=False)
