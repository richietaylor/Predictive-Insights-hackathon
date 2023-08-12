import pandas as pd

# Paths to the CSV files containing the predictions
predictions_file1 = 'predictions1.csv'
predictions_file2 = 'predictions2.csv'
predictions_file3 = 'predictions3.csv' # Adding the third file

# Reading the prediction files
predictions1 = pd.read_csv(predictions_file1)
predictions2 = pd.read_csv(predictions_file2)
predictions3 = pd.read_csv(predictions_file3) # Reading the third file

# Making sure the 'Person_id' columns match in all files
if not (predictions1['Person_id'] == predictions2['Person_id']).all() or not (predictions1['Person_id'] == predictions3['Person_id']).all():
    raise ValueError("The 'Person_id' columns must match in all prediction files.")

# Averaging the probability predictions from all three files
average_probabilities = (predictions1['Probability_Unemployed'] + predictions2['Probability_Unemployed'] + predictions3['Probability_Unemployed']) / 3

# Creating a DataFrame with 'Person_id' and the averaged probability
predictions_ensemble_df = pd.DataFrame({
    'Person_id': predictions1['Person_id'],
    'Probability_Unemployed': average_probabilities
})

# Path to save the ensemble CSV file
predictions_ensemble_csv_path = 'predictions_ensemble.csv'

# Saving the DataFrame to a CSV file
predictions_ensemble_df.to_csv(predictions_ensemble_csv_path, index=False)

print(f"Ensemble predictions saved to {predictions_ensemble_csv_path}")
