import pandas as pd

# List of paths to the CSV files containing the predictions
predictions_files = [
    'predictions1.csv',
    'predictions2.csv',
    'predictions3.csv',
    # Add more files here as needed
]

# Read the first file to initialize
predictions_ensemble_df = pd.read_csv(predictions_files[0])
average_probabilities = predictions_ensemble_df['Probability_Unemployed']

# Iterate through the rest of the files, adding the probabilities
for file_path in predictions_files[1:]:
    predictions = pd.read_csv(file_path)
    
    # Making sure the 'Person_id' columns match
    if not (predictions_ensemble_df['Person_id'] == predictions['Person_id']).all():
        raise ValueError(f"The 'Person_id' column in {file_path} does not match the others.")

    average_probabilities += predictions['Probability_Unemployed']

# Divide by the number of files to get the average
average_probabilities /= len(predictions_files)

# Update the 'Probability_Unemployed' column with the averaged probabilities
predictions_ensemble_df['Probability_Unemployed'] = average_probabilities

# Path to save the ensemble CSV file
predictions_ensemble_csv_path = 'predictions_ensemble.csv'

# Saving the DataFrame to a CSV file
predictions_ensemble_df.to_csv(predictions_ensemble_csv_path, index=False)

print(f"Ensemble predictions saved to {predictions_ensemble_csv_path}")
