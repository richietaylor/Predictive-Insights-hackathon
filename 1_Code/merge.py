import pandas as pd

# List of paths to the CSV files containing the predictions
predictions_files = [
    'predictions1.csv',
    'predictions2.csv',
    'predictions3.csv',
    'predictions4.csv',
    'predictions5.csv',

    # Add more files here as needed
]

# List of weights for each CSV file (should be the same length as predictions_files)
weights = [
    1.0, 
    1.0, 
    2.0, 
    0.0,
    2.0,
    
]

# Read the first file to initialize
predictions_ensemble_df = pd.read_csv(predictions_files[0])
weighted_probabilities = predictions_ensemble_df['Probability_Unemployed'] * weights[0]

# Iterate through the rest of the files, adding the weighted probabilities
for idx, file_path in enumerate(predictions_files[1:]):
    predictions = pd.read_csv(file_path)
    
    # Making sure the 'Person_id' columns match
    if not (predictions_ensemble_df['Person_id'] == predictions['Person_id']).all():
        raise ValueError(f"The 'Person_id' column in {file_path} does not match the others.")

    weighted_probabilities += predictions['Probability_Unemployed'] * weights[idx + 1]

# Divide by the sum of weights to get the weighted average
weighted_probabilities /= sum(weights)

# Update the 'Probability_Unemployed' column with the weighted probabilities
predictions_ensemble_df['Probability_Unemployed'] = weighted_probabilities

# Save the result to a new CSV file
predictions_ensemble_df.to_csv('ensemble_predictions.csv', index=False)
