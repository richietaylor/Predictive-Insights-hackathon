import pandas as pd

def add_education_column(csv_file_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Check if the required columns exist in the DataFrame
    required_columns = ["Matric", "Diploma", "Degree"]
    for col in required_columns:
        if col not in df.columns:
            print(f"Column '{col}' not found in the CSV file.")
            return None
    
    # Create the 'Education' column based on extracted values
    df["Education"] = df.apply(
        lambda row: "Matric" if row["Matric"] == 1
        else "Diploma" if row["Diploma"] == 1
        else "Degree" if row["Degree"] == 1
        else "Other", axis=1
    )
    
    return df

# Replace 'your_data.csv' with the actual path to your CSV file
csv_file_path = 'Train.csv'
df_with_education = add_education_column(csv_file_path)

if df_with_education is not None:
    print("Education column added successfully:")
    print(df_with_education)
