import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Load the training data
train_data = pd.read_csv('Train.csv')

# Handling Missing Values for Numerical Columns
train_data.fillna(train_data.mean(numeric_only=True), inplace=True)

# Encoding Categorical Variables
label_encoders = {}
for column in train_data.select_dtypes(include=['object']).columns:
    if column not in ['Person_id', 'Survey_date']:
        le = LabelEncoder()
        train_data[column] = le.fit_transform(train_data[column])
        label_encoders[column] = le

# Removing unnecessary columns
train_data.drop(columns=['Person_id', 'Survey_date'], inplace=True)

# Separating the dependent and independent variables
X_train = train_data.drop(columns=['Target'])
y_train = train_data['Target']

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Splitting the dataset into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Converting to PyTorch tensors
train_dataset = TensorDataset(torch.tensor(X_train_split, dtype=torch.float32), torch.tensor(y_train_split.values, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val_split, dtype=torch.float32), torch.tensor(y_val_split.values, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Building the neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.fc(x)

model = NeuralNetwork(X_train_scaled.shape[1])

# Loss and optimizer
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Training the model
for epoch in range(10):
    model.train()
    for batch_x, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_x).squeeze()
        loss = loss_fn(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Load the test data
test_data = pd.read_csv('Test.csv')

# Preprocessing the test data (similar to the training data)
# ...
# Handling Missing Values for Numerical Columns in the test data
test_data.fillna(train_data.mean(numeric_only=True), inplace=True)

# Handling Missing Values for Categorical Columns and Encoding in the test data
for column, le in label_encoders.items():
    mode_value = train_data[column].mode()[0]
    test_data[column].fillna(mode_value, inplace=True)
    test_data[column] = test_data[column].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

# Removing unnecessary columns from the test data
test_data_processed = test_data.drop(columns=['Person_id', 'Survey_date'])

# Standardize the test features using the same scaler fit on the training data
test_data_processed = scaler.transform(test_data_processed)



# Making predictions on the test data
test_data_tensor = torch.tensor(test_data_processed, dtype=torch.float32)
model.eval()
with torch.no_grad():
    y_test_prob_pred_torch = model(test_data_tensor)

# Creating a DataFrame with "Person_id" and the predicted probability of unemployment
predictions_prob_df = pd.DataFrame({
    'Person_id': test_data['Person_id'],
    'Probability_Unemployed': y_test_prob_pred_torch.numpy().flatten()
})

# Path to save the CSV file
predictions_csv_path = 'predictions4.csv'

# Saving the DataFrame to a CSV file
predictions_prob_df.to_csv(predictions_csv_path, index=False)

print(f"Predictions saved to {predictions_csv_path}")
