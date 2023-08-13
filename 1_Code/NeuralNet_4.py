import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def add_noise(data, noise_level=0.1):
    noise = np.random.normal(scale=noise_level, size=data.shape)
    return data + noise

# Hyperparameters
batch_size = 128
epochs = 100
learning_rate = 0.001
hidden_layers = [64,32]  # Number of units in hidden layers

# Load the training data
train_data = pd.read_csv("Train.csv")

# Handling Missing Values for Numerical Columns
train_data.fillna(train_data.min(numeric_only=True), inplace=True)

# Encoding Categorical Variables
label_encoders = {}
for column in train_data.select_dtypes(include=["object"]).columns:
    if column not in ["Person_id", "Survey_date"]:
        le = LabelEncoder()
        train_data[column] = le.fit_transform(train_data[column])
        label_encoders[column] = le

# Removing unnecessary columns
train_data.drop(columns=["Person_id", "Survey_date"], inplace=True)

# Separating the dependent and independent variables
X_train = train_data.drop(columns=["Target"])
y_train = train_data["Target"]

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Splitting the dataset into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

# Apply noise to the training data
X_train_noisy = add_noise(X_train_split)

# Converting to PyTorch tensors
train_dataset = TensorDataset(
    torch.tensor(X_train_noisy, dtype=torch.float32),
    torch.tensor(y_train_split.values, dtype=torch.float32),
)
val_dataset = TensorDataset(
    torch.tensor(X_val_split, dtype=torch.float32),
    torch.tensor(y_val_split.values, dtype=torch.float32),
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout_rate):
        super(NeuralNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.BatchNorm1d(hidden_layers[0]))  # Batch Normalization for the first hidden layer
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_layers[i + 1]))  # Batch Normalization for subsequent hidden layers
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_layers[-1], 1))
        layers.append(nn.Sigmoid())
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

model = NeuralNetwork(X_train_scaled.shape[1], hidden_layers,dropout_rate=0.2)

# Loss and optimizer
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.01)


# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    predicted = (outputs > 0.5).float()
    correct = (predicted == labels).float().sum()
    accuracy = correct / len(labels)
    return accuracy



# Training the model
for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_accuracy = 0
    for batch_x, batch_y in train_loader:
        outputs = model(batch_x).squeeze()
        loss = loss_fn(outputs, batch_y)
        train_loss += loss.item()
        train_accuracy += calculate_accuracy(outputs, batch_y).item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)
    train_accuracy /= len(train_loader)

    # Validation loss and accuracy
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x).squeeze()
            loss = loss_fn(outputs, batch_y)
            val_loss += loss.item()
            val_accuracy += calculate_accuracy(outputs, batch_y).item()

    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Load the test data
test_data = pd.read_csv("Test.csv")

# Handling Missing Values for Numerical Columns in the test data
test_data.fillna(train_data.min(numeric_only=True), inplace=True)

# Handling Missing Values for Categorical Columns and Encoding in the test data
for column, le in label_encoders.items():
    mode_value = train_data[column].mode()[0]
    test_data[column].fillna(mode_value, inplace=True)
    test_data[column] = test_data[column].apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )

# Removing unnecessary columns from the test data
test_data_processed = test_data.drop(columns=["Person_id", "Survey_date"])

# Standardize the test features using the same scaler fit on the training data
test_data_processed = scaler.transform(test_data_processed)

# Making predictions on the test data
test_data_tensor = torch.tensor(test_data_processed, dtype=torch.float32)
model.eval()
with torch.no_grad():
    y_test_prob_pred_torch = model(test_data_tensor)

# Creating a DataFrame with "Person_id" and the predicted probability of unemployment
predictions_prob_df = pd.DataFrame(
    {
        "Person_id": test_data["Person_id"],
        "Probability_Unemployed": y_test_prob_pred_torch.numpy().flatten(),
    }
)

# Path to save the CSV file
predictions_csv_path = "predictions4.csv"

# Saving the DataFrame to a CSV file
predictions_prob_df.to_csv(predictions_csv_path, index=False)

print(f"Predictions saved to {predictions_csv_path}")
