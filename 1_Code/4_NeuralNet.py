import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import pandas as pd
from imblearn.over_sampling import ADASYN
# Load training dataset
train_data = pd.read_csv("processed_train_advanced_train.csv")
X_train = train_data.drop(columns=["Person_id", "Target"]).values
y_train = train_data["Target"].values

# Apply ADASYN to the training data
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

# Normalize the training data
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_resampled)

# Convert training data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
y_train_tensor = torch.tensor(y_resampled[:, None], dtype=torch.float32)

# Load validation dataset (from processed_train_advanced_test.csv)
val_data = pd.read_csv("processed_train_advanced_test.csv")
X_val = val_data.drop(columns=["Person_id", "Target"]).values
y_val = val_data["Target"].values

# Normalize the validation data using the same scaler
X_val_normalized = scaler.transform(X_val)

# Convert validation data to PyTorch tensors
X_val_tensor = torch.tensor(X_val_normalized, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val[:, None], dtype=torch.float32)
# Create DataLoader
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

num_features = X_train_normalized.shape[1]

# Define the Neural Network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, dropout_prob=0.5):
        super(NeuralNetwork, self).__init__()
        
        # First layer
        self.layer1 = nn.Linear(input_dim, 10)
        self.batch_norm1 = nn.BatchNorm1d(10)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        
        # Second layer
        self.layer2 = nn.Linear(10, 5)
        self.batch_norm2 = nn.BatchNorm1d(5)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        
        # Third layer (output layer)
        self.layer3 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.layer2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.sigmoid(self.layer3(x))
        return x

model = NeuralNetwork(num_features)

# Loss and Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Parameters for early stopping
patience = 10
best_val_loss = float('inf')
counter = 0


# Training loop
epochs = 300
# Training loop with early stopping and gradient clipping
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
    
    # Validation loss for early stopping
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
    model.train()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    
    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break


# Validate
model.eval()
with torch.no_grad():
    val_predictions = model(X_val_tensor)
    score = roc_auc_score(y_val_tensor, val_predictions)
    print(f"AUC-ROC Score on validation data: {score:.4f}")



# Load the test advanced dataset
test_advanced_data = pd.read_csv("processed_test_advanced.csv")
X_test_advanced = test_advanced_data.drop(columns=["Person_id"]).values

# Normalize the test data using the same scaler used for training and validation data
X_test_advanced_normalized = scaler.transform(X_test_advanced)

# Convert the test data to PyTorch tensor
X_test_advanced_tensor = torch.tensor(X_test_advanced_normalized, dtype=torch.float32)

# Make predictions with the trained model
model.eval()
with torch.no_grad():
    test_advanced_predictions = model(X_test_advanced_tensor)

# Save the predictions to neuralnet.csv
predictions_df = pd.DataFrame({
    'Person_id': test_advanced_data['Person_id'],
    'Probability_Unemployed': test_advanced_predictions.numpy().flatten()
})

predictions_df.to_csv('neuralnet.csv', index=False)
print("Predictions saved to neuralnet.csv")