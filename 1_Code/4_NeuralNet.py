import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Hyperparameters
batch_size = 128
epochs = 100
learning_rate = 0.0001
hidden_layers = [512,512]  # Number of units in hidden layers
dropout_rate = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data.fillna(data.min(numeric_only=True), inplace=True)
    columns_to_encode = [column for column in data.select_dtypes(include=["object"]).columns if column not in ["Person_id", "Survey_date"]]
    label_encoders = {}  # Keep track of the encoders for each column
    for column in columns_to_encode:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    data.drop(columns=["Person_id", "Survey_date"], inplace=True)
    return data, label_encoders

def split_and_scale(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val, scaler

def add_noise(data, noise_level=0.1):
    noise = np.random.normal(scale=noise_level, size=data.shape)
    return data + noise

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout_rate):
        super(NeuralNetwork, self).__init__()
        layers = [nn.Linear(input_size, hidden_layers[0]),
                  nn.ReLU(),
                  nn.BatchNorm1d(hidden_layers[0]),
                  nn.Dropout(dropout_rate)]
        for i in range(len(hidden_layers) - 1):
            layers += [nn.Linear(hidden_layers[i], hidden_layers[i + 1]),
                       nn.ReLU(),
                       nn.BatchNorm1d(hidden_layers[i + 1]),
                       nn.Dropout(dropout_rate)]
        layers += [nn.Linear(hidden_layers[-1], 1), nn.Sigmoid()]
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

def calculate_metrics(outputs, labels):
    predicted = (outputs > 0.5).float()
    correct = (predicted == labels).float().sum().item()
    accuracy = correct / len(labels)
    precision = precision_score(labels.cpu().numpy(), predicted.cpu().numpy())
    recall = recall_score(labels.cpu().numpy(), predicted.cpu().numpy())
    f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy())
    roc_auc = roc_auc_score(labels.cpu().numpy(), outputs.detach().cpu().numpy()) if len(np.unique(labels)) > 1 else None

    return accuracy, precision, recall, f1, roc_auc

def train_model(train_loader, val_loader, input_size):
    model = NeuralNetwork(input_size, hidden_layers, dropout_rate)
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=0.4,dampening=0.2)
    scheduler = torch.optim.lr_scheduler.StepLR(gamma=1,step_size=5,optimizer=optimizer)
    # Training the model
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_accuracy = 0
        train_precision = 0
        train_recall = 0
        train_f1 = 0
        train_roc_auc = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x).squeeze()
            loss = loss_fn(outputs, batch_y)
            train_loss += loss.item()
            metrics = calculate_metrics(outputs, batch_y)
            train_accuracy += metrics[0]
            train_precision += metrics[1]
            train_recall += metrics[2]
            train_f1 += metrics[3]
            if metrics[4] is not None:
                train_roc_auc += metrics[4]
            optimizer.zero_grad()
            loss.backward()
            scheduler.step()

        # Calculate averages
        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        train_precision /= len(train_loader)
        train_recall /= len(train_loader)
        train_f1 /= len(train_loader)
        train_roc_auc /= len(train_loader)

        # Validation loss and accuracy
        model.eval()
        val_loss = 0
        val_accuracy = 0
        val_precision = 0
        val_recall = 0
        val_f1 = 0
        val_roc_auc = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x).squeeze()
                loss = loss_fn(outputs, batch_y)
                val_loss += loss.item()
                metrics = calculate_metrics(outputs, batch_y)
                val_accuracy += metrics[0]
                val_precision += metrics[1]
                val_recall += metrics[2]
                val_f1 += metrics[3]
                if metrics[4] is not None:
                    val_roc_auc += metrics[4]

        # Calculate averages
        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        val_precision /= len(val_loader)
        val_recall /= len(val_loader)
        val_f1 /= len(val_loader)
        val_roc_auc /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1-Score: {train_f1:.4f}, ROC-AUC: {train_roc_auc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}, ROC-AUC: {val_roc_auc:.4f}")

    return model

def make_predictions(model, test_data, encoders, scaler):
    # Handling Missing Values for Numerical Columns in the test data
    test_data.fillna(test_data.min(numeric_only=True), inplace=True)
    # Handling Missing Values for Categorical Columns and Encoding in the test data
    for column, le in encoders.items():
        mode_value = le.classes_[0]
        test_data[column].fillna(mode_value, inplace=True)
        test_data[column] = le.transform(test_data[column])
    # Removing unnecessary columns from the test data
    test_data_processed = test_data.drop(columns=["Person_id", "Survey_date"])
    # Standardize the test features using the same scaler fit on the training data
    test_data_processed = scaler.transform(test_data_processed)
    # Making predictions on the test data
    test_data_tensor = torch.tensor(test_data_processed, dtype=torch.float32).to(device)  # Send to GPU
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
    return predictions_prob_df

def main():
     # Preprocess training data
    train_data, encoders = preprocess_data("Train.csv")
    X_train, X_val, y_train, y_val, scaler = split_and_scale(train_data, "Target")

    # Apply noise to the training data
    X_train_noisy = add_noise(X_train)

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train_noisy, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val.values, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Train the model
    model = train_model(train_loader, val_loader, X_train.shape[1])

    # Load and preprocess test data
    test_data = pd.read_csv("Test.csv")
    predictions_prob_df = make_predictions(model, test_data, encoders, scaler)

    # Path to save the CSV file
    predictions_csv_path = "neuralnet.csv"
    # Saving the DataFrame to a CSV file
    predictions_prob_df.to_csv(predictions_csv_path, index=False)

    print(f"Predictions saved to {predictions_csv_path}")

if __name__ == "__main__":
    main()
