import torch
import torch.nn as nn
import torch.optim as optim

class PyTorchMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PyTorchMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset

class SklearnCompatibleMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, hidden_dim, output_dim, epochs=10, lr=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device("cuda:0")
        self.model = PyTorchMLP(self.input_dim, self.hidden_dim, self.output_dim).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X, y):
        # Convert X and y to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, (batch_X, batch_y) in enumerate(dataloader):
                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch_X).squeeze()
                loss = self.criterion(outputs, batch_y.float())

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                # Print statistics
                running_loss += loss.item()
                if i % 10 == 9:  # Print every 10 batches
                    print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 10:.3f}")
                    running_loss = 0.0

            # Print average loss for the epoch
            avg_loss = running_loss / len(dataloader)
            print(f"Epoch [{epoch + 1}/{self.epochs}] - Average Loss: {avg_loss:.3f}")

        return self

    def predict(self, X):
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = outputs.max(1)
        return predicted.cpu().numpy()

    def predict_proba(self, X):
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probas = nn.Sigmoid(outputs).squeeze()
        return probas.cpu().numpy()
    
