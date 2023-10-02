import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import os

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def find_lr(model, train_loader, criterion, optimizer, init_value=1e-10, final_value=1.0, device="cuda"):
    """
    Finds a good learning rate for a given model and dataset.
    
    Parameters:
        model (torch.nn.Module): The neural network model.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        init_value (float, optional): The initial learning rate. Defaults to 1e-7.
        final_value (float, optional): The final learning rate. Defaults to 1.0.
        device (str, optional): The device (e.g. "cuda" or "cpu"). Defaults to "cuda".
        
    Returns:
        tuple: Logarithm of the learning rates and the corresponding losses.
    """
    
    number_in_epoch = len(train_loader) - 1
    update_step = (final_value / init_value) ** (1 / number_in_epoch)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    best_loss = float('inf')
    batch_num = 0
    losses = []
    log_lrs = []
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        batch_num += 1
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        
        # Stop if the loss is exploding
        if batch_num > 1 and loss.item() > 4 * best_loss:
            return log_lrs[10:-5], losses[10:-5]
        
        # Record the best loss
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        # Store the values
        losses.append(loss.item())
        log_lrs.append(np.log10(lr))
        
        # Do backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Update the lr for the next step
        lr *= update_step
        optimizer.param_groups[0]['lr'] = lr
    
    return log_lrs[10:-5], losses[10:-5]

device = torch.device("cuda:0")
# Load the training data
train_data = pd.read_csv('processed_train.csv')
train_data.drop(columns=['Person_id'], inplace=True)
X = train_data.drop(columns=['Target']).to_numpy()
y = train_data['Target'].to_numpy()

# Scaling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0,path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# Neural network with skip connection
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.5):
        super(NeuralNetwork, self).__init__()
        
        # Layers
        self.fc1 = nn.Linear(input_dim, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 500)  # Adding another layer for depth
        self.fc_skip1 = nn.Linear(500, 500)  # Skip connection for the first layer
        self.fc_skip2 = nn.Linear(500, 500)  # Skip connection for the second layer
        self.fc4 = nn.Linear(500, 1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(500)
        self.bn2 = nn.BatchNorm1d(500)
        self.bn3 = nn.BatchNorm1d(500)  # Adding another batch normalization layer for the new layer
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)  # Dropout for the new layer

    def forward(self, x):
        # Initial transformation
        x_initial = self.dropout1(self.bn1(torch.relu(self.fc1(x))))
        
        # Transformation through fc2
        x_transformed1 = self.dropout2(self.bn2(torch.relu(self.fc2(x_initial))))
        
        # Skip connection after fc2
        x_skip1 = self.fc_skip1(x_initial)
        
        # Combining the skip connection and fc2 output
        x = x_transformed1 + x_skip1
        
        # Transformation through fc3
        x_transformed2 = self.dropout3(self.bn3(torch.relu(self.fc3(x))))
        
        # Skip connection after fc3
        x_skip2 = self.fc_skip2(x)
        
        # Combining the skip connection and fc3 output
        x = x_transformed2 + x_skip2
        
        # Final transformation and sigmoid activation
        x = torch.sigmoid(self.fc4(x))
        
        return x


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        X_batch, y_batch = to_device(batch, device)
        optimizer.zero_grad()
        y_pred = model(X_batch).squeeze()
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            X_batch, y_batch = to_device(batch, device)
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    return total_loss / len(loader), all_predictions, all_targets


# # Sample DataLoader for the LR finder
sample_dataset = TensorDataset(torch.FloatTensor(X_scaled), torch.FloatTensor(y))
sample_loader = DataLoader(sample_dataset, batch_size=50, shuffle=True)

# Create a sample model and optimizer for the LR finder
sample_model = NeuralNetwork(X_scaled.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(sample_model.parameters())

# Use the LR finder
logs, losses = find_lr(sample_model, sample_loader, criterion, optimizer)

# Plot the results
import matplotlib.pyplot as plt
plt.plot(logs, losses)
plt.xlabel("Log Learning Rate")
plt.ylabel("Loss")
plt.show()


# Hyperparameters
learning_rate = 0.00001
epochs = 100
batch_size = 100



# Stratified k-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
roc_aucs = []

fold = 1
for train_index, val_index in skf.split(X_scaled, y):
    print(f"\nTraining on Fold {fold}\n{'='*20}\n")
    X_train_fold, X_val_fold = X_scaled[train_index], X_scaled[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_fold)
    y_train_tensor = torch.FloatTensor(y_train_fold)
    X_val_tensor = torch.FloatTensor(X_val_fold)
    y_val_tensor = torch.FloatTensor(y_val_fold)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    best_roc_auc = 0.0

    # Model, Loss, Optimizer
    model = NeuralNetwork(X_train_tensor.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, 
                                          steps_per_epoch=len(train_loader), 
                                          epochs=epochs, 
                                          anneal_strategy='cos', 
                                          pct_start=0.1, # 10% of training is used for warm-up
                                          div_factor=25)  # initial_lr = max_lr / div_factor
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    # Training loop
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_predictions, val_targets = evaluate(model, val_loader, criterion, device)
        roc_auc = roc_auc_score(val_targets, val_predictions)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_roc_auc = roc_auc
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, ROC AUC: {roc_auc:.4f}")

        # early_stopping(val_loss, model)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

    # Check if checkpoint exists before loading
    if os.path.isfile('checkpoint.pt'):
        model.load_state_dict(torch.load('checkpoint.pt'))
    else:
        print("No checkpoint found. Using the model from the last epoch.")
    val_loss, val_predictions, val_targets = evaluate(model, val_loader, criterion, device)
    roc_auc = roc_auc_score(val_targets, val_predictions)
    roc_aucs.append(roc_auc)

    fold += 1

average_roc_auc = np.mean(roc_aucs)
print(f"\nEstimated ROC AUC across {skf.n_splits} folds: {average_roc_auc:.4f}")
