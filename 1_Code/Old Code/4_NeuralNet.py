import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import ADASYN
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 1. Data Preparation
train_data = pd.read_csv("processed_train.csv")
X_train = train_data.drop(columns=["Person_id", "Target"]).values
y_train = train_data["Target"].values

# ADASYN
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

# Normalize
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_resampled)

# Split the normalized data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_normalized, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Base Model Predictions

# Initialize base models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(n_estimators=100, random_state=42)
lgbm = LGBMClassifier(n_estimators=100, random_state=42)

# Placeholder for out-of-fold predictions
oof_rf = np.zeros((X_train_split.shape[0],))
oof_xgb = np.zeros((X_train_split.shape[0],))
oof_lgbm = np.zeros((X_train_split.shape[0],))


# Stratified K-Fold for out-of-fold predictions
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Stratified K-Fold for out-of-fold predictions
for train_idx, val_idx in kfold.split(X_train_split, y_train_split):
    X_train_fold, X_val_fold = X_train_split[train_idx], X_train_split[val_idx]
    y_train_fold, y_val_fold = y_train_split[train_idx], y_train_split[val_idx]
    
    # Random Forest
    rf.fit(X_train_fold, y_train_fold)
    oof_rf[val_idx] = rf.predict_proba(X_val_fold)[:, 1]
    
    # XGBoost
    xgb.fit(X_train_fold, y_train_fold)
    oof_xgb[val_idx] = xgb.predict_proba(X_val_fold)[:, 1]
    
    # LightGBM
    lgbm.fit(X_train_fold, y_train_fold)
    oof_lgbm[val_idx] = lgbm.predict_proba(X_val_fold)[:, 1]

# Append base model predictions to your training data
X_train_augmented = np.hstack((X_train_split, oof_rf.reshape(-1, 1), oof_xgb.reshape(-1, 1), oof_lgbm.reshape(-1, 1)))

# Convert augmented data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_augmented, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_split[:, None], dtype=torch.float32)

# Using the models to predict for the validation data
oof_val_rf = rf.predict_proba(X_val_split)[:, 1]
oof_val_xgb = xgb.predict_proba(X_val_split)[:, 1]
oof_val_lgbm = lgbm.predict_proba(X_val_split)[:, 1]
X_val_augmented = np.hstack((X_val_split, oof_val_rf.reshape(-1, 1), oof_val_xgb.reshape(-1, 1), oof_val_lgbm.reshape(-1, 1)))

# Convert the augmented validation data to PyTorch tensors
X_val_tensor = torch.tensor(X_val_augmented, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_split[:, None], dtype=torch.float32)

# DataLoader
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

# 2. Neural Network Architecture
num_features = X_train_augmented.shape[1]

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(50, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.seq(x)

model = NeuralNetwork(num_features)

# 3. Training Loop
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=0.01)
patience = 10
best_val_loss = float('inf')
counter = 0
epochs = 300

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Validation loss for early stopping
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_predictions = val_outputs.numpy().flatten()
        roc_auc = roc_auc_score(y_val_split, val_predictions)
        print(f"Validation ROC-AUC: {roc_auc:.4f}")
        val_loss = criterion(val_outputs, y_val_tensor)
    model.train()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# 4. Test Predictions
test_data = pd.read_csv("processed_test.csv")
X_test = test_data.drop(columns=["Person_id"]).values
X_test_normalized = scaler.transform(X_test)

# Retrain base models on the entire training dataset
rf.fit(X_train_normalized, y_resampled)
xgb.fit(X_train_normalized, y_resampled)
lgbm.fit(X_train_normalized, y_resampled)


# Don't forget to also adjust the augmented input for the test set:
test_rf = rf.predict_proba(X_test_normalized)[:, 1]
test_xgb = xgb.predict_proba(X_test_normalized)[:, 1]
test_lgbm = lgbm.predict_proba(X_test_normalized)[:, 1]
X_test_augmented = np.hstack((X_test_normalized, test_rf.reshape(-1, 1), test_xgb.reshape(-1, 1), test_lgbm.reshape(-1, 1)))

X_test_tensor = torch.tensor(X_test_augmented, dtype=torch.float32)

model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)

predictions_df = pd.DataFrame({
    'Person_id': test_data['Person_id'],
    'Probability_Unemployed': test_predictions.numpy().flatten()
})

predictions_df.to_csv('neuralnet.csv', index=False)
print("Predictions saved to neuralnet.csv")
