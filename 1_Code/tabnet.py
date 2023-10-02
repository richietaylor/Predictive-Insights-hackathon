import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
import numpy as np

# Load the training data
train_data = pd.read_csv('processed_train.csv')
test_data = pd.read_csv('processed_test.csv')

tabnet_params = {
    "n_d" : 64,
    "n_a" : 64,
    "n_steps" : 5,
    "gamma" : 1.5,
    "n_independent" : 2,
    "n_shared" : 2,
    "cat_idxs" : cat_idxs,
    "cat_dims" : cat_dims,
    "cat_emb_dim" : 1,
    "lambda_sparse" : 1e-4,
    "momentum" : 0.3,
    "clip_value" : 2.,
    "optimizer_fn" : torch.optim.Adam,
    "optimizer_params" :dict(lr=2e-2)
}

# Removing unnecessary columns
train_data.drop(columns=['Person_id'], inplace=True)

# Identify categorical columns
categorical_cols = train_data.columns[train_data.dtypes == 'object'].tolist()

# Perform label encoding on categorical columns
for col in categorical_cols:
    full_data = pd.concat([train_data[col], test_data[col]], axis=0)
    encoded = pd.factorize(full_data)[0]
    
    train_data[col] = encoded[:len(train_data)]
    test_data[col] = encoded[len(train_data):]

# Separating the dependent and independent variables
X_train = train_data.drop(columns=['Target']).values
y_train = train_data['Target'].values

# TabNet unsupervised pretraining
unsupervised_model = TabNetPretrainer(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type='entmax'
)
unsupervised_model.fit(
    X_train=X_train,
    max_epochs=100,
    patience=10,
    batch_size=100,
    virtual_batch_size=100,
    num_workers=0,
    drop_last=False
)

# Stratified k-fold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
roc_aucs = []

for train_index, val_index in skf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # Transfer pre-trained weights to TabNetClassifier and train
    clf = TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":5, "gamma":0.5},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax'
    )
    unsupervised_model.save_model('pretrained_tabnet_weights')
    clf.load_model('pretrained_tabnet_weights')

    
    clf.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        eval_name=["val"],
        eval_metric=["auc"],
        max_epochs=100,
        patience=10,
        batch_size=100,
        virtual_batch_size=50,
        num_workers=0,
        drop_last=False
    )
    
    # Evaluate the model
    val_preds = clf.predict_proba(X_val_fold)[:, 1]
    roc_auc = roc_auc_score(y_val_fold, val_preds)
    roc_aucs.append(roc_auc)

# Average ROC AUC
average_roc_auc = np.mean(roc_aucs)
print(f"Average ROC AUC across {n_splits} folds: {average_roc_auc:.4f}")

# Train TabNet on the entire training dataset using the best hyperparameters found
final_clf = TabNetClassifier(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":10, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax'
)
final_clf.load_model(unsupervised_model)
final_clf.fit(
    X_train, y_train,
    max_epochs=100,
    patience=10,
    batch_size=256,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

# Making predictions on the test data
X_test = test_data.drop(columns=['Person_id']).values
y_test_prob_pred_tabnet = final_clf.predict_proba(X_test)[:, 1]

# Create a DataFrame for the predictions
predictions_prob_df = pd.DataFrame({
    'Person_id': test_data['Person_id'],
    'Probability_Unemployed': y_test_prob_pred_tabnet
})

# Save predictions to CSV
predictions_prob_df.to_csv('tabnet_predictions.csv', index=False)