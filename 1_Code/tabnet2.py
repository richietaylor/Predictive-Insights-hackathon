#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
import torch
from sklearn.metrics import roc_auc_score, accuracy_score

import copy
import torch

train = pd.read_csv('processed_train.csv')
test = pd.read_csv('processed_test.csv')

# remove the only 5 cover type target
# train = train[train.Cover_Type!=5].reset_index(drop=True)
print(train.shape)
print(test.shape)


a = train.nunique().reset_index(drop=False).rename(columns={"index": "feat_name", 0: "count"})

# drop columns with a single value
drop_cols = ["Person_id"] + list(a[a["count"] < 2 ].feat_name)
target = ["Target"]

# categorical features are columns with small modalities
cat_features = [col for col in list(a[a["count"] < 10 ].feat_name) if col not in drop_cols+target]
num_features = [col for col in train.columns if col not in drop_cols+target+cat_features]

features = cat_features + num_features


# This is only needed if using embeddings (not used at the moment)

train[cat_features] = train[cat_features].astype(str)
test[cat_features] = test[cat_features].astype(str)


from sklearn.preprocessing import LabelEncoder

categorical_columns = []
categorical_dims =  {}
for col in cat_features:
    l_enc = LabelEncoder()
    train[col] = train[col].fillna("VV_likely")
    train[col] = l_enc.fit_transform(train[col].values)
    categorical_columns.append(col)
    categorical_dims[col] = len(l_enc.classes_)
    
    test[col] = l_enc.transform(test[col].values)

cat_idxs = [] #[ i for i, f in enumerate(features) if f in cat_features]
cat_dims = [] #[ categorical_dims[f] for i, f in enumerate(features) if f in cat_features]

X_test = test[features].values

BS = 8192*2
VBS = BS #512
BS = 200
max_epochs=50

tabnet_params = {"n_d" : 64,
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
                 "optimizer_params" :dict(lr=2e-2),}


params = copy.deepcopy(tabnet_params)
params["scheduler_fn"]=torch.optim.lr_scheduler.StepLR
params["scheduler_params"]={"is_batch_level":False,
                            "gamma":0.95,
                            "step_size": 1,}

# Pretrain the model on test set

X_unsup_valid = train[features].values[:100000]
params = tabnet_params.copy()

unsupervised_model = TabNetPretrainer(device_name='cuda',**params)

unsupervised_model.fit(
    X_train=X_test,
    eval_set=[X_unsup_valid],
    pretraining_ratio=0.8,
    max_epochs=50,
    patience=5,
    batch_size=100,
    virtual_batch_size=100
)

# Split for cross validation or single validation
from sklearn.model_selection import StratifiedKFold

N_SPLITS=5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

cv_preds = np.zeros((X_test.shape[0], N_SPLITS))

fold_idx=0
for train_idx, val_idx in skf.split(train, train[target]):

    # Create the numpy datasets

    X_train = train.loc[train_idx, features].values
    Y_train = train.loc[train_idx, target].values.reshape(-1)

    X_val = train.loc[val_idx, features].values
    Y_val = train.loc[val_idx, target].values.reshape(-1)

    # Train a tabnet classifier

    params = copy.deepcopy(tabnet_params)

    # Scheduling scheme here is the only part not similar to the original paper
    # but the dataset is not exactly the same

    params["scheduler_fn"]=torch.optim.lr_scheduler.StepLR
    params["scheduler_params"]={"is_batch_level":False,
                                "gamma":0.95,
                                "step_size": 5,}
    # params["scheduler_fn"]=torch.optim.lr_scheduler.OneCycleLR
    # params["scheduler_params"]={"is_batch_level":True,
    #                             "max_lr":5e-2,
    #                             "steps_per_epoch":int(X_train.shape[0] / BS),
    #                             "epochs":max_epochs}

    clf = TabNetClassifier(device_name='cuda',**params)

    clf.fit(
        X_train,
        Y_train,
        eval_set=[(X_train, Y_train), (X_val, Y_val)],
        eval_name=['train', 'valid'],
        eval_metric=['auc'],
        max_epochs=max_epochs,
        patience=20,
        drop_last=True,
        batch_size=BS,
        virtual_batch_size=BS,
    #     weights=1,
        from_unsupervised=unsupervised_model
    )
    
    preds = clf.predict(X_test)
    cv_preds[:, fold_idx] = preds
    fold_idx+=1
    
# Predict the probabilities on the test set
predicted_probs = clf.predict_proba(X_test)[:, 1]

# Save the predictions to a CSV file
predictions_df = pd.DataFrame(
    {"Person_id": test["Person_id"], "Probability_Unemployed": predicted_probs}
)
predictions_df.to_csv("tabnet_predictions.csv", index=False)