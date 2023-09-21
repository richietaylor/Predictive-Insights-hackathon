@echo off
python ./1_Code/processor.py
python ./1_Code/1_lightgbm.py
python ./1_Code/2_xgboost.py
python ./1_Code/3_cboost.py
python ./1_Code/5_randomForest.py
python ./1_Code/merge.py