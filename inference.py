#!/usr/bin/env python
# coding: utf-8

# inference.py: 
# Use optimized hyperparameters and radiomics features to predict test data.
# (C) J. Bleker & C. Roest (2023)

import sys

# Import sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Import other packages
import pandas as pd
import numpy as np

# Print import statements
import xgboost as xgb

# Enter the best parameters determined in the hyperparameter optimization
# Below values are shown as an example
best_params = {
    "booster": "gbtree",
    "subsample": 0.95,
    "reg_alpha": 2.7998614888500217,
    "reg_lambda": 10.17242426973007,
    "colsample_bytree": 0.7802385309471809,
    "min_child_weight": 11,
    "max_depth": 3,
    "learning_rate": 0.07100000000000001,
    "gamma": 1.6899192782900374,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "scale_pos_weight": 1.5833333333333333,
    "num_boost_round": 392,
}

# Enter the optimal features determined in the hyperparameter optimization
# Below values are shown as an example
features = [
    "log-sigma-3-0-mm-3D_glszm_SmallAreaHighGrayLevelEmphasis-DWI-b1400",
    "logarithm_glcm_SumEntropy-ADC",
    "log-sigma-3-0-mm-3D_glszm_ZonePercentage-DWI-b1400",
    # ...
]


# Define job script input for optuna
if __name__ == "__main__":
    # Set this to the CSV input file containing radiomics features
    INPUT_FILE = sys.argv[1]

    # Set this to the desired output CSV file, we will write the predicted likelihood
    # scores to this file for each test case.
    OUT_FILE = sys.argv[2]

    print(end="> Importing data and splitting\n", flush=True)
    all_data_df = pd.read_csv(INPUT_FILE, index_col=0)

    # Take x and y
    x = all_data_df.loc[:, all_data_df.columns != "y_label"]
    y = all_data_df.loc[:, "y_label"]

    # Split in train and test
    x_TRAIN, x_test, y_TRAIN, y_test = train_test_split(
        x, y, test_size=0.2, random_state=13
    )
    print("Done.", flush=True)

    # Split feature parameters from the model parameters
    param = {
        key: best_params[key]
        for key in best_params
        if key not in ["fs_features", "fs_imp", "fs_repeat", "fs_metric"]
    }

    # Datasets
    print("Making DMatrices", flush=True)
    final_train_x = x_TRAIN[features]
    train_set = xgb.DMatrix(final_train_x, label=y_TRAIN)

    final_test_x = x_test[features]
    test_set = xgb.DMatrix(final_test_x, label=y_test)

    # fit final model with best results
    print("Fitting model..", flush=True)
    final_model = xgb.train(
        dtrain=train_set, params=param, num_boost_round=best_params["num_boost_round"]
    )

    # Obtain AUC, and write test predictions to a CSV file
    pred_test = final_model.predict(test_set)
    test_auc = roc_auc_score(y_test, pred_test)
    print("Test score {}".format(test_auc))
    with open(OUT_FILE, "w+") as f:
        f.write("pred;label\n")
        for p, l in zip(pred_test, y_test):
            print(p, l)
            f.write(f"{p};{l}\n")
