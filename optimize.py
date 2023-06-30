#!/usr/bin/env python
# coding: utf-8

# optimize.py: 
# Radiomics hyperparameter optimization using Extreme Gradient Boosting (XGB) and optuna.
# (C) J. Bleker & C. Roest (2023)

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from heapq import nlargest
from os import path
import numpy as np
import argparse
import csv

import xgboost as xgb
import optuna 

def objective(trial):
    """
    Objective function which gets optimized by optuna.
    Previous trials are used to suggest a good candidate from the hyperparameter 
    space and calculates the validation AUC.
    """
    
    # Define the parameter space to search
    parama = {
        'fs_features'    : trial.suggest_int('fs_features', 2, 150, 1, log=True),
        'fs_imp'       : trial.suggest_categorical('fs_imp', ['weight', 'gain', 'cover', 'total_gain', 'total_cover']),
        'fs_repeat'    : trial.suggest_int('fs_repeat', 3, 10),
        'fs_metric'     : trial.suggest_categorical('fs_metric', ['auc', 'error', 'aucpr', 'map']),
        'booster'          : trial.suggest_categorical('booster', ['gbtree', 'dart']),
        'subsample'        : trial.suggest_discrete_uniform('subsample', 0.6, 1.0, 0.05),
        'reg_alpha'            : trial.suggest_uniform('reg_alpha', 0.1, 50),
        'reg_lambda'           : trial.suggest_uniform('reg_lambda', 0.1, 50),
        'colsample_bytree' : trial.suggest_float("colsample_bytree", 0.1, 1.0),                
        'min_child_weight' : trial.suggest_int('min_child_weight', 2, 25),  
        'max_depth'       : trial.suggest_int('max_depth', 3, 18, 1),
        'learning_rate'  :  trial.suggest_discrete_uniform('learning_rate', 0.001, 0.1, 0.01),
        'gamma' : trial.suggest_uniform('gamma', 0.01, 3)
    }  

    # Define parameters that do not require optimization
    parama['objective'] = 'binary:logistic'
    parama['eval_metric'] = 'auc'
    
    # If the "dart" booster was suggested, suggest some dart-specific parameters
    if parama["booster"] == "dart":
        parama["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        parama["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        parama["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
        parama["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)         
       
    # Calculate the class weighting based on the proportion of labels
    class_frequency = y_TRAIN.value_counts()
    scale_pos_w = class_frequency[0]/class_frequency[1]
    parama['scale_pos_weight'] = scale_pos_w
       
    # Print the selected parameters in job report
    print("> Parameters (trial #{}):".format(trial.number))
    for key in parama:
        print("  - {k} :\t{v}".format(k=key.ljust(20), v=parama[key]), flush=True)   
        
    # Separate feature parameters from the model parameters  
    feature_params = {key: parama[key] for key in parama if key in ['fs_features', 'fs_imp', 'fs_repeat', 'fs_metric']}
    param = {key: parama[key] for key in parama if key not in ['fs_features', 'fs_imp', 'fs_repeat', 'fs_metric']} 

    # Create validation set for feat selection
    feat_set = xgb.DMatrix(x_TRAIN, label = y_TRAIN)

    # Fit model and get best features
    feat_imp_list = []
    for i in range(feature_params['fs_repeat']):
      print("Feature round {}".format(i+1))
      cv_results = xgb.cv(params=param, dtrain=feat_set, num_boost_round = 1500, nfold = 10, early_stopping_rounds=15, metrics=feature_params['fs_metric'], seed=i+1) 
      test_name = 'test-' + feature_params['fs_metric'] + '-mean'
      n_estimators = len(cv_results[test_name])
      bst = xgb.train(dtrain=feat_set, params=param, num_boost_round=n_estimators)
      feat_imp_list.append(bst.get_score(importance_type=feature_params['fs_imp']))
    
    # final features
    final_dict = {}
    for current_dict in feat_imp_list:
        for key, value in current_dict.items():
            if final_dict.get(key):
                final_dict[key] += value
            else:
                final_dict[key] = value
    
    # Select features
    features = nlargest(feature_params['fs_features'], final_dict, key = final_dict.get)
    
    # Catch cases where the number of features = 0
    if len(features) == 0:
      columns = list(x_TRAIN.columns)
      features = [columns[0]]
    
    # Datasets
    final_train_x = x_TRAIN[features]
    train_set = xgb.DMatrix(final_train_x, label = y_TRAIN) 
    
    final_test_x = x_test[features]
    test_set = xgb.DMatrix(final_test_x, label = y_test)
    
    # Fit model and get best trees
    xgb_results = xgb.cv(params=param, dtrain=train_set, num_boost_round = 4500, nfold = 5, early_stopping_rounds=45, metrics='auc', seed=25)
    num_boost = len(xgb_results['test-auc-mean'])
    score_opti = xgb_results['test-auc-mean'][num_boost-1]
    print('Cross validated final score {}'.format(score_opti))
    
    # fit final model with best results
    final_model = xgb.train(dtrain=train_set, params=param, num_boost_round=num_boost)
    param['num_boost_round'] = num_boost
    
    # The test-set performance is already calculated and stored here, but model 
    # selection is performed using the validation performance.
    test_pred = final_model.predict(test_set)
    test_auc = roc_auc_score(y_test, test_pred)
    print("Test score {}".format(test_auc))    

    # Write to the csv file ('a' means append)
    print("> Writing to", OUT_FILE, flush=True)
    of_connection = open(OUT_FILE, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([score_opti, test_auc, param, features, trial.number])
    of_connection.close()    
    return score_opti

# Define job script input for optuna 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name",    "-s", help="[required] Name of the experiment.")
    parser.add_argument("--input-file",    "-i", help="[required] File with processed radiomics data.")
    parser.add_argument("--output-dir",    "-o", help="Output directory for hyperparameter results.", default='studies_new')
    parser.add_argument("--output-model",    "-k", help="Output directory for saved models.", default='models_new')
    parser.add_argument("--max-evals",     "-m", help="Max different hyperparameters to test." , type=int, default=1000)
    parser.add_argument("--num-jobs",      "-j", help="Number of hyperparameter settings to try simulateously.", type=int, default=1)
    args = parser.parse_args()

    if not all([args.study_name, args.input_file]):
        print("Missing one or more of required arguments --study-name, --input-file", flush=True)
        exit()

    # Used by optuna to identify the experiment
    STUDY_NAME    = args.study_name
    DATABASE_PATH = "sqlite:///studies_new/{study_name}.db".format(study_name=STUDY_NAME)

    # Output filename; study name + random integer to make sure all parallel processes write to a different file
    OUT_DIR       = args.output_dir
    OUT_FILE      = "{}/results-{}-{}.csv".format(OUT_DIR, STUDY_NAME, np.random.randint(0,10000))
    MAX_EVALS     = args.max_evals
    JOBS_PER_NODE = args.num_jobs
    
    # Create output file and write headers
    print("> Output file:", OUT_FILE, flush=True)
    with open(OUT_FILE, 'w') as of_connection:
        writer = csv.writer(of_connection)
        
        # Write column names
        headers = ['score', 'test_auc', 'Hyperparameters', 'Features',  'iteration']
        writer.writerow(headers)
   
    print(end="> Importing data and splitting\n", flush=True)
    all_data_df = pd.read_csv(args.input_file, index_col=0)
    
    # Take x and y
    x = all_data_df.loc[:, all_data_df.columns != 'y_label']
    y = all_data_df.loc[:,'y_label']
    
    # Split in train and test
    x_TRAIN, x_test, y_TRAIN, y_test = train_test_split(x, y, test_size=0.2, random_state=13)
    print("Done.", flush=True)

    storage = optuna.storages.RDBStorage(
        url=DATABASE_PATH
    )

    # Create a study DB or load the DB for distributed jobs
    study = optuna.create_study(
        study_name=STUDY_NAME, 
        storage=storage, 
        direction='maximize',
        load_if_exists=True
        )

    study_id = storage.get_study_id_from_name(STUDY_NAME)

    # Perform optimization
    study.optimize(objective, n_trials=MAX_EVALS, n_jobs=JOBS_PER_NODE)
