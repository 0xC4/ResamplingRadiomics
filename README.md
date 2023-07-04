# MRI resampling in multi-center prostate MRI radiomics
This repository contains the experimental code and analysis scripts used for the study "The effect of MR image resampling on the performance of radiomics AI in multicenter prostate MRI".
If you use the code in this repository in your study, please cite to corresponding study:
```
TBD
```

## Radiomics modelling
### Requirements
Radiomics modelling was performed in Python version 3.6.4.
The following requirements apply for the training of radiomics models and hyperparameter tuning:
```
xgboost==1.7.6
scikit-learn==1.2.2
optuna==3.1.0
numpy==1.21.3
pandas==1.3.4
```

For our experiments we used a python virtual environment (venv) to manage installed packages.
This can be set up with the following commands: 
```
# Create the virtual environment
python -m venv env

# Load the environment (choose which applies)
# Linux
source env/bin/activate

# Windows
env/Scripts/activate.ps1

# Install packages to the enviroment
pip install xgboost==1.7.6
pip install scikit-learn==1.2.2
pip install optuna==3.1.0
pip install numpy==1.21.3
pip install pandas==1.3.4
```

After the environment has been set up once, you can load it in the future by running:
```
# Linux
source env/bin/activate

# Windows
env/Scripts/activate.ps1
```

### 1. Feature extraction
TODO
```
TODO
```

### 2. Hyperparameter optimization
We used Optuna to optimize the hyperparameters for each radiomics model.
Optuna searches the search space to optimize an objective function, both of which are defined in `optimize.py`.
To start a hyperparameter search for a set of radiomics features extracted in Step 1, run the following command:
```
python optimize.py --study-name test_experiment --input-file /path/to/radiomics/features.csv 
```
This will create a database containing Optuna results in `studies_new/test_experiment.db`, and a CSV file containing validation AUCs in `studies_new/results-test_experiment-XXXX.csv`

Optuna also supports distributed optimization, for which a central database is used to store objective values for each candidate set of hyperparameters.
An example SLURM job-script that was used to run a distributed hyperparameter search across multiple CPU jobs is shown in `optimize.sh`.
To run this script on your HPC cluster, modules / resources / partitions needs to be adjusted to your cluster's configuration. 

### 3. Inference
We now use the optimal hyperparameters with the best validation performance to predict the held-out test-cases.
This step is shown in `inference.py`.
First, open the fil and update the `best_params` dictionary, and the `features` list to the optimal hyperparameters and features.
```
# Best hyperparameters
best_params = {
    "booster": "gbtree",
    "subsample": 0.95,
    "reg_alpha": 2.7998614888500217,
    "reg_lambda": 10.17242426973007,
    # ...
}

# Best features
features = [
    "log-sigma-3-0-mm-3D_glszm_SmallAreaHighGrayLevelEmphasis-DWI-b1400",
    "logarithm_glcm_SumEntropy-ADC",
    "log-sigma-3-0-mm-3D_glszm_ZonePercentage-DWI-b1400",
    # ...
]
```

Then, run the following command to write the predictions to a CSV file and obtain the test AUC score:
```
python inference.py /path/to/radiomics/features.csv output/predictions.csv
```
