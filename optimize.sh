#!/bin/bash

# optimize.sh
# Job used on SLURM cluster to start hyperoptimization trials.
# To complete many trials in a short amount of time, we start multiple simultaneous
# jobs using the "--array" SBATCH argument.
# (C) J. Bleker & C. Roest (2023)

#SBATCH --array=1-10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --partition=vulture
#SBATCH --cpus-per-task=12
#SBATCH --mem=64000
#SBATCH --job-name=Res.Rad.

EXPERIMENT_NAME=T2W_2D_Linear_0.5 
RADIOMICS_CSV=/path/to/data/dir/T2W_2D_Linear_0.5

# Remove all currently loaded modules from the LMOD environment
module purge

# Load the tested set of modules
module load Python/3.6.4-foss-2018a
module load Boost/1.67.0-foss-2018a
module load CMake

# Load the environment with required python modules
source /path/to/env/bin/activate

# Start running hyperparameter optimization
# NOTE: --max-evals limits the number of trials to 35. 
# This is done because it was noticed that running many more trials in a single run
# can result in jobs getting stuck. 
# Instead, we therefore keep the jobs short and run more jobs.
./optimize.py --study-name $EXPERIMENT_NAME --input-file $RADIOMICS_CSV --max-evals 35

