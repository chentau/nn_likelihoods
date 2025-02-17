#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J fcn_train_test
##SBATCH -J full_ddm_train_test
##SBATCH -J weibull_train_test
##SBATCH -J ornstein_train_test
##SBATCH -J nf_train_test

# output file
#SBATCH --output /users/afengler/batch_job_out/fcn_train_test_%A_%a.out
##SBATCH --output /users/afengler/batch_job_out/ddm_train_test_%A_%a.out
##SBATCH --output /users/afengler/batch_job_out/full_ddm_train_test_%A_%a.out
##SBATCH --output /users/afengler/batch_job_out/weibull_train_test_%A_%a.out
##SBATCH --output /users/afengler/batch_job_out/ornstein_train_test_%A_%a.out
##SBATCH --output /users/afengler/batch_job_out/nf_train_test_%A_%a.out

#priority
#SBATCH --account=bibs-frankmj-condo

# Request runtime, memory, cores:
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -N 1
##SBATCH --array=1-500

# Run a command

python -u /users/afengler/git_repos/nn_likelihoods/kde_train_test.py
#python -u /users/afengler/git_repos/nn_likelihoods/navarro_fuss_train_test.py
