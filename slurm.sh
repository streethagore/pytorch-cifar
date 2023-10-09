#!/bin/bash

#SBATCH --job-name=delay
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=03:30:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

# Load the conda module
source /usr/local/miniconda/etc/profile.d/conda.sh
conda activate async

python -u main.py --delay $1 | tee "delay_${1}.log"
