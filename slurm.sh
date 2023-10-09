#!/bin/bash

#SBATCH --job-name=delay
#SBATCH --nodes=1
#SBATCH --partition=jazzy
#SBATCH --gpus-per-node=1
#SBATCH --time=03:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Load the conda module
source /usr/local/miniconda/etc/profile.d/conda.sh
conda activate async

delay=$1

python -u main.py --delay $delay | tee "delay_${delay}.log"
