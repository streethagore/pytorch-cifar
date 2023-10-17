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

delay=$1
logfile="output/resnet18-delay_${delay}"

decay_mode=$2
logfile="${logfile}-${decay_mode}"

logfile="${logfile}.log"
python -u main.py --lr 0.05 --delay $delay --decay-mode $decay_mode | tee "${logfile}"
