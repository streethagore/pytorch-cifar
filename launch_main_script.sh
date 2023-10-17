#!/bin/bash

#SBATCH --job-name=delay
#SBATCH --nodes=1
#SBATCH --partition=jazzy
#SBATCH --gpus-per-node=1
#SBATCH --time=03:30:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

# Load the conda module
source /usr/local/miniconda/etc/profile.d/conda.sh
conda activate async

delay=$1
logfile="output/resnet18-delay_${delay}"

custom_decay=$2
if [ $custom_decay == 'true' ]; then
  decay_cmd='--custom-decay'
  logfile="${logfile}-custom_decay"
else
  decay_cmd=''
  logfile="${logfile}-standard_decay"
fi

logfile="${logfile}.log"
python -u main.py --lr 0.05 --delay $delay $decay_cmd | tee "${logfile}"
