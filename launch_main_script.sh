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

decay_delayed=$3
if [ $decay_delayed == 'true' ]; then
  decay_delayed_cmd='--decay-delayed'
  logfile="${logfile}_delayed"
elif [ $decay_delayed == 'false' ]; then
  decay_delayed_cmd=''
fi

momentum=$4
if [ $momentum == 'true' ]; then
  momentum=0.9
  logfile="${logfile}-with_momentum"
elif [ $momentum == 'false' ]; then
  momentum=0.0
  logfile="${logfile}-no_momentum"
fi

weight_decay=$5
if [ $weight_decay == 'true' ]; then
  weight_decay=5e-4
  logfile="${logfile}-with_decay"
elif [ $weight_decay == 'false' ]; then
  weight_decay=0.0
  logfile="${logfile}-no_decay"
fi

logfile="${logfile}.log"
echo "python -u main.py --lr 0.05 --delay $delay --momentum $momentum --weight-decay $weight_decay --decay-mode $decay_mode $decay_delayed_cmd" | tee "${logfile}"
python -u main.py --lr 0.05 --delay $delay --momentum $momentum --weight-decay $weight_decay --decay-mode $decay_mode $decay_delayed_cmd | tee -a "${logfile}"
