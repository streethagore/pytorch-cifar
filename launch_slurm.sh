#!/bin/bash

partition=$1

for delay in {0..1..1}; do
  for decay_mode in 'pytorch' 'loss' 'weights'; do
    for decay_delayed in 'true' 'false'; do
      sbatch -p $partition launch_main_script.sh $delay $decay_mode $decay_delayed
    done
  done
done
