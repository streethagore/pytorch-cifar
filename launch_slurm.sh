#!/bin/bash

partition=$1

for delay in {0..10..1}; do
  for custom_decay in 'true' 'false'; do
    sbatch -p $partition launch_main_script.sh $delay $custom_decay
  done
done
