#!/bin/bash

partition=$1
delay=0

decay_mode='pytorch'
decay_delayed='false'
for momentum in 'false' 'true'; do
  for weight_decay in 'false' 'true'; do
    echo "delay $delay - decay mode $decay_mode - delayed decay $decay_delayed - momentum $momentum - weight decay $weight_decay"
    sbatch -p $partition launch_main_script.sh $delay $decay_mode $decay_delayed $momentum $weight_decay
  done
done

for decay_mode in 'loss' 'weights'; do
  for decay_delayed in 'true' 'false'; do
    for momentum in 'false' 'true'; do
      for weight_decay in 'false' 'true'; do
        echo "delay $delay - decay mode $decay_mode - delayed decay $decay_delayed - momentum $momentum - weight decay $weight_decay"
        sbatch -p $partition launch_main_script.sh $delay $decay_mode $decay_delayed
      done
    done
  done
done
