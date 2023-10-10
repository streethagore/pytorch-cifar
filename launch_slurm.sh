#!/bin/bash

for delay in 1 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34; do
  sbatch launch_main_script.sh $delay
done
