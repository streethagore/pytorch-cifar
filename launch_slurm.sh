#!/bin/bash

#for delay in 0 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35; do
for delay in {2..32..2}; do
  sbatch launch_main_script.sh $delay
done
