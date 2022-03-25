#!/bin/bash
#SBATCH -t 0-23:58
#SBATCH -A def-dkrass
source /home/eliransc/projects/def-dkrass/eliransc/queues/bin/activate
python /home/eliransc/projects/def-dkrass/eliransc/redirected_call/code/main_accuracy.py
