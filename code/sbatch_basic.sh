#!/bin/bash
#SBATCH -t 0-23:58
#SBATCH -A def-dkrass
#SBATCH --mem 10000
source /home/eliransc/.virtualenvs/deep_queue/bin/activate
python /gpfs/fs1/home/d/dkrass/eliransc/redirected_call/code/sim_two_stations_two_classes.py

