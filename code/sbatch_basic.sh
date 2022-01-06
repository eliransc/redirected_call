#!/bin/bash
#SBATCH -t 0-23:58
#SBATCH -A def-dkrass
source /home/eliransc/.virtualenvs/deep_queue/bin/activate
python /scratch/d/dkrass/eliransc/inter_departure/redirected_call/code/sim_two_stations_two_classes.py

