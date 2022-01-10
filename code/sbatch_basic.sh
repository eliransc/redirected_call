#!/bin/bash
#SBATCH -t 0-10:58
#SBATCH -A def-dkrass
source /home/d/dkrass/eliransc/queues/bin/activate
python /scratch/d/dkrass/eliransc/inter_departure/redirected_call/code/sim_two_stations_two_classes.py

