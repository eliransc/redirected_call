#!/bin/bash
#SBATCH -t 0-15:58
#SBATCH -A def-dkrass
source /home/eliransc/projects/def-dkrass/eliransc/queues/bin/activate #/home/d/dkrass/eliransc/queues/bin/activate
python /home/eliransc/projects/def-dkrass/eliransc/inter_departure/redirected_call/code/sim_two_stations_two_classes.py #/scratch/d/dkrass/eliransc/inter_departure/redirected_call/code/sim_two_stations_two_classes.py

