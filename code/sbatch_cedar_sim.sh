#!/bin/bash
#SBATCH -t 0-02:58
#SBATCH -A def-dkrass
source /home/eliransc/projects/def-dkrass/eliransc/queues/bin/activate
python /home/eliransc/projects/def-dkrass/eliransc/inter_departure/redirected_call/code/sim_two_stations_two_classes.py

