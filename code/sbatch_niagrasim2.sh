#!/bin/bash
#SBATCH -t 0-23:58
#SBATCH -A def-dkrass
source /home/d/dkrass/eliransc/queues/bin/activate
python /scratch/d/dkrass/eliransc/new_redircted/redirected_call/code/sim_rho_0.py
