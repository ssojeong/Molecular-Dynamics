#!/bin/bash
#PBS -N n128tau0.02
#PBS -l select=1:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o out-n128d0.02.txt
#PBS -P 13003073
#PBS -q normal
###PBS -q ai 

cd $PBS_O_WORKDIR

#module load miniforge3/24.3.0
#conda activate pytorch

#python maintrain09.py > results/traj_len08ws08tau0.05ngrid12api0lw8421ew1repw10poly1l_dpt180000/log_rerun
python maintest_combined.py > log/n128rho0.85T0.9LUF101_tau0.02


