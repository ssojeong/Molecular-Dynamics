#!/bin/bash
#PBS -N LW
#PBS -l select=1:ngpus=1
#PBS -l walltime=120:00:00
#PBS -j oe
#PBS -o out-run.txt
#PBS -P 13003073
#PBS -q normal
####PBS -q ai 

cd $PBS_O_WORKDIR
./run_test.sh
#python lyapunov_ML.py 128 0.025 0.47 1 20.9 163 pred_len08C1d256l2mbpw163t24.7_tau0.1_lyapunovML > t
#python lyapunov_wolf.py 32 0.68 0.48 0.01 1000 > ../../../data_sets/gen_by_MD/noML-metric-lt0.01every0.1t0.7t100/n32rho0.68T0.48/log_tau0.01
#python lyapunov_wolf.py 32 0.68 0.48 0.001 10000 > ../../../data_sets/gen_by_MD/noML-metric-lt0.01every0.1t0.7t100/n32rho0.68T0.48/log_tau0.001
