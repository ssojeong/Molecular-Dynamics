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
#python velocity_verlet_MD.py ../../data_sets/gen_by_MD/solid_tau1e-7/n16T0.59seed1772nsamples5/MD_config.dict > ../../data_sets/gen_by_MD/solid_tau1e-7/n16T0.59seed1772nsamples5/log
python velocity_verlet_MD.py ../../data_sets/gen_by_MD/solid_tau1e-7/n16T0.75seed1772nsamples5/MD_config.dict > ../../data_sets/gen_by_MD/solid_tau1e-7/n16T0.75seed1772nsamples5/log
