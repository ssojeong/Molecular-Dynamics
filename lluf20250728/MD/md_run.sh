#!/usr/bin/bash

# e.g. ./md_run.sh ../../data_sets/gen_by_MD/3d/n32rho0.85lt0.1stps 32 0.9 97 0
# $1: ../../data_sets/gen_by_MD/3d/n32rho0.85lt0.1stps
# $2: nparticle
# $3: temp
# $4: seed
# $5 : gamma

folder="$1" 

for i in "$folder"/n$2T$3seed$4*nsamples*; do

    echo "$i"
    CUDA_VISIBLE_DEVICES='' python velocity_verlet_MD.py $i/MD_config.dict  $5 $3

done
