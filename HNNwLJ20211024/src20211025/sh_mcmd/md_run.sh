#!/bin/bash

# e.g. ./md_run.sh ../data/gen_by_MD/n16rho0.1 16 0.27
# $1: ../data/gen_by_MC/titan02
# $2: nparticle
# $3: temp

folder="$1" 

for i in "$folder"/n$2T$3seed*; do

    echo "$i"
    CUDA_VISIBLE_DEVICES='' python  MD_sampler.py $i/MD_config.dict  $i/ML_config.dict

done
