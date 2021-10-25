#!/bin/bash

# e.g. ./mc_run.sh ../data/gen_by_MC/n16rho0.1 16 0.27
# $1: ../data/gen_by_MC/titan02
# $2: nparticle
# $3: temp

folder="$1"

for i in "$folder"/n$2T$3seed*; do

    echo "$i"
    CUDA_VISIBLE_DEVICES='' python  MC_sampler.py $i/MC_config.dict

done

