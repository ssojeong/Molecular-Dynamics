#!/bin/bash

# e.g. ./mc_run.sh ../data/gen_by_MC/n16rho0.1 16
# $1: ../data/gen_by_MC/titan02
# $2: nparticle
# $3: temp
# $4: num.

folder="$1"

for i in "$folder"/n$2T$3seed$4*nsamples25; do
#for i in "$folder"/n$2T$3seed*nsamples1; do

    echo "$i"
    #CUDA_VISIBLE_DEVICES='' python  MC_sampler.py $i/MC_config.dict test
    CUDA_VISIBLE_DEVICES='' python  MC_sampler.py $i/MC_config.dict train

done

