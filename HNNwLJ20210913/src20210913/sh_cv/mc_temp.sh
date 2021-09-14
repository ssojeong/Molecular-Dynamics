#!/bin/bash
#  use to measure cv about one sample
# e.g. ./mc_run.sh 16 1
# $1: nparticle $2: nsamples

temp=(0.19 0.27 0.35 0.43 0.51 0.59 0.67 0.75 0.83)
for i in 123456; do

        for j in $(seq 0 8); do

                echo "n$1 T${temp[$j]} seed $i nsamples $2"

                #usage <programe> <nparticle> <temperature> <seed> <nsamples> <dq>
                CUDA_VISIBLE_DEVICES='' python  MC_sampler.py ../data/gen_by_MC/n16rho0.1/n$1T${temp[$j]}seed${i}nsamples$2/MC_config.dict
        done
done

