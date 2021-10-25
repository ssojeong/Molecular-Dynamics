#!/bin/bash

rho=$1

for i in 0.27 0.47 0.71
do
  echo $i
  CUDA_VISIBLE_DEVICES='' python MD_sampler.py ../data/gen_by_MD/fhnn/f-n16lambda0.0hard1e20990/f-n16rho$1T$i-hard1/MD_config.dict \
   ../data/gen_by_MD/fhnn/f-n16lambda0.0hard1e20990/f-n16rho$1T$i-hard1/ML_config.dict > ./log/f-n16rho$1T$i-hard1e20990.log

done
 
