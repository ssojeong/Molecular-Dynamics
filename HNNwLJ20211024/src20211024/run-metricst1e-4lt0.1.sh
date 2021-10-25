#!/bin/bash

rho=$1

for i in 0.27 0.47 0.71
do
  echo $i
  CUDA_VISIBLE_DEVICES='' python MD_sampler.py ../data/gen_by_MD/pwhnn/pw-n16hard1metricst1e-4lt0.1/pw-n16rho$1T$i-hard1/MD_config.dict \
   ../data/gen_by_MD/pwhnn/pw-n16hard1metricst1e-4lt0.1/pw-n16rho$1T$i-hard1/ML_config.dict > ./log/pw-n16rho$1T$i-hard1metricst1e-4lt0.1.log

done
 