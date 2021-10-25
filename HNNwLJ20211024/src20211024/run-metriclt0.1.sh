#!/bin/bash

rho=$1

for i in 0.27 0.47 0.71
do
  echo $i
  CUDA_VISIBLE_DEVICES='' python MD_sampler.py ../data/gen_by_MD/noML/noML-metric-lt0.1/noML-n16rho$1T$i-hard1/MD_config.dict \
   ../data/gen_by_MD/noML/noML-metric-lt0.1/noML-n16rho$1T$i-hard1/ML_config.dict > ./log/noML-n16rho$1T$i-hard1metriclt0.1.log

done
 
