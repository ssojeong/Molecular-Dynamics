#!/bin/bash

echo forward 1rho0.10, first run no save model .....
CUDA_VISIBLE_DEVICES='' python loss_distribution.py ../data/gen_by_ML/pw-test-dist/1rho0.10/MD_config.dict ../data/gen_by_ML/pw-test-dist/1rho0.10/ML_config.dict

for i in  {1..1}
do
    for j in 2rho0.14 4rho0.20 7rho0.27 8rho0.38
    do
        echo forward $j
	CUDA_VISIBLE_DEVICES='' python loss_distribution.py ../data/gen_by_ML/pw-test-dist/$j/MD_config.dict ../data/gen_by_ML/pw-test-dist/$j/ML_config.dict
    done

done
