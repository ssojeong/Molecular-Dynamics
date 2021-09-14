#!/bin/bash

echo forward rho0.1, first run no save model ..... 
CUDA_VISIBLE_DEVICES='' python ML_trainer.py ../data/gen_by_ML/pw-auto-lambda0.1/rho0.1/MD_config.dict ../data/gen_by_ML/pw-auto-lambda0.1/rho0.1/ML_config.dict
mkdir ../data/gen_by_ML/pw-auto-lambda0.1/rho0.1/forward_1
echo cp ../data/gen_by_ML/pw-auto-lambda0.1/rho0.1/rho0.1_21.pth ../data/gen_by_ML/pw-auto-lambda0.1/rho0/forward_1/forward_1_21.pth
cp ../data/gen_by_ML/pw-auto-lambda0.1/rho0.1/rho0.1_21.pth ../data/gen_by_ML/pw-auto-lambda0.1/rho0.1/forward_1/forward_1_21.pth
echo cp ../data/gen_by_ML/pw-auto-lambda0.1/rho0.1/rho0.1_loss.txt ../data/gen_by_ML/pw-auto-lambda0.1/rho0.1/forward_1/forward_1_loss.txt
cp ../data/gen_by_ML/pw-auto-lambda0.1/rho0.1/rho0.1_loss.txt ../data/gen_by_ML/pw-auto-lambda0.1/rho0.1/forward_1/forward_1_loss.txt

for i in  {1..8}
do
    for j in rho0.2 rho0.3 rho0.4
    do
        echo forward $j
        #cd ../data/gen_by_ML/pw-auto-lambda0.1/$j
	CUDA_VISIBLE_DEVICES='' python ML_trainer.py ../data/gen_by_ML/pw-auto-lambda0.1/$j/MD_config.dict ../data/gen_by_ML/pw-auto-lambda0.1/$j/ML_config.dict
        mkdir ../data/gen_by_ML/pw-auto-lambda0.1/${j}/forward_${i}
	echo cp ../data/gen_by_ML/pw-auto-lambda0.1/$j/forward_${i}/${j}_21.pth ../data/gen_by_ML/pw-auto-lambda0.1/$j/forward_${i}/forward_${i}_21.pth
	cp ../data/gen_by_ML/pw-auto-lambda0.1/$j/${j}_21.pth ../data/gen_by_ML/pw-auto-lambda0.1/$j/forward_${i}/forward_${i}_21.pth
	echo cp ../data/gen_by_ML/pw-auto-lambda0.1/$j/forward_${i}/${j}_loss.txt ../data/gen_by_ML/pw-auto-lambda0.1/$j/forward_${i}/forward_${i}_loss.txt
	cp ../data/gen_by_ML/pw-auto-lambda0.1/$j/${j}_loss.txt ../data/gen_by_ML/pw-auto-lambda0.1/$j/forward_${i}/forward_${i}_loss.txt
    done
 
    for j in rho0.3 rho0.2 rho0.1
    do
        echo backward $j
	CUDA_VISIBLE_DEVICES='' python ML_trainer.py ../data/gen_by_ML/pw-auto-lambda0.1/$j/MD_back_config.dict ../data/gen_by_ML/pw-auto-lambda0.1/$j/ML_back_config.dict
        mkdir ../data/gen_by_ML/pw-auto-lambda0.1/${j}/backward_${i}
	echo cp ../data/gen_by_ML/pw-auto-lambda0.1/$j/backward_${i}/${j}_21.pth ../data/gen_by_ML/pw-auto-lambda0.1/$j/backward_${i}/backward_${i}_21.pth
	cp ../data/gen_by_ML/pw-auto-lambda0.1/$j/${j}_21.pth ../data/gen_by_ML/pw-auto-lambda0.1/$j/backward_${i}/backward_${i}_21.pth
	echo cp ../data/gen_by_ML/pw-auto-lambda0.1/$j/${j}_loss.txt ../data/gen_by_ML/pw-auto-lambda0.1/$j/backward_${i}/backward_${i}_loss.txt
	cp ../data/gen_by_ML/pw-auto-lambda0.1/$j/${j}_loss.txt ../data/gen_by_ML/pw-auto-lambda0.1/$j/backward_${i}/backward_${i}_loss.txt
    done

done
