#!/bin/bash

echo forward 1rho0.10, first run no save model .....
CUDA_VISIBLE_DEVICES='' python ML_trainer.py ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/1rho0.10/MD_config.dict ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/1rho0.10/ML_config.dict
mkdir ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/1rho0.10/forward_1
echo cp ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/1rho0.10/1rho0.10_5.pth ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/rho0/forward_1/forward_1_5.pth
cp ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/1rho0.10/1rho0.10_5.pth ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/1rho0.10/forward_1/forward_1_5.pth
echo cp ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/1rho0.10/1rho0.10_loss.txt ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/1rho0.10/forward_1/forward_1_loss.txt
cp ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/1rho0.10/1rho0.10_loss.txt ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/1rho0.10/forward_1/forward_1_loss.txt

for i in  {1..900}
do
    for j in 2rho0.14 3rho0.10 4rho0.20 5rho0.10 6rho0.14 7rho0.27 8rho0.38
    do
        echo forward $j
        #cd ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/$j
	CUDA_VISIBLE_DEVICES='' python ML_trainer.py ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/$j/MD_config.dict ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/$j/ML_config.dict
        mkdir ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/${j}/forward_${i}
	echo cp ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/$j/${j}_5.pth ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/$j/forward_${i}/forward_${i}_5.pth
	cp ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/$j/${j}_5.pth ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/$j/forward_${i}/forward_${i}_5.pth
	echo cp ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/$j/${j}_loss.txt ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/$j/forward_${i}/forward_${i}_loss.txt
	cp ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/$j/${j}_loss.txt ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/$j/forward_${i}/forward_${i}_loss.txt
    done

    for j in 7rho0.27 6rho0.14 5rho0.10 4rho0.20 3rho0.10 2rho0.14 1rho0.10
    do
        echo backward $j
	CUDA_VISIBLE_DEVICES='' python ML_trainer.py ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/$j/MD_back_config.dict ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/$j/ML_back_config.dict
        mkdir ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/${j}/backward_${i}
	echo cp ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/$j/${j}_5.pth ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/$j/backward_${i}/backward_${i}_5.pth
	cp ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/$j/${j}_5.pth ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/$j/backward_${i}/backward_${i}_5.pth
	echo cp ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/$j/${j}_loss.txt ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/$j/backward_${i}/backward_${i}_loss.txt
	cp ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/$j/${j}_loss.txt ../data/gen_by_ML/pw-auto-lambda0.1lexpreg/$j/backward_${i}/backward_${i}_loss.txt
    done

done
