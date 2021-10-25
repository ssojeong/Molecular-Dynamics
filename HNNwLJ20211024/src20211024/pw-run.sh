#!/bin/bash

echo forward 1rho0.10, first run no save model ..... 
CUDA_VISIBLE_DEVICES='' python ML_trainer.py ../data/gen_by_ML/pw-auto-test/1rho0.10/MD_config.dict ../data/gen_by_ML/pw-auto-test/1rho0.10/ML_config.dict
mkdir ../data/gen_by_ML/pw-auto-test/1rho0.10/forward_1
echo cp ../data/gen_by_ML/pw-auto-test/1rho0.10/1rho0.10_5.pth ../data/gen_by_ML/pw-auto-test/rho0/forward_1/forward_1_5.pth
cp ../data/gen_by_ML/pw-auto-test/1rho0.10/1rho0.10_5.pth ../data/gen_by_ML/pw-auto-test/1rho0.10/forward_1/forward_1_5.pth
echo cp ../data/gen_by_ML/pw-auto-test/1rho0.10/1rho0.10_loss.txt ../data/gen_by_ML/pw-auto-test/1rho0.10/forward_1/forward_1_loss.txt
cp ../data/gen_by_ML/pw-auto-test/1rho0.10/1rho0.10_loss.txt ../data/gen_by_ML/pw-auto-test/1rho0.10/forward_1/forward_1_loss.txt

