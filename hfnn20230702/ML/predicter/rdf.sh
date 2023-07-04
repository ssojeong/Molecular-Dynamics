#!/bin/bash
## 32 0.035 0.46  32 0.4 0.46   32 0.68 0.48
## 64 0.053 0.47  64 0.3 0.45   64 0.72 0.48
## 128 0.025 0.47 128 0.25 0.44 128 0.66 0.47
npar=$1
rho=$2
temp=$3
name=$4

echo $npar $rho $temp  window sliding step 1 g
CUDA_VISIBLE_DEVICES='' python rdf.py $npar $rho $temp   1 0.8 $4 pred_len08C1d256l2mbpw163t24.7_tau0.1.pt &
echo $npar $rho $temp  window sliding step 4 g
CUDA_VISIBLE_DEVICES='' python rdf.py $npar $rho $temp   4 0.8 $4 pred_len08C4d256l2mbpw131t24.7_tau0.1.pt &
echo $npar $rho $temp window sliding step 8 g
CUDA_VISIBLE_DEVICES='' python rdf.py $npar $rho $temp   8 0.8 $4 pred_len08C8d256l2mbpw097t24.7_tau0.1.pt &
echo $npar $rho $temp window sliding step 12 g
CUDA_VISIBLE_DEVICES='' python rdf.py $npar $rho $temp   12 0.8 $4 pred_len08C12d256l2mbpw013t24.7_tau0.1.pt &
echo $npar $rho $temp window sliding step 16 g
CUDA_VISIBLE_DEVICES='' python rdf.py $npar $rho $temp   16 0.8 $4 pred_len08C16d256l2mbpw009t24.7_tau0.1.pt

echo $npar $rho $temp  window sliding step 1 g
CUDA_VISIBLE_DEVICES='' python rdf.py $npar $rho $temp   1 2.7 $4 pred_len08C1d256l2mbpw163t24.7_tau0.1.pt &
echo $npar $rho $temp  window sliding step 4 g
CUDA_VISIBLE_DEVICES='' python rdf.py $npar $rho $temp   4 2.7 $4 pred_len08C4d256l2mbpw131t24.7_tau0.1.pt &
echo $npar $rho $temp  window sliding step 8 g
CUDA_VISIBLE_DEVICES='' python rdf.py $npar $rho $temp   8 2.3 $4 pred_len08C8d256l2mbpw097t24.7_tau0.1.pt &
echo $npar $rho $temp  window sliding step 12 g
CUDA_VISIBLE_DEVICES='' python rdf.py $npar $rho $temp  12 3.1 $4 pred_len08C12d256l2mbpw013t24.7_tau0.1.pt &
echo $npar $rho $temp  window sliding step 16 g
CUDA_VISIBLE_DEVICES='' python rdf.py $npar $rho $temp  16 2.3 $4 pred_len08C16d256l2mbpw009t24.7_tau0.1.pt

echo $npar $rho $temp  window sliding step 1 g
CUDA_VISIBLE_DEVICES='' python rdf.py $npar $rho $temp   1 4.7 $4 pred_len08C1d256l2mbpw163t24.7_tau0.1.pt &
echo $npar $rho $temp  window sliding step 4 g
CUDA_VISIBLE_DEVICES='' python rdf.py $npar $rho $temp   4 4.7 $4 pred_len08C4d256l2mbpw131t24.7_tau0.1.pt &
echo $npar $rho $temp  window sliding step 8 g
CUDA_VISIBLE_DEVICES='' python rdf.py $npar $rho $temp   8 4.7 $4 pred_len08C8d256l2mbpw097t24.7_tau0.1.pt &
echo $npar $rho $temp  window sliding step 12 g
CUDA_VISIBLE_DEVICES='' python rdf.py $npar $rho $temp   12 4.3 $4 pred_len08C12d256l2mbpw013t24.7_tau0.1.pt &
echo $npar $rho $temp  window sliding step 16 g
CUDA_VISIBLE_DEVICES='' python rdf.py $npar $rho $temp   16 3.9 $4 pred_len08C16d256l2mbpw009t24.7_tau0.1.pt
