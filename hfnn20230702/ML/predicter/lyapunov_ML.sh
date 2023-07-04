#!/bin/bash
## 32 0.035 0.46  32 0.4 0.46   32 0.68 0.48
## 64 0.053 0.47  64 0.3 0.45   64 0.72 0.48
## 128 0.025 0.47 128 0.25 0.44 128 0.66 0.47
npar=$1
rho=$2
temp=$3
tmax=$4

echo ${npar} ${rho} ${temp}  window sliding step 1 g
python lyapunov_ML.py ${npar} ${rho} ${temp} 1 ${tmax} 163 pred_len08C1d256l2mbpw163t24.7_tau0.1_lyapunovML &
echo ${npar} ${rho} ${temp} window sliding step 4 g
python lyapunov_ML.py ${npar} ${rho} ${temp} 4 ${tmax} 131 pred_len08C4d256l2mbpw131t24.7_tau0.1_lyapunovML &
echo ${npar} ${rho} ${temp} window sliding step 8 g
python lyapunov_ML.py ${npar} ${rho} ${temp} 8 ${tmax} 097 pred_len08C8d256l2mbpw097t24.7_tau0.1_lyapunovML &
echo $npar $rho $temp window sliding step 12 g
python lyapunov_ML.py $npar $rho $temp 12 ${tmax} 013 pred_len08C12d256l2mbpw013t24.7_tau0.1_lyapunovML &
echo $npar $rho $temp window sliding step 16 g
python lyapunov_ML.py $npar $rho $temp 16 ${tmax} 009 pred_len08C16d256l2mbpw009t24.7_tau0.1_lypunovML

#echo 32 0.035 0.46 g
#python lyapunov_wolf.py 32 0.035 0.46 1e-3 10000
#echo 32 0.4 0.46 lg
#python lyapunov_wolf.py 32 0.4 0.46 1e-3 10000
#echo 32 0.68 0.48 l
#python lyapunov_wolf.py 32 0.68 0.48 1e-3 10000
#echo 64 0.053 0.47 g
#python lyapunov_wolf.py 64 0.053 0.47 1e-3 10000
#echo 64 0.3 0.45 lg
#python lyapunov_wolf.py 64 0.3 0.45 1e-3 10000
#echo 64 0.72 0.48 l
#python lyapunov_wolf.py 64 0.72 0.48 1e-3 10000
#echo 128 0.025 0.47 g
#python lyapunov_wolf.py 128 0.025 0.47 1e-3 10000
#echo 128 0.25 0.44 lg
#python lyapunov_wolf.py 128 0.25 0.44 1e-3 10000
#echo 128 0.66 0.47 l
#python lyapunov_wolf.py 128 0.66 0.47 1e-3 10000
#echo 32 0.035 0.46 g
#python lyapunov_wolf.py 32 0.035 0.46 1e-2 1000
#echo 32 0.4 0.46 lg
#python lyapunov_wolf.py 32 0.4 0.46 1e-2 1000
#echo 32 0.68 0.48 l
#python lyapunov_wolf.py 32 0.68 0.48 1e-2 1000
#echo 64 0.053 0.47 g
#python lyapunov_wolf.py 64 0.053 0.47 1e-2 1000
#echo 64 0.3 0.45 lg
#python lyapunov_wolf.py 64 0.3 0.45 1e-2 1000
#echo 64 0.72 0.48 l  
#python lyapunov_wolf.py 64 0.72 0.48 1e-2 1000
#echo 128 0.025 0.47 g
#python lyapunov_wolf.py 128 0.025 0.47 1e-2 1000
#echo 128 0.25 0.44 lg
#python lyapunov_wolf.py 128 0.25 0.44 1e-2 1000
#echo 128 0.66 0.47 l
#python lyapunov_wolf.py 128 0.66 0.47 1e-2 1000
