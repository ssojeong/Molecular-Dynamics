#!/bin/bash
## 32 0.035 0.46  32 0.4 0.46   32 0.68 0.48
## 64 0.053 0.47  64 0.3 0.45   64 0.72 0.48
## 128 0.025 0.47 128 0.25 0.44 128 0.66 0.47
npar=$1
rho=$2
temp=$3
tmax=$4

echo $npar $rho $temp ${tmax} 
#python exp_instability_wolf.py $npar $rho $temp  1 ${tmax} md > analysis/lyapunov/log_n${npar}rho${rho}T${temp}tau0.001_md

python exp_instability_wolf.py $npar $rho $temp 1 ${tmax} pred_len08C1d256l2mbpw163t24.7_tau0.1.pt > analysis/lyapunov/log_n${npar}rho${rho}T${temp}C1tau0.1t${tmax}_ml
python exp_instability_wolf.py $npar $rho $temp 4 ${tmax} pred_len08C4d256l2mbpw131t24.7_tau0.1.pt > analysis/lyapunov/log_n${npar}rho${rho}T${temp}C4tau0.1t${tmax}_ml
python exp_instability_wolf.py $npar $rho $temp 8 ${tmax} pred_len08C8d256l2mbpw097t24.7_tau0.1.pt > analysis/lyapunov/log_n${npar}rho${rho}T${temp}C8tau0.1t${tmax}_ml
python exp_instability_wolf.py $npar $rho $temp 12 ${tmax} pred_len08C12d256l2mbpw013t24.7_tau0.1.pt > analysis/lyapunov/log_n${npar}rho${rho}T${temp}C12tau0.1t${tmax}_ml
python exp_instability_wolf.py $npar $rho $temp 16 ${tmax} pred_len08C16d256l2mbpw009t24.7_tau0.1.pt > analysis/lyapunov/log_n${npar}rho${rho}T${temp}C16tau0.1t${tmax}_ml

#echo 32 0.035 0.46 g
#python exp_instability_wolf.py 32 0.035 0.46 0.001 md> analysis/lyapunov/log_n32rho0.035T0.46tau0.001_md
#echo 32 0.4 0.46 lg
#python exp_instability_wolf.py 32 0.4 0.46 0.001 md> analysis/lyapunov/log_n32rho0.4T0.46tau0.001_md
#echo 32 0.68 0.48 l
#python exp_instability_wolf.py 32 0.68 0.48 0.001 md> analysis/lyapunov/log_n32rho0.68T0.48tau0.001_md
#echo 64 0.053 0.47 g
#python exp_instability_wolf.py 64 0.053 0.47 0.001 md> analysis/lyapunov/log_n64rho0.053T0.47tau0.001_md
#echo 64 0.3 0.45 lg
#python exp_instability_wolf.py 64 0.3 0.45 0.001 md> analysis/lyapunov/log_n64rho0.3T0.45tau0.001_md
#echo 64 0.72 0.48 l
#python exp_instability_wolf.py 64 0.72 0.48 0.001 md> analysis/lyapunov/log_n64rho0.72T0.48tau0.001_md
#echo 128 0.025 0.47 g
#python exp_instability_wolf.py 128 0.025 0.47 0.001 md> analysis/lyapunov/log_n128rho0.025T0.47tau0.001_md
#echo 128 0.25 0.44 lg
#python exp_instability_wolf.py 128 0.25 0.44 0.001 md> analysis/lyapunov/log_n128rho0.25T0.44tau0.001_md
#echo 128 0.66 0.47 l
#python exp_instability_wolf.py 128 0.66 0.47 0.001 md> analysis/lyapunov/log_n128rho0.66T0.47tau0.001_md
#echo 32 0.035 0.46 g
#python exp_instability_wolf.py 32 0.035 0.46 0.01 md> analysis/lyapunov/log_n32rho0.035T0.46tau0.01_md
#echo 32 0.4 0.46 lg
#python exp_instability_wolf.py 32 0.4 0.46 0.01 md> analysis/lyapunov/log_n32rho0.4T0.46tau0.01_md
#echo 32 0.68 0.48 l
#python exp_instability_wolf.py 32 0.68 0.48 0.01 md> analysis/lyapunov/log_n32rho0.68T0.48tau0.01_md
#echo 64 0.053 0.47 g
#python exp_instability_wolf.py 64 0.053 0.47 0.01 md> analysis/lyapunov/log_n64rho0.053T0.47tau0.01_md
#echo 64 0.3 0.45 lg
#python exp_instability_wolf.py 64 0.3 0.45 0.01 md> analysis/lyapunov/log_n64rho0.3T0.45tau0.01_md
#echo 64 0.72 0.48 l
#python exp_instability_wolf.py 64 0.72 0.48 0.01 md> analysis/lyapunov/log_n64rho0.72T0.48tau0.01_md
#echo 128 0.025 0.47 g
#python exp_instability_wolf.py 128 0.025 0.47 0.01 md> analysis/lyapunov/log_n128rho0.025T0.47tau0.01_md
#echo 128 0.25 0.44 lg
#python exp_instability_wolf.py 128 0.25 0.44 0.01 md> analysis/lyapunov/log_n128rho0.25T0.44tau0.01_md
#echo 128 0.66 0.47 l
#python exp_instability_wolf.py 128 0.66 0.47 0.01 md> analysis/lyapunov/log_n128rho0.66T0.47tau0.01_md
