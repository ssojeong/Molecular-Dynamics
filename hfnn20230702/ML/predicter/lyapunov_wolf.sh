#!/bin/bash

echo 32 0.035 0.46 g
python lyapunov_wolf.py 32 0.035 0.46 1e-2 1000
echo 32 0.4 0.46 lg
python lyapunov_wolf.py 32 0.4 0.46 1e-2 1000
echo 32 0.68 0.48 l
python lyapunov_wolf.py 32 0.68 0.48 1e-2 1000
echo 64 0.053 0.47 g
python lyapunov_wolf.py 64 0.053 0.47 1e-2 1000
echo 64 0.3 0.45 lg
python lyapunov_wolf.py 64 0.3 0.45 1e-2 1000
echo 64 0.72 0.48 l
python lyapunov_wolf.py 64 0.72 0.48 1e-2 1000
echo 128 0.025 0.47 g
python lyapunov_wolf.py 128 0.025 0.47 1e-2 1000
echo 128 0.25 0.44 lg
python lyapunov_wolf.py 128 0.25 0.44 1e-2 1000
echo 128 0.66 0.47 l
python lyapunov_wolf.py 128 0.66 0.47 1e-2 1000

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
