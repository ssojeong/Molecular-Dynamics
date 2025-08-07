#!/bin/bash

# gas
python plot_e_cv_all.py 16 0.035 2 1 600000 g -0.02 0.1  -0.2 2
python plot_e_cv_all.py 32 0.035 2 1 600000 g -0.02 0.1  -0.2 2
python plot_e_cv_all.py 64 0.035 2 1 600000 g -0.02 0.1  -0.2 2
python plot_e_cv_all.py 128 0.035 2 1 600000 g -0.02 0.1  -0.2 2

# liquid+gas
python plot_e_cv_all.py 16 0.3 2 10 800000 lg  -0.04 0.32 -0.04 4
python plot_e_cv_all.py 32 0.3 2 10 800000 lg  -0.04 0.32  -0.04 4
python plot_e_cv_all.py 64 0.3 2 10 800000 lg  -0.04 0.32  -0.04 4
python plot_e_cv_all.py 128 0.3 2 10 800000 lg  -0.04 0.32  -0.04 4

# liquid
python plot_e_cv_all.py 16 0.71 2 20 600000 l -0.04 0.1  -0.2 3
python plot_e_cv_all.py 32 0.71 2 20 600000 l -0.04 0.1  -0.2 3
python plot_e_cv_all.py 64 0.71 2 20 600000 l -0.04 0.1  -0.2 3
python plot_e_cv_all.py 128 0.71 2 20 600000 l -0.04 0.1  -0.2 3

