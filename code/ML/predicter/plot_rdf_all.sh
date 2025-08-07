#!/bin/bash

## gas
python plot_rdf_all.py 16 0.035 2 1 600000 g 6 14 1.2 2.8
python plot_rdf_all.py 32 0.035 2 1 600000 g 6 14 1.2 2.8
python plot_rdf_all.py 64 0.035 2 1 600000 g 6 14 1.2 2.8
python plot_rdf_all.py 128 0.035 2 1 600000 g 6 14 1.2 2.8

# liquid+gas
python plot_rdf_all.py 16 0.3 2 10 800000 lg 3.9 7.6 1.3 3
python plot_rdf_all.py 32 0.3 2 10 800000 lg 3.9 7.6 1.3 3
python plot_rdf_all.py 64 0.3 2 10 800000 lg 3.9 7.6 1.3 3
python plot_rdf_all.py 128 0.3 2 10 800000 lg 3.9 7.6 1.3 3
#
#liquid
python plot_rdf_all.py 16 0.71 2 20 600000 l 3.1 3.9 1.46 1.68
python plot_rdf_all.py 32 0.71 2 20 600000 l 3.1 3.9 1.46 1.68
python plot_rdf_all.py 64 0.71 2 20 600000 l 3.1 3.9 1.46 1.68
python plot_rdf_all.py 128 0.71 2 20 600000 l 3.1 3.9 1.46 1.68

#
