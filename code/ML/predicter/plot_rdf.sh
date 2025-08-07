#!/bin/bash

python plot_rdf.py 16 0.035 2 0.46 1 809 600000 g
python plot_rdf.py 32 0.035 2 0.46 1 809 600000 g
python plot_rdf.py 64 0.035 2 0.46 1 809 600000 g
python plot_rdf.py 128 0.035 2 0.46 1 809 600000 g

# liquid+gas
python plot_rdf.py 16 0.3 2 0.46 10 485 800000 lg
python plot_rdf.py 32 0.3 2 0.46 10 485 800000 lg
python plot_rdf.py 64 0.3 2 0.46 10 485 800000 lg
python plot_rdf.py 128 0.3 2 0.46 10 485 800000 lg

#liquid
python plot_rdf.py 16 0.71 2 0.46 20 747 600000 l
python plot_rdf.py 32 0.71 2 0.46 20 747 600000 l
python plot_rdf.py 64 0.71 2 0.46 20 747 600000 l
python plot_rdf.py 128 0.71 2 0.46 20 747 600000 l

#
