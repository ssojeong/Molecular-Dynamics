#!/bin/bash

python plot_rdf_relative_all.py 16 0.035 2 1  g -0.3 0.15 -0.25 0.05
python plot_rdf_relative_all.py 32 0.035 2 1  g -0.3 0.15 -0.25 0.05
python plot_rdf_relative_all.py 64 0.035 2 1  g -0.3 0.15 -0.25 0.05
python plot_rdf_relative_all.py 128 0.035 2 1  g -0.3 0.15 -0.25 0.05

 liquid+gas
python plot_rdf_relative_all.py 16 0.3 2 10  lg -0.25 0.3 -0.1 0.55
python plot_rdf_relative_all.py 32 0.3 2 10  lg -0.25 0.3 -0.1 0.55
python plot_rdf_relative_all.py 64 0.3 2 10  lg -0.25 0.3 -0.1 0.55
python plot_rdf_relative_all.py 128 0.3 2 10  lg -0.25 0.3 -0.1 0.55
#
#liquid
python plot_rdf_relative_all.py 16 0.71 2 20  l -0.15 0.09 -0.07 0.07
python plot_rdf_relative_all.py 32 0.71 2 20  l -0.15 0.09 -0.07 0.07
python plot_rdf_relative_all.py 64 0.71 2 20  l -0.15 0.09 -0.07 0.07
python plot_rdf_relative_all.py 128 0.71 2 20  l -0.15 0.09 -0.07 0.07

#
