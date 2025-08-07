#!/bin/bash

python save_rdf_all.py 16 0.035 2 1 809 600000 g 200
python save_rdf_all.py 32 0.035 2 1 809 600000 g 200
python save_rdf_all.py 64 0.035 2 1 809 600000 g 200
python save_rdf_all.py 128 0.035 2 1 809 600000 g 200

# liquid+gas
python save_rdf_all.py 16 0.3 2 10 485 800000 lg 200
python save_rdf_all.py 32 0.3 2 10 485 800000 lg 200
python save_rdf_all.py 64 0.3 2 10 485 800000 lg 200
python save_rdf_all.py 128 0.3 2 10 485 800000 lg 200

##liquid
python save_rdf_all.py 16 0.71 2 20 747 600000 l 200
python save_rdf_all.py 32 0.71 2 20 747 600000 l 200
python save_rdf_all.py 64 0.71 2 20 747 600000 l 200
python save_rdf_all.py 128 0.71 2 20 747 600000 l 200

