#for j in 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95
#for j in 0.35 0.4 0.45
#for j in 0.25 0.3
#for j in 0.15 0.2
#for j in 0.05 0.1
for j in 0.01
do
	echo "T: $j"
	python 02_LJ-mc_pos.py --N 2 --rho  0.2 --temp $j 
done

