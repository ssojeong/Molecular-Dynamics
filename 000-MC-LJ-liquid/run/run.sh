for i in 2 3 4 5 6 7 8 
do
	echo "N: $i"
	#for j in $(seq 0.01 0.01 0.1) 0.2 0.4 0.6  $(seq 0.7 0.01 0.8)
	for j in 0.36 0.3
	#for j in $(seq 0.01 0.02 0.1)
	do
		echo "density: $j"
		python 02_LJ-mc.py --N $i --rho $j

	done

done

