#Temp=(0.01 0.35 0.85 0.01 0.35 0.85)
Temp=(0.01 0.35 0.85 0.01 0.35 0.85)
interations=(10000 10000 10000 1000 1000 1000) 
ts=(0.001 0.001 0.001 0.01 0.01 0.01)  
for j in {0..5}
do
	echo "$j"
	python Langevin_sample.py --particle 6 --temp ${Temp[$j]} --samples 1000 --iterations ${interations[$j]} --ts ${ts[$j]}

done



