#Temp=(0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95)
Temp=(0.01 0.35 0.85)
#dq=(0.1 1 1) #N_particle 2
#dq=(0.05 0.4 0.8)  #N_particle 4 
dq=(0.04 0.4 1)  #N_particle 6 0.04 0.4 1
for j in {0..2}
do
	python MCMC_sample.py --particle 6 --temp ${Temp[$j]} --dq ${dq[$j]}

done



