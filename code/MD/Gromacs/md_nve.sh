#PBS -q normal
#PBS -l select=1:ncpus=1:mem=96GB
#PBS -P 13004299
#PBS -j oe
#PBS -l walltime=24:00:00

cd $PBS_O_WORKDIR
export CUDA_VISIBLE_DEVICES=""
module load gromacs/5.1.2
export GMXLIB=/home/users/astar/bii/liuwei/scratch/
export PATH="/home/users/astar/bii/lijg/package/aspire2a/gmx2024.3_d/bin:$PATH"

for i in {0..2000}; do
    seed=$((i * 1001))

    batch_idx=$(printf "%03d" $((i / 10000)))      # e.g., 000
    sub_id=$(printf "id_%05d" "$i")
    dir="300k_${batch_idx}/${sub_id}"
    
    mkdir -p "$dir"
    cd "$dir"

    FILE="box_nve.gro"
    if [ -f "$FILE" ]; then
        echo "$FILE exists. Skipping dir $dir..."
        cd - >/dev/null
        continue
    fi

    cp ../../common_files/* .

    gmx_d insert-molecules -f box.gro -ci SOL.gro -nmol 8 -o 8SOL.gro -seed "$seed"
    gmx_d grompp -f em1.mdp -c 8SOL.gro -p box8.top -o box_em1.tpr -maxwarn 444
    gmx_d mdrun -s box_em1.tpr -c box_em1.gro -g em1.log -v > job_em1.log 2>&1

    sed -i "s/gen_seed *= *[0-9]\+/gen_seed = ${seed}/" md_nvt.mdp

    gmx_d grompp -f md_nvt.mdp -c box_em1.gro -p box8.top -o box_nvt.tpr -maxwarn 44
    gmx_d mdrun -s box_nvt.tpr -c box_nvt.gro -e box_nvt.edr -g md_nvt.log -v > job_md_nvt_log 2>&1

    gmx_d grompp -f md_nve.mdp -c box_nvt.gro -p box8.top -o box_nve.tpr -maxwarn 44
    gmx_d mdrun -s box_nve.tpr -c box_nve.gro -e box_nve.edr -o box_nve.trr -g md_nve.log -v > job_md_nve_log 2>&1

    cd - >/dev/null
done

# The line below shows how to use set MD steps and how many cpus to use.
# gmx mdrun -nsteps 100000 -s box_nve.tpr -c box_nve.gro -e box_nve.edr -o box_nve.trr -g md_nve.log -v -ntmpi 1 -ntomp 8 > job_md_nve.log 2>&1
