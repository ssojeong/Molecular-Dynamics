#PBS -q normal
#PBS -l select=1:ncpus=1:mem=96GB
#PBS -P 13004299
#PBS -j oe
#PBS -l walltime=24:00:00

cd $PBS_O_WORKDIR
export CUDA_VISIBLE_DEVICES=""
module load gromacs/5.1.2
module load pytorch/2.6.0-py3
export GMXLIB=/home/users/astar/bii/liuwei/scratch/
export PATH="/home/users/astar/bii/lijg/package/aspire2a/gmx2024.3_d/bin:$PATH"

# gmx energy -s box_nve.tpr -f box_nve.edr -o ET    # This is for energy analysis

for i in {1..10000}; do
    pt_path="$PBS_O_WORKDIR/torch_pts/$(printf "%05d" "$i").pt"

    if [ -f "$pt_path" ]; then
        continue
    fi

    batch_idx=$(printf "%03d" $((i / 10000)))      # e.g., 000
    sub_id=$(printf "id_%05d" "$i")
    dir="batch_${batch_idx}/${sub_id}"

    cd "$dir"
    rm uncompressed.gro

    FILE="box_nve.trr"
    if [ ! -f "$FILE" ]; then
        echo "$FILE does not exist. Skipping dir $dir..."
        cd - >/dev/null
        continue
    fi

    gmx_d trjconv -s box_nve.tpr -b 1 -f box_nve.trr -pbc whole -center -ur compact -o uncompressed.gro << EOF
0
0
EOF

    python "$PBS_O_WORKDIR/dataprocessor_per_id.py" --ip_path "uncompressed.gro" --op_path "$pt_path"
    rm uncompressed.gro

    cd - >/dev/null

done

