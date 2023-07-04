#!/bin/bash

# combine trajectory along nsamples at several folders
# run something like this
# $1 dir_foldername; $2 newfilename $3 compilefile $4 pt filename
# e.g. ./combineKfiles_folders.sh ../gen_by_MC/test ../gen_by_MC/test/T0.03/newfilename.pt combine2files_sample_wise.py n4_data.pt

data_dir=$1
newfilename=$2
compilefile=$3
filename=$4

echo "=== data dir ==="
echo "$data_dir"

foldernames=`ls -l $data_dir |grep "^d" | awk -F" " '{print $9}'`
ls -l $data_dir |grep "^d" | awk -F" " '{print $9}' | wc -l

n=0

for i in $foldernames; do

  file_dir="$1/$i/$4"
  echo "$((n+1)) load $file_dir"

  if [ $n -eq 0 ]
  then
    #echo "cp $file_dir ./tmp"
    cp $file_dir ./tmp
  else
    #echo "combine 2 file for tmp $file_dir tmp"
    python $compilefile tmp $file_dir tmp
  fi
  n=$((n+1))
done

#echo cp tmp $newfilename
cp tmp $newfilename

