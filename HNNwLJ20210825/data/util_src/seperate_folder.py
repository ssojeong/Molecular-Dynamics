#!/bin/bash
# mv folders that already pt files to tmp folder 
data_dir=$1
mv_folder=$2

foldernames=`ls -l $data_dir |grep "^d" | awk -F" " '{print $9}'`

for i in $foldernames; do

  folder_dir="$1/$i"
  file_dir=`ls $1/$i/*_id*`
  echo "foldername $folder_dir"
  if [[ -s "$file_dir" ]]
  then
    echo "*pt file exists... mv $folder_dir $2"
    mv  $folder_dir $2
  else 
    echo "nothing"
  fi  

done 
