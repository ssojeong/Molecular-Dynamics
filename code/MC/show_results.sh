#!/bin/bash
# ./show_results.sh  log

echo "analyse $1"

folder="$1"

rm $folder/*txt

for i in "$folder"/*; do
  echo $i
  filename=$(basename "$i")
  cat $i | grep 's0 ' | grep E | cut -d " " -f 3,5 > log/${filename}_s0.txt
#  cat $i | grep 's1 ' | grep E | cut -d " " -f 3,5 > log/${filename}_s1.txt
#  cat $i | grep 's2 ' | grep E | cut -d " " -f 3,5 > log/${filename}_s2.txt
#  cat $i | grep s3 | grep E | cut -d " " -f 3,5 > log/${filename}_s3.txt
#  cat $i | grep s4 | grep E | cut -d " " -f 3,5 > log/${filename}_s4.txt
#  cat $i | grep s5 | grep E | cut -d " " -f 3,5 > log/${filename}_s5.txt
#  cat $i | grep s6 | grep E | cut -d " " -f 3,5 > log/${filename}_s6.txt
#  cat $i | grep s7 | grep E | cut -d " " -f 3,5 > log/${filename}_s7.txt
#  cat $i | grep s8 | grep E | cut -d " " -f 3,5 > log/${filename}_s8.txt
#  cat $i | grep s9 | grep E | cut -d " " -f 3,5 > log/${filename}_s9.txt


done

