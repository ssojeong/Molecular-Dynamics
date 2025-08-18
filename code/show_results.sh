#!/bin/bash

# HK20220426
echo "analyse $1"


#cat $1 | grep urmse     | grep train | cut -d " " -f 1,2,3,4 > "$1_lr.txt"
cat $1 | grep *prmse    | grep train | cut -d " " -f 1,2,3,4,5,6,7,8,9,10,11,12,13,14 > "$1_prmse.txt"
cat $1 | grep qrmse    | grep train | cut -d " " -f 1,2,3,4,5,6,7,8,9,10,11,12,13,14 > "$1_qrmse.txt"
cat $1 | grep ermse     | grep train | cut -d " " -f 1,2,3,4,5,6,7,8,9,10,11,12,13,14 > "$1_ermse.txt"
cat $1 | grep krmse     | grep train | cut -d " " -f 1,2,3,4,5,6,7,8,9,10,11,12,13,14 > "$1_krmse.txt"
cat $1 | grep urmse     | grep train | cut -d " " -f 1,2,3,4,5,6,7,8,9,10,11,12,13,14 > "$1_urmse.txt"
cat $1 | grep relurep   | grep train | cut -d " " -f 1,2,3,4,5,24,25,26,27,28,29,30,31,32 > "$1_relurep.txt"
cat $1 |  grep -A 1 mshape |  grep -A 1 train  | grep rep | cut -d " " -f 1,2,3,4,5,6,7,8,9,10 > "$1_rep.txt"
cat $1 | grep qshape   | grep train | cut -d " " -f 1,2,3,4,5,6,7,8,9,10,11,12,13,14 > "$1_qshape.txt"
cat $1 | grep pshape   | grep train | cut -d " " -f 1,2,3,4,5,15,16,17,18,19,20,21,22,23  > "$1_pshape.txt"
cat $1 | grep eshape   | grep train | cut -d " " -f 1,2,3,4,5,24,25,26,27,28,29,30,31,32 > "$1_eshape.txt"
cat $1 |  grep -A 1 mshape |  grep -A 1 train  | grep rep | cut -d " " -f 11,12,13,14,15,16,17,18,19,20 > "$1_poly.txt"
cat $1 | grep For | grep 'x axis' | cut -d " " -f 2,4,5,6,7 > "$1_outputx.txt"
cat $1 | grep For | grep 'y axis' | cut -d " " -f 2,4,5,6,7 > "$1_outputy.txt"
cat $1 | grep total    | grep train > "$1_total.txt"
cat $1 | grep 'tau 0'     | grep train > "$1_tau.txt"
#cat $1 | grep time

cat $1 | grep *prmse | grep eval | cut -d " " -f 1,2,3,4,5,6,7,8,9,10,11,12,13,14 > "$1_prmse_eval.txt"
cat $1 | grep qrmse | grep eval | cut -d " " -f 1,2,3,4,5,6,7,8,9,10,11,12,13,14 > "$1_qrmse_eval.txt"
cat $1 | grep krmse  | grep eval | cut -d " " -f 1,2,3,4,5,6,7,8,9,10,11,12,13,14 > "$1_krmse_eval.txt"
cat $1 | grep urmse  | grep eval | cut -d " " -f 1,2,3,4,5,6,7,8,9,10,11,12,13,14 > "$1_urmse_eval.txt"
cat $1 | grep relurep | grep eval | cut -d " " -f 1,2,3,4,5,24,25,26,27,28,29,30,31,32> "$1_relurep_eval.txt"
cat $1 | grep ermse  | grep eval | cut -d " " -f 1,2,3,4,5,6,7,8,9,10,11,12,13,14 > "$1_ermse_eval.txt"
cat $1 |  grep -A 1 mshape |  grep -A 1 eval  | grep rep | cut -d " " -f 1,2,3,4,5,6,7,8,9,10 > "$1_rep_eval.txt"
cat $1 | grep qshape   | grep eval | cut -d " " -f 1,2,3,4,5,6,7,8,9,10,11,12,13,14 > "$1_qshape_eval.txt"
cat $1 | grep pshape   | grep eval | cut -d " " -f 1,2,3,4,5,15,16,17,18,19,20,21,22,23 > "$1_pshape_eval.txt"
cat $1 | grep eshape   | grep eval | cut -d " " -f 1,2,3,4,5,24,25,26,27,28,29,30,31,32 > "$1_eshape_eval.txt"
cat $1 | grep total | grep eval > "$1_total_eval.txt"
cat $1 | grep 'tau 0'     | grep eval > "$1_tau_eval.txt"


