#!/bin/bash

# HK20220426
echo "analyse $1"

cat $1 | grep prmse    | grep train | cut -d " " -f 1,2,3,4,5,6,7,8
cat $1 | grep qrmse    | grep train | cut -d " " -f 1,2,3,4,5,6,7,8
cat $1 | grep emae     | grep train | cut -d " " -f 1,2,3,4,5,6,7,8
cat $1 | grep emse     | grep train | cut -d " " -f 1,2,3,4,5,9,10,11,12
cat $1 | grep mmae     | grep train | cut -d " " -f 1,2,3,4,5,13,14,15,16
cat $1 | grep eweight  | grep train | cut -d " " -f 1,2,3,4,5,22,23,24,25
cat $1 | grep poly     | grep train | cut -d " " -f 1,2,3,4,5,26,27,28,29
cat $1 | grep total    | grep train
cat $1 | grep tau      | grep train
cat $1 | grep time

#cat $1 | grep prmse | grep eval | cut -d " " -f 1,2,3,4,5,6,7,8
#cat $1 | grep qrmse | grep eval | cut -d " " -f 1,2,3,4,5,6,7,8
#cat $1 | grep emae  | grep eval | cut -d " " -f 1,2,3,4,5,6,7,8
#cat $1 | grep mmae  | grep eval | cut -d " " -f 1,2,3,4,5,13,14,15,16
#cat $1 | grep total | grep eval

