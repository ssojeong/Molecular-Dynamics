#!/bin/bash

file1='results/gap10_b0.01_n128-128-128_d256_poly1'
cat $file1/0.log > $file1/log_combine
./show_results.sh $file1/log_combine

file2='results/gap10_b0.01_n128-128-128_d256_poly2'
cat $file2/0.log > $file2/log_combine
./show_results.sh $file2/log_combine

file3='results/gap10_b0.01_n128-128-128_d256_poly3'
cat $file3/0.log > $file3/log_combine
./show_results.sh $file3/log_combine

file4='results/gap10_b0.01_n128-128-128_d256_poly4'
cat $file4/0.log > $file4/log_combine
./show_results.sh $file4/log_combine

file5='results/gap10_b0.01_n128-128-128_d256'
cat $file5/0.log > $file5/log_combine
./show_results.sh $file5/log_combine

file6='results/gap10_b0.01_n128-128-128-128-128-128_d256_poly1'
cat $file6/0.log > $file6/log_combine
./show_results.sh $file6/log_combine

file7='results/gap10_b0.01_n128-128-128-128-128-128_d256_poly3'
cat $file7/0.log > $file7/log_combine
./show_results.sh $file7/log_combine

python loss_weight_ws1.py load_file_tau0.1.dict poly1_ 0 "water liquid; dpt=45000; pwnet 3; poly deg=1" &
python loss_weight_ws1.py load_file_tau0.1.dict poly2_ 0 "water liquid; dpt=45000; pwnet 3; poly deg=2" &
python loss_weight_ws1.py load_file_tau0.1.dict poly3_ 0 "water liquid; dpt=45000; pwnet 3; poly deg=3" &
python loss_weight_ws1.py load_file_tau0.1.dict poly4_ 0 "water liquid; dpt=45000; pwnet 3; poly deg=4" &
python loss_weight_ws1.py load_file_tau0.1.dict poly5_ 0 "water liquid; dpt=45000; pwnet 3; poly deg=5" &
python loss_weight_ws1.py load_file_tau0.1.dict poly1pw6_ 0 "water liquid; dpt=45000; pwnet 6; poly deg=1" &
python loss_weight_ws1.py load_file_tau0.1.dict poly3pw6_ 0 "water liquid; dpt=45000; pwnet 6; poly deg=3"
