#!/bin/bash

file1='results/traj_len08ws08tau0.1ngrid12api0lw8421ew1repw10poly1l_dpt100'
cat $file1/log > $file1/log_combine
./show_results.sh $file1/log_combine

file2='results/traj_len08ws08tau0.1ngrid12api0lw8421ew1repw10poly1l_dpt1000'
cat $file2/log > $file2/log_combine
./show_results.sh $file2/log_combine

file3='results/traj_len08ws08tau0.1ngrid12api0lw8421ew1repw10poly4l_dpt1000'
cat $file3/log > $file3/log_combine
./show_results.sh $file3/log_combine

file4='results/traj_len08ws08tau0.1ngrid12api0lw8421ew1repw10poly1l_dpt45000'
cat $file4/log > $file4/log_combine
./show_results.sh $file4/log_combine

file5='results/traj_len08ws08tau0.1ngrid12api0lw8421ew1repw10poly4l_dpt45000'
cat $file5/log > $file5/log_combine
./show_results.sh $file5/log_combine

#python loss_weight.py load_file_tau0.1.dict traj8ws8tau0.1ngrid12w8421ew1repw10poly1l_ 0,1/8,0,1/4,0,1/2,0,1 "liquid; dpt=100; L=1/8L2+1/4L4+1/2L6+L8; ew=1; repw=10; poly deg=1"  &
#python loss_weight.py load_file_tau0.1.dict traj8ws8tau0.1ngrid12w8421ew1repw10poly1ls1000_ 0,1/8,0,1/4,0,1/2,0,1 "liquid; dpt=1000; L=1/8L2+1/4L4+1/2L6+L8; ew=1; repw=10; poly deg=1" &
#python loss_weight.py load_file_tau0.1.dict traj8ws8tau0.1ngrid12w8421ew1repw10poly4ls1000_ 0,1/8,0,1/4,0,1/2,0,1 "liquid; dpt=1000; L=1/8L2+1/4L4+1/2L6+L8; ew=1; repw=10; poly deg=4" &
python loss_weight.py load_file_tau0.1.dict traj8ws8tau0.1ngrid12w8421ew1repw10poly1ls45000_ 0,1/8,0,1/4,0,1/2,0,1 "liquid; dpt=45000; L=1/8L2+1/4L4+1/2L6+L8; ew=1; repw=10; poly deg=1" &
python loss_weight.py load_file_tau0.1.dict traj8ws8tau0.1ngrid12w8421ew1repw10poly4ls45000_ 0,1/8,0,1/4,0,1/2,0,1 "liquid; dpt=45000; L=1/8L2+1/4L4+1/2L6+L8; ew=1; repw=10; poly deg=4"


