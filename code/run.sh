#!/bin/bash

file4='results/traj_len08ws08tau0.05ngrid12api0lw8421ew1repw10poly4l_dpt45000'
cat $file4/log > $file4/log_combine
./show_results.sh $file4/log_combine

file5='results/traj_len08ws08tau0.02ngrid12api0lw8421ew1repw10poly4l_dpt45000'
cat $file5/log > $file5/log_combine
./show_results.sh $file5/log_combine

file6='results/traj_len08ws08tau0.05ngrid12api0lw8421ew1repw10poly4lreload_dpt45000'
cat $file6/log > $file6/log_combine
./show_results.sh $file6/log_combine

file7='results/traj_len08ws08tau0.02ngrid12api0lw8421ew1repw10poly4lreload_dpt45000'
cat $file7/log > $file7/log_combine
./show_results.sh $file7/log_combine

file9='results/traj_len08ws08tau0.05ngrid12api0lw8421ew1repw10poly1l_dpt45000'
cat $file9/log > $file9/log_combine
./show_results.sh $file9/log_combine

file8='results/traj_len08ws08tau0.02ngrid12api0lw8421ew1repw10poly1l_dpt45000'
cat $file8/log > $file8/log_combine
./show_results.sh $file8/log_combine

python loss_weight.py load_file_tau0.1.dict traj8ws8tau0.05ngrid12w8421ew1repw10poly4l_ 0,1/8,0,1/4,0,1/2,0,1 "scratch liquid; dpt=45000; 64 particles; ai tau=0.05; 1GPU; batch size 16; L=1/8L2+1/4L4+1/2L6+L8; ew=1; repw=10; poly deg=4" &

python loss_weight.py load_file_tau0.1.dict traj8ws8tau0.02ngrid12w8421ew1repw10poly4l_ 0,1/8,0,1/4,0,1/2,0,1 "scratch liquid; dpt=45000; 64 particles; ai tau =0.02; 1GPU; batch size 16; L=1/8L2+1/4L4+1/2L6+L8; ew=1; repw=10; poly deg=4" &

python loss_weight.py load_file_tau0.1.dict traj8ws8tau0.05ngrid12w8421ew1repw10poly4lreload_ 0,1/8,0,1/4,0,1/2,0,1 "reload; liquid; dpt=45000; 64 particles; ai tau=0.05; 1GPU; batch size 16; L=1/8L2+1/4L4+1/2L6+L8; ew=1; repw=10; poly deg=4" &

python loss_weight.py load_file_tau0.1.dict traj8ws8tau0.02ngrid12w8421ew1repw10poly4lreload_ 0,1/8,0,1/4,0,1/2,0,1 "reload; liquid; dpt=45000; 64 particles; ai tau =0.02; 1GPU; batch size 16; L=1/8L2+1/4L4+1/2L6+L8; ew=1; repw=10; poly deg=4" &

python loss_weight.py load_file_tau0.1.dict traj8ws8tau0.05ngrid12w8421ew1repw10poly1l_ 0,1/8,0,1/4,0,1/2,0,1 "reload; liquid; dpt=45000; 64 particles; ai tau=0.05; 1GPU; batch size 16; L=1/8L2+1/4L4+1/2L6+L8; ew=1; repw=10; poly deg=1" &

python loss_weight.py load_file_tau0.1.dict traj8ws8tau0.02ngrid12w8421ew1repw10poly1l_ 0,1/8,0,1/4,0,1/2,0,1 "reload; liquid; dpt=45000; 64 particles; ai tau =0.02; 1GPU; batch size 16; L=1/8L2+1/4L4+1/2L6+L8; ew=1; repw=10; poly deg=1" &


