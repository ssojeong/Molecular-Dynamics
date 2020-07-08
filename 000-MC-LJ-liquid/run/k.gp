set grid
set title '2d LJ potential, rho=0.01'
set xlabel 'Temperature'
set ylabel 'Specific Heat'
set xrange[0.1:1.2]
p \
"k2"  u 1:2 w p pt 6 ps 2 lw 1 lt 1 t 'N=2',\
"k2"  u 1:2 smooth csplines    lt 1 not , \
"k3"  u 1:2 w p pt 6 ps 1 lw 1 lt 2 t 'N=3',\
"k3"  u 1:2 smooth csplines    lt 2 not , \
"k4"  u 1:2 w p pt 6 ps 1 lw 1 lt 3 t 'N=4',\
"k4"  u 1:2 smooth csplines    lt 3 not , \
"k5"  u 1:2 w p pt 6 ps 1 lw 1 lt 4 t 'N=5',\
"k5"  u 1:2 smooth csplines    lt 4 not , \
"k6"  u 1:2 w p pt 6 ps 1 lw 1 lt 5 t 'N=6',\
"k6"  u 1:2 smooth csplines    lt 5 not , \
"k7"  u 1:2 w p pt 6 ps 1 lw 1 lt 6 t 'N=7',\
"k7"  u 1:2 smooth csplines    lt 6 not , \
"k8"  u 1:2 w p pt 6 ps 1 lw 1 lt 7 t 'N=8',\
"k8"  u 1:2 smooth csplines    lt 7 not , \
0 

