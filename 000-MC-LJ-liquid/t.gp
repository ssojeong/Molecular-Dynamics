set size square
set grid
N=2
rho=0.336
BoxSize=sqrt(N/rho)
#BoxSize=sqrt(N*rho)*2.
set xrange[0:BoxSize]
set yrange[0:BoxSize]

p 't' u 1:2 w p pt 6 ps 2 lt 1 not , \
  't' u 3:4 w p pt 6 ps 2 lt 2 not

