set logscale xy
set terminal png
set output 'error.png'
plot 'disc_error.csv' with linespoints title "disc",\
     'bilinear_error.csv' with linespoints title "bilinear",\
     0.3*x**-0.75 with lines title "pow(N, -0.75)",\
     0.3*x**-1.5 with lines title "pow(N, -1.5)"
