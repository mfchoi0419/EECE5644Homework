
set terminal pngcairo size 1100,700 enhanced font "Arial,12"
set output "Q2_MAP_ML_Comparison.png"

set title "Question 2: MAP vs ML Validation MSE"
set xlabel "Prior Variance (Gamma)"
set ylabel "Validation Mean Squared Error"

set logscale x 10
set format x "10^{%L}"

set grid
set key top right

set datafile separator comma

set style line 1 lc rgb "#2563eb" lw 2 pt 7 ps 1.2
set style line 2 lc rgb "#dc2626" lw 2 dt 2

plot "Q2_MAP_ML_Comparison.csv" using 1:3 with linespoints ls 1 title "MAP Validation MSE", \
     "Q2_MAP_ML_Comparison.csv" using 1:4 with lines ls 2 title "ML Validation MSE"
