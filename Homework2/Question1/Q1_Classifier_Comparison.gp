
set terminal pngcairo size 1400,800 enhanced font "Arial,12"
set output "Q1_Classifier_Comparison.png"

set datafile separator comma

set title "Question 1: Classifier Validation Error Comparison"
set ylabel "Classification Error Rate"
set xlabel "Classifier" offset 0,-4

set yrange [0:0.55]
set ytics 0,0.05,0.55
set grid ytics

set style fill solid 0.85 border -1
set boxwidth 0.6
set key off

# Increase bottom margin
set bmargin 9

# Set horizontal labels below the bars
set xtics rotate by 0 offset 0,-1

set xtics ("Bayes\nOptimal" 0, \
           "Linear\n20" 1, \
           "Linear\n200" 2, \
           "Linear\n2000" 3, \
           "Quadratic\n20" 4, \
           "Quadratic\n200" 5, \
           "Quadratic\n2000" 6)

plot "Q1_Classifier_Comparison.csv" every ::1 \
    using 0:3 with boxes lc rgb "#4472C4"
