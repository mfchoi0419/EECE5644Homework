set terminal pngcairo size 1000, 750
set output 'Question1_ROC.png'
set title 'Question1 Part A: ROC Curve'
set xlabel 'False Positive Rate (FPR)'
set ylabel 'True Positive Rate (TPR)'
set xrange [0:1]
set yrange [0:1]
set grid
set key outside right
set datafile separator ','
plot 'Question1_ROC.csv' using 2:3 with lines linewidth 2 title 'Gaussian ROC', \
x with lines dashtype 2 title 'Random Classifier', \
'Question1_OperatingPoints.csv' every ::1::1 using 1:2 with points pointtype 7 pointsize 2 title 'Theoretical Threshold', \
'Question1_OperatingPoints.csv' every ::2::2 using 1:2 with points pointtype 5 pointsize 2 title 'Empirical Threshold'
