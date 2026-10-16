set terminal pngcairo size 1000,750 enhanced font 'Arial,12'
set output 'Question1_LDA_ROC.png'
set title 'Fisher LDA ROC Curve'
set xlabel 'False Positive Rate (FPR)'
set ylabel 'True Positive Rate (TPR)'
set xrange [0:1]
set yrange [0:1]
set grid
set key bottom right
set datafile separator ','
set object 1 circle at 0.0388043,0.94142 size screen 0.007 fillstyle solid fillcolor 'red' front
plot 'Question1_LDA_ROC.csv' using 2:3 with lines linewidth 2 title 'Fisher LDA ROC', x with lines dashtype 2 title 'Random Classifier'
