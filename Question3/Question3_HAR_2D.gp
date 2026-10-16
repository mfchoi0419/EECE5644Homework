set terminal pngcairo size 1000,750
set output 'Question3_HAR_2D.png'
set title 'HAR - 2D Feature Distribution'
set xlabel 'Standardized Feature 1'
set ylabel 'Standardized Feature 2'
set grid
set key outside right
set datafile separator ','
set datafile columnheaders
plot 'har_visualization.csv' using 1:($562==1?$2:1/0) with points pointtype 7 pointsize 0.4 title 'Class 1', \
'har_visualization.csv' using 1:($562==2?$2:1/0) with points pointtype 7 pointsize 0.4 title 'Class 2', \
'har_visualization.csv' using 1:($562==3?$2:1/0) with points pointtype 7 pointsize 0.4 title 'Class 3', \
'har_visualization.csv' using 1:($562==4?$2:1/0) with points pointtype 7 pointsize 0.4 title 'Class 4', \
'har_visualization.csv' using 1:($562==5?$2:1/0) with points pointtype 7 pointsize 0.4 title 'Class 5', \
'har_visualization.csv' using 1:($562==6?$2:1/0) with points pointtype 7 pointsize 0.4 title 'Class 6'
