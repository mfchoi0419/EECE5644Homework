set terminal pngcairo size 1000,750
set output 'Question3_HAR_3D.png'
set title 'HAR - 3D Feature Distribution'
set xlabel 'Standardized Feature 1'
set ylabel 'Standardized Feature 2'
set zlabel 'Standardized Feature 3'
set grid
set key outside right
set datafile separator ','
set datafile columnheaders
set view 60,30
set lmargin 12
set zlabel offset 2,0
splot 'har_visualization.csv' using 1:2:(column(562)==1?column(3):1/0) with points pointtype 7 pointsize 0.4 title 'Class 1', \
'har_visualization.csv' using 1:2:(column(562)==2?column(3):1/0) with points pointtype 7 pointsize 0.4 title 'Class 2', \
'har_visualization.csv' using 1:2:(column(562)==3?column(3):1/0) with points pointtype 7 pointsize 0.4 title 'Class 3', \
'har_visualization.csv' using 1:2:(column(562)==4?column(3):1/0) with points pointtype 7 pointsize 0.4 title 'Class 4', \
'har_visualization.csv' using 1:2:(column(562)==5?column(3):1/0) with points pointtype 7 pointsize 0.4 title 'Class 5', \
'har_visualization.csv' using 1:2:(column(562)==6?column(3):1/0) with points pointtype 7 pointsize 0.4 title 'Class 6'
