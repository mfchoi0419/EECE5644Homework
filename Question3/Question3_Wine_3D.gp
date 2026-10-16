set terminal pngcairo size 1000,750
set output 'Question3_Wine_3D.png'
set title 'Wine Quality - 3D Feature Distribution'
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
splot 'wine_visualization.csv' using 1:2:(column(12)==3?column(3):1/0) with points pointtype 7 pointsize 0.4 title 'Class 3', \
'wine_visualization.csv' using 1:2:(column(12)==4?column(3):1/0) with points pointtype 7 pointsize 0.4 title 'Class 4', \
'wine_visualization.csv' using 1:2:(column(12)==5?column(3):1/0) with points pointtype 7 pointsize 0.4 title 'Class 5', \
'wine_visualization.csv' using 1:2:(column(12)==6?column(3):1/0) with points pointtype 7 pointsize 0.4 title 'Class 6', \
'wine_visualization.csv' using 1:2:(column(12)==7?column(3):1/0) with points pointtype 7 pointsize 0.4 title 'Class 7', \
'wine_visualization.csv' using 1:2:(column(12)==8?column(3):1/0) with points pointtype 7 pointsize 0.4 title 'Class 8', \
'wine_visualization.csv' using 1:2:(column(12)==9?column(3):1/0) with points pointtype 7 pointsize 0.4 title 'Class 9'
