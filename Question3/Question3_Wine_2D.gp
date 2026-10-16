set terminal pngcairo size 1000,750
set output 'Question3_Wine_2D.png'
set title 'Wine Quality - 2D Feature Distribution'
set xlabel 'Standardized Feature 1'
set ylabel 'Standardized Feature 2'
set grid
set key outside right
set datafile separator ','
set datafile columnheaders
plot 'wine_visualization.csv' using 1:($12==3?$2:1/0) with points pointtype 7 pointsize 0.4 title 'Class 3', \
'wine_visualization.csv' using 1:($12==4?$2:1/0) with points pointtype 7 pointsize 0.4 title 'Class 4', \
'wine_visualization.csv' using 1:($12==5?$2:1/0) with points pointtype 7 pointsize 0.4 title 'Class 5', \
'wine_visualization.csv' using 1:($12==6?$2:1/0) with points pointtype 7 pointsize 0.4 title 'Class 6', \
'wine_visualization.csv' using 1:($12==7?$2:1/0) with points pointtype 7 pointsize 0.4 title 'Class 7', \
'wine_visualization.csv' using 1:($12==8?$2:1/0) with points pointtype 7 pointsize 0.4 title 'Class 8', \
'wine_visualization.csv' using 1:($12==9?$2:1/0) with points pointtype 7 pointsize 0.4 title 'Class 9'
