set terminal pngcairo size 1600,1200 enhanced font "Arial,14"
set output "MAP_3D.png"

set title "MAP Classification - 3D Gaussian Mixture"
set xlabel "x1"
set ylabel "x2"
set zlabel "x3"

set grid
set key outside right
set view 65,35
set pointsize 0.6

set xyplane at -4

splot \
    "MAP_Class1_Correct.dat" using 1:2:3 with points pt 7 lc rgb "green" title "Class 1 Correct", \
    "MAP_Class1_Incorrect.dat" using 1:2:3 with points pt 7 lc rgb "red" title "Class 1 Incorrect", \
    "MAP_Class2_Correct.dat" using 1:2:3 with points pt 6 lc rgb "green" title "Class 2 Correct", \
    "MAP_Class2_Incorrect.dat" using 1:2:3 with points pt 6 lc rgb "red" title "Class 2 Incorrect", \
    "MAP_Class3_Correct.dat" using 1:2:3 with points pt 9 lc rgb "green" title "Class 3 Correct", \
    "MAP_Class3_Incorrect.dat" using 1:2:3 with points pt 9 lc rgb "red" title "Class 3 Incorrect", \

unset output