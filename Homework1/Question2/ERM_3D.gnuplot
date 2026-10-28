set terminal pngcairo size 1600,1200 enhanced font "Arial,14"

set xlabel "x1"
set ylabel "x2"
set zlabel "x3"

set grid
set key outside right
set pointsize 0.4

set view 65, 35

set xrange [-4:7]
set yrange [-4:7]
set zrange [-4:7]

# ---------------------------------------
# ERM Lambda_10
# ---------------------------------------
set output "ERM10_3D.png"
set title "ERM Classification - Lambda_10"

splot \
    "ERM10_Class1_Correct.dat" using 1:2:3 with points pt 7 lc rgb "green" title "Class 1 Correct", \
    "ERM10_Class1_Incorrect.dat" using 1:2:3 with points pt 7 lc rgb "red" title "Class 1 Incorrect", \
    "ERM10_Class2_Correct.dat" using 1:2:3 with points pt 6 lc rgb "green" title "Class 2 Correct", \
    "ERM10_Class2_Incorrect.dat" using 1:2:3 with points pt 6 lc rgb "red" title "Class 2 Incorrect", \
    "ERM10_Class3_Correct.dat" using 1:2:3 with points pt 9 lc rgb "green" title "Class 3 Correct", \
    "ERM10_Class3_Incorrect.dat" using 1:2:3 with points pt 9 lc rgb "red" title "Class 3 Incorrect", \

# ---------------------------------------
# ERM Lambda_100
# ---------------------------------------

set output "ERM100_3D.png"
set title "ERM Classification - Lambda_100"

splot \
    "ERM100_Class1_Correct.dat" using 1:2:3 with points pt 7 lc rgb "green" title "Class 1 Correct", \
    "ERM100_Class1_Incorrect.dat" using 1:2:3 with points pt 7 lc rgb "red" title "Class 1 Incorrect", \
    "ERM100_Class2_Correct.dat" using 1:2:3 with points pt 6 lc rgb "green" title "Class 2 Correct", \
    "ERM100_Class2_Incorrect.dat" using 1:2:3 with points pt 6 lc rgb "red" title "Class 2 Incorrect", \
    "ERM100_Class3_Correct.dat" using 1:2:3 with points pt 9 lc rgb "green" title "Class 3 Correct", \
    "ERM100_Class3_Incorrect.dat" using 1:2:3 with points pt 9 lc rgb "red" title "Class 3 Incorrect", \

set output