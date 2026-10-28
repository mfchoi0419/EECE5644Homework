
set terminal pngcairo size 1000,750 enhanced font "Arial,12"
set output "Q1_Optimal_ROC.png"

set title "Question 1 - Theoretically Optimal Classifier ROC"
set xlabel "False Positive Rate (FPR)"
set ylabel "True Positive Rate (TPR)"

set xrange [0:1]
set yrange [0:1]

set grid
set key bottom right

set datafile separator comma

set style line 1 lc rgb "#2563eb" lw 2
set style line 2 lc rgb "#dc2626" pt 7 ps 1.8
set style line 3 lc rgb "#808080" dt 2 lw 1.5


set label 1 "Min P(error) = 0.2727" at 0.19,0.53


# Minimum-error operating point
minFPR = 907.0 / 6050.0
minTPR = 2130.0 / 3950.0

# Red circular marker
set object 1 circle at first minFPR,minTPR \
    size screen 0.008 \
    fillstyle solid 1.0 \
    fillcolor rgb "red" \
    front

# Annotation
set label 1 "Min P(error) = 0.2727" \
    at 0.19,0.53 front

# Plot ROC curve and random baseline
plot \
    "Q1_Optimal_ROC.csv" every ::1 using 1:2 \
        with lines lc rgb "blue" lw 2 \
        title "Optimal Classifier ROC", \
    x with lines lc rgb "gray" dt 2 \
        title "Random Classifier"

set output

