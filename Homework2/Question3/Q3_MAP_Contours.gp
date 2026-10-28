
set terminal pngcairo size 1000,800 enhanced font "Arial,12"

set xlabel "X Position"
set ylabel "Y Position"
set xrange [-2:2]
set yrange [-2:2]
set size ratio -1
set grid
set key outside right top

# Use identical contour levels for all K
set contour base
unset surface
set view map
set cntrparam levels discrete 0.5,1,2,4,6,10,15,20,30,40,60,80

# Plot styles
set style line 1 lc rgb "blue" lw 1.5
set style line 2 lc rgb "red" pt 1 ps 2 lw 2
set style line 3 lc rgb "green" pt 7 ps 1.5
set style line 4 lc rgb "black" pt 2 ps 2 lw 2

do for [K=1:4] {

    gridfile = sprintf("Q3_MAP_Grid_K%d.dat", K)
    contourfile = sprintf("Q3_Contour_Data_K%d.dat", K)

    truefile = sprintf("Q3_K%d_True.dat", K)
    mapfile = sprintf("Q3_K%d_MAP.dat", K)
    landmarkfile = sprintf("Q3_K%d_Landmarks.dat", K)

    # Step 1: Generate contour data from the grid
    set table contourfile
    splot gridfile using 1:2:3 with lines
    unset table

    # Step 2: Plot contours and position markers
    unset contour
    set output sprintf("Q3_MAP_Contour_K%d.png", K)
    set title sprintf("MAP Objective Function Contours (K = %d)", K)

    plot contourfile using 1:2 with lines ls 1 title "MAP Objective", \
         truefile using 1:2 with points ls 2 title "True Position", \
         mapfile using 1:2 with points ls 3 title "MAP Estimate", \
         landmarkfile using 1:2 with points ls 4 title "Landmarks"

    # Re-enable contours for next iteration
    set contour base
}

unset output
