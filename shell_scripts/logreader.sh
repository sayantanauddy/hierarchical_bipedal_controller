#!/bin/bash

logfile=$1
echo "Logfile: $logfile"

# Create a file to store the results
resultfile=$(echo $logfile| sed 's/\.[^.]*$//')_results.txt
touch $resultfile

echo "Resultfile: $resultfile"
echo "Logfile: $logfile" >> $resultfile
echo "Resultfile: $resultfile" >> $resultfile
echo "" >> $resultfile

# Read the Min fitness for each generation
echo 'Min fitness for generations:' >> $resultfile
grep 'Min' $logfile | cut -d' ' -f5 >> $resultfile
echo "" >> $resultfile

# Read the Max fitness for each generation
echo 'Max fitness for generations:' >> $resultfile
grep 'Max' $logfile | cut -d' ' -f5 >> $resultfile
echo "" >> $resultfile

# Read the Avg fitness for each generation
echo 'Avg fitness for generations:' >> $resultfile
grep 'Avg' $logfile | cut -d' ' -f5 >> $resultfile
echo "" >> $resultfile

# Read the Std fitness for each generation
echo 'Std fitness for generations:' >> $resultfile
grep 'Std' $logfile | cut -d' ' -f5 >> $resultfile
echo "" >> $resultfile

# Read the x-distance and up times for the Max fitnesses
listVar=$(grep 'Max' $logfile | cut -d' ' -f5)

# Read the x-distances
echo "X-Distances for max fitnesses" >> $resultfile
for maxfit in $listVar; do
    maxdist=$(grep -B15 "fitness: $maxfit" $logfile | grep 'end_x' | cut -d' ' -f4 | sed 's/end_x=//' | sed 's/,//')
    echo $maxdist >> $resultfile
done
echo "" >> $resultfile

# Read the up_times
echo "Up times for max fitnesses" >> $resultfile
for maxfit in $listVar; do
    up_time=$(grep -B15 "fitness: $maxfit" $logfile | grep 'up_time' | cut -d' ' -f4 | sed 's/up_time=//' | sed 's/fitness//')
    echo $up_time >> $resultfile
done
echo "" >> $resultfile
