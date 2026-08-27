#!/bin/bash

here="`pwd`"
ts="`date +%Y%m%d`"
outfile="cudaqap-output-${ts}.out"

cat /dev/null > ${outfile}

echo "Warning: the 20a and 25a tests will take a *** very long time *** to complete."

for idx in \
  '12a' \
  '20a' \
  '25a'
do
  for ((ctr=0; ctr<10; ctr++))
  do
    echo "./cudaqap -s ./sample-data/chr${idx}.dat"
    echo "%> ./cudaqap -s ./sample-data/chr${idx}.dat" >> ${outfile} 2>&1
    ./cudaqap -s ./sample-data/chr${idx}.dat >> ${outfile} 2>&1
    echo "" >> ${outfile} 2>&1
  done
done

