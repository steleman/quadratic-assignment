#!/bin/bash

here="`pwd`"
ts="`date +%Y%m%d`"
outfile="cudaqap-output-${ts}.out"

cat /dev/null > ${outfile}

for idx in \
  '100' \
  '144' \
  '256' \
  '400' \
  '625' \
  '900'
do
  for ((ctr=0; ctr<10; ctr++))
  do
    echo "./cudaqap -d ./dstdata-${idx}.dat -f ./fldata-${idx}.dat"
    echo "%> ./cudaqap -d ./dstdata-${idx}.dat -f ./fldata-${idx}.dat" >> ${outfile} 2>&1
    ./cudaqap -d ./dstdata-${idx}.dat -f ./fldata-${idx}.dat >> ${outfile} 2>&1
    echo "" >> ${outfile} 2>&1
  done
done

for idx in \
  '12a' \
  '20a' \
  '25a'
do
  for ((ctr=0; ctr<10; ctr++))
  do
    echo "./cudaqap -s ./chr${idx}.dat"
    echo "%> ./cudaqap -s ./chr${idx}.dat" >> ${outfile} 2>&1
    ./cudaqap -s ./chr${idx}.dat >> ${outfile} 2>&1
    echo "" >> ${outfile} 2>&1
  done
done

