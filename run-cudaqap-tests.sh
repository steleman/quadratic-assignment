#!/bin/bash

here="`pwd`"
outfile="cudaqap-output.out"

cat /dev/null > ${outfile}

for idx in \
  '100' \
  '144' \
  '256' \
  '400' \
  '625' \
  '900'
do
  echo "./cudaqap -d ./dstdata-${idx}.dat -f ./fldata-${idx}.dat"
  echo "%> ./cudaqap -d ./dstdata-${idx}.dat -f ./fldata-${idx}.dat" >> ${outfile} 2>&1
  ./cudaqap -d ./dstdata-${idx}.dat -f ./fldata-${idx}.dat >> ${outfile} 2>&1
  echo "" >> ${outfile} 2>&1

  echo "./cudaqap -d ./dstdata-${idx}.dat -f ./fldata-${idx}.dat"
  echo "%> ./cudaqap -d ./dstdata-${idx}.dat -f ./fldata-${idx}.dat" >> ${outfile} 2>&1
  ./cudaqap -d ./dstdata-${idx}.dat -f ./fldata-${idx}.dat >> ${outfile} 2>&1
  echo "" >> ${outfile} 2>&1

  echo "./cudaqap -d ./dstdata-${idx}.dat -f ./fldata-${idx}.dat"
  echo "%> ./cudaqap -d ./dstdata-${idx}.dat -f ./fldata-${idx}.dat" >> ${outfile} 2>&1
  ./cudaqap -d ./dstdata-${idx}.dat -f ./fldata-${idx}.dat >> ${outfile} 2>&1
  echo "" >> ${outfile} 2>&1
done

for idx in \
  '12a' \
  '20a' \
  '25a'
do
  echo "./cudaqap -s ./chr${idx}.dat"
  echo "%> ./cudaqap -s ./chr${idx}.dat" >> ${outfile} 2>&1
  ./cudaqap -s ./chr${idx}.dat >> ${outfile} 2>&1
  echo "" >> ${outfile} 2>&1

  echo "./cudaqap -s ./chr${idx}.dat"
  echo "%> ./cudaqap -s ./chr${idx}.dat" >> ${outfile} 2>&1
  ./cudaqap -s ./chr${idx}.dat >> ${outfile} 2>&1
  echo "" >> ${outfile} 2>&1

  echo "./cudaqap -s ./chr${idx}.dat"
  echo "%> ./cudaqap -s ./chr${idx}.dat" >> ${outfile} 2>&1
  ./cudaqap -s ./chr${idx}.dat >> ${outfile} 2>&1
  echo "" >> ${outfile} 2>&1
done

