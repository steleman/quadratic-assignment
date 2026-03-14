My own research on the Quadratic Assignment Problem
===================================================

This is some of my own research on the Quadratic Assignment Problem (QAP).
The main program is `qap`. `genrandomdata` is a random data generator to be used with `qap`.

This is Work-In-Progress. I haven't touched these in over a year. I've been working on other things since then. I'm just using Github as a file backup. :-)

There are two `Makefiles` here: `Makefile.gcc` and `Makefile.clang`. It should be pretty obvious what they are and what they do. :-)

The `sample-data` directory contains sample data.

Saying `<program-name> --help` will show you how to use each program.

All of this work was done on Linux (Fedora).
--------------------------------------------

Quadratic Assignment Problem
============================

This repo contains a Proof-Of-Concept implementation of the brute-force approach to the Quadratic Assignment Problem. The program name is `qap`.

```
%> ./qap --help
Usage: qap [-h | --help]
           [ -f <flow-input-file> | --flinput <flow-input-file>]
           [ -d <distance-input-file> | --dstinput <distance-input-file>]
           [-p | --print]
           (print vector contents).
```

Example:

```
%> ./qap -f ./fldata-100.dat -d ./dstata-100.dat -p
Minimum cost: 165908
Iterations: 3628800
Flow Graph Vector:
{ { 6, 7, 22, 23, 30, 52, 65, 68, 73, 77 },
  { 2, 5, 20, 39, 48, 52, 57, 76, 79, 82 },
  { 3, 4, 16, 18, 30, 41, 43, 45, 60, 75 },
  { 6, 14, 49, 55, 61, 62, 65, 67, 76, 97 },
  { 22, 26, 30, 39, 41, 47, 51, 61, 75, 82 },
  { 9, 24, 46, 53, 62, 68, 74, 76, 97, 98 },
  { 8, 12, 13, 16, 20, 44, 69, 73, 78, 83 },
  { 2, 10, 15, 25, 50, 52, 67, 73, 76, 85 },
  { 23, 60, 72, 76, 78, 82, 87, 88, 90, 95 },
  { 17, 25, 27, 29, 30, 64, 74, 85, 99, 100 } }

Distance Vector:
{ { 1, 4, 6, 20, 21, 60, 61, 63, 70, 78 },
  { 5, 13, 15, 41, 46, 50, 56, 59, 64, 75 },
  { 2, 19, 24, 49, 52, 53, 73, 78, 81, 91 },
  { 2, 4, 26, 34, 38, 45, 59, 91, 95, 100 },
  { 23, 26, 36, 41, 45, 47, 74, 87, 93, 95 },
  { 1, 17, 30, 34, 49, 53, 71, 88, 95, 96 },
  { 3, 10, 15, 27, 30, 45, 58, 59, 86, 93 },
  { 15, 21, 26, 35, 65, 72, 82, 88, 93, 99 },
  { 3, 7, 16, 19, 24, 53, 69, 82, 93, 100 },
  { 3, 11, 17, 19, 21, 37, 78, 92, 93, 95 } }

Assignment Vector:
{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }

Clock Resolution: 0.000000001.
CPU time: 0.000000000128 second(s).
```

The auxiliary program `genrandomdata` can be used to generate random data to be used by the `qap` program. Example usage of `genrandomdata`:

```
%> ./genrandomdata -q -f -o ./fldata-1000.dat -u 100 -m 10 -M 10 -n 1000
%> ./genrandomdata -q -f -o ./dstdata-1000.dat -u 100 -m 10 -M 10 -n 1000
```

The use case above will generate Flow Graph and Distance Graph input files for the `qap` program.


```
%> ./genrandomdata --help
Usage: genrandomdata  [--help | -h]
                      (print this message)
                      [--quiet | -q]
                      (nothing printed to stdout)
                      [--auto | -a]
                      (auto-fill the generated sets to cover the full universe)
                      [--fixed | -f]
                      (constant (fixed) set size. maxsize == minsize).
                      [--output <filename> | -o <filename>]
                      (output filename)
                      [--usize <universe-size> | -u <universe-size>]
                      (number of elements in the universe)
                      [--minsize <minimum-subset-size> | -m <minimum-subset-size>]
                      (minimum size of a generated subset)
                      [--maxsize <maximum-subset-size> | -M <maximum-subset-size>]
                      (maximum size of a generated subset)
                      [--nsets <number-of-sets> | -n <number-of-sets>]
```

Note: `genrandomdata` is *very* slow, because it has to fit the randomly generated set element numbers within the boundary constraints indicated on command-line.

The files suffixed with `*.dat` - included here - are sample input files suitable for use by `qap`.

Files with names beginning with `fldata` are Flow Graph input files. Files with names beginning with `dstdata` are Distance Graph input files.

`Makefile` is for Linux. `Makefile.clang` is for MacOS.

