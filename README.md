Quadratic Assignment Problem
============================

This repo contains an implementation of a serial (brute-force) approach to the Quadratic Assignment Problem, and a branch-and-bound solver that do the same job without enumerating. There are five programs:

- `qap` - the CPU/serial brute-force reference implementation, in `qap.cpp`.
- `cudaqap` - the CUDA/GPU brute-force port, in `cudaqap.cu`.
- `cgbncudaqap` - a variant of `cudaqap` with wide arithmetic, in `cgbncudaqap.cu`. It counts iterations per thread in a `uint64_t` and totals them in an arbitrary-precision NVIDIA CGBN accumulator, and it widens the permutation rank to 128 bits, which lifts the problem-size cap from `N <= 20` to `N <= 24`.
- `cgbncudaqapbb` - the GPU branch and bound, in `cgbncudaqapbb.cu`. It uses `qapbb`'s Gilmore-Lawler bound on a tree shaped so that the work can be spread across tens of thousands of threads, and it checks its own coverage of the permutation space in arbitrary precision.
- `genrandomdata` - the random instance generator, in `genrandomdata.cpp`.

`cgbncudaqapbb-design.pdf` is a 23-page design document for `cgbncudaqapbb.cu`, covering the bound, the search tree, the parallel decomposition, the arbitrary-precision accounting, and credits for every algorithm the implementation uses. It is generated from `cgbncudaqapbb-design.tex` by `pdflatex` (run twice).

The three brute-force solvers enumerate *every* permutation of the assignment vector and keep the minimum of `sum_ij FLG[i][j] * DST[AS[i]][AS[j]]`. They must agree on three things: the minimum cost, the iteration count - which is exactly `N!` for an `N x N` instance - and the minimizing assignment itself.

The two branch-and-bound solvers return the same minimum cost without visiting anything like `N!` nodes, which is what puts `N = 25` and `N = 30` in reach at all. See **Performance** below for the comparisons, and [`cudaqap.md`](https://github.com/steleman/quadratic-assignment/blob/main/cudaqap.md) for the full measurements.

Building
--------

```
%> make                                 # builds genrandomdata and qap
%> make qap                             # CPU solver only
%> make genrandomdata                   # data generator only
%> make -f Makefile.cuda                # all three GPU solvers
%> make -f Makefile.cuda cudaqap        # GPU brute force only
%> make -f Makefile.cuda cgbncudaqap    # CGBN variant only
%> make -f Makefile.cuda cgbncudaqapbb  # GPU branch and bound only
```

Plain `make` and `make all` are equivalent and build both CPU programs; `make clean` removes them. `make -f Makefile.cuda` builds `cudaqap`, `cgbncudaqap` and `cgbncudaqapbb`. `qapbb` is a single self-contained C file with no dependencies and is not in any Makefile.

`Makefile.cuda` hardcodes `CUDA_VERSION = 12.9` and `CUDA_CC = 89` (Ada, `sm_89`). Change those two variables to match your CUDA installation and your GPU's Compute Capability. Details are in `Makefile.cuda`.

`cgbncudaqap` and `cgbncudaqapbb` have two dependencies `cudaqap` does not.
They need my fork of NVIDIA's [CGBN](https://github.com/steleman/nvidia-cgbn).
`CGBN_INCDIR` in `Makefile.cuda`, defaulting to `/usr/local/include`,
which must be the directory *containing* `cgbn/`, so that `#include <cgbn/cgbn.h>`
resolves - and it links GMP. GMP is not optional: `cgbn.h` is compiled on the
host pass as well as the device pass, and it selects its host backend from
whether `<gmp.h>` has already been seen. Without it the host pass stops at
`#error You must use GMP for now`. GMP also formats the arbitrary-precision
counters in decimal at the end of the run, and builds `cgbncudaqapbb`'s table
of factorials.

`Makefile` is for Linux, `Makefile.clang` for macOS:

```
%> make -f Makefile.clang    # genrandomdata + qap, built with clang++
```

`Makefile.clang` is not simply the Linux Makefile with the compiler swapped. It
drops `_GNU_SOURCE`, which is glibc-specific, and must *not* define
`_XOPEN_SOURCE`: on Darwin that selects strict POSIX and hides the BSD
extensions these sources rely on, `getopt_long` among them. It defines
`_DARWIN_C_SOURCE` instead. It does not build `cudaqap`, since CUDA is not
available on current macOS. `clock_getres(3)` needs macOS 10.12 or later.

qap
---

```
%> ./qap --help
Usage: qap [-h | --help]
           [-s <input-file> | --single-file <input-file>]
           (use single-file matrix input format).
           [ -f <flow-input-file> | --flinput <flow-input-file>]
           [ -d <distance-input-file> | --dstinput <distance-input-file>]
           [-p | --print]
           (print vector contents).
```

Example:

```
%> ./qap -f ./fldata-100.dat -d ./dstdata-100.dat -p
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
{ 9, 8, 7, 6, 5, 3, 4, 2, 1, 0 }

Clock Resolution: 0.000000001.
CPU time: 0.140266343 second(s).
```

The `Assignment Vector` printed by `-p` is the permutation that achieves the reported minimum cost. Where several permutations tie, both programs report the lexicographically first. `qap` and `cudaqap` return the same assignment, not merely the same cost.

The reported `CPU time:` / `GPU time:` covers the solve only. It excludes reading the input files, and for `cudaqap` it also excludes CUDA context setup, which costs a roughly fixed 0.13 seconds of wall clock. If the system clock cannot be read, the timing lines are omitted and the search result is still reported.

cudaqap
-------

```
%> ./cudaqap --help
Usage: cudaqap [-h | --help]
           [-s <input-file> | --single-file <input-file>]
           (use single-file matrix input format).
           [ -f <flow-input-file> | --flinput <flow-input-file>]
           [ -d <distance-input-file> | --dstinput <distance-input-file>]
           [ -m <gpu-shared-memory-size> | --shmem-size <gpu-shared-memory-size>]
           (minimum GPU dynamic shared memory per block, in KB.
            default is 0 - allocate exactly what the assignment
            vector requires).
           [-p | --print] (print vector contents).
```

`cudaqap` takes the same inputs as `qap` and adds `-m`, which raises the floor on dynamic shared memory per block. By default it allocates exactly what the assignment vector needs, so `-m` is not normally required.

```
%> ./cudaqap -s ./chr12a.dat
Minimum cost: 9552
Iterations:   479001600
NumThreads:   89088
CPU Clock Resolution: 0.000000001.
GPU time: 0.104282046 second(s).
```

That is the QAPLIB published optimum for `chr12a`, over all 479001600 permutations. With `-p`, both programs report the same minimizing assignment, `{ 6, 4, 11, 1, 0, 2, 8, 10, 9, 5, 7, 3 }`. See **Performance** below for how that compares with `qap`, and for what `NumThreads` means.

cgbncudaqap
-----------

`cgbncudaqap` takes exactly the same options as `cudaqap` and solves exactly the
same problem. It differs only in how it counts and how it addresses the
permutation space:

- **Per-thread iteration count.** Each thread keeps its own `uint64_t` counter
  and increments it once per permutation it actually scores. `cudaqap` instead
  did one `atomicAdd` of `Hi - Lo` into a single device-wide `uint64_t`.
- **Arbitrary-precision global count.** The total is `N!`, which passes `2^64`
  at `N = 21` - `21!` is 51090942171709440000, about 2.8x what a `uint64_t`
  holds. When a thread finishes it stores its local count, and two cooperative
  CGBN kernels sum those counts into one 1024-bit accumulator. 1024 bits is
  enough for `170!`, so the accumulator is never the binding constraint.
- **128-bit permutation ranks.** This is what actually raises the size cap.
  `cudaqap` refuses `N > 20` because it partitions work by lexicographic
  permutation rank and a rank has to fit in a `uint64_t`. A rank is per-thread
  state, and a CGBN value is *not* a scalar - its limbs are spread across TPI
  cooperating lanes, so a single thread can neither hold one nor update one.
  The widest type one thread can own is `__uint128_t`, and `34!` (2.95e38) is
  the largest factorial below `2^128`.

```
%> ./cgbncudaqap -s ./chr12a.dat
Minimum cost: 9552
Iterations:   479001600
Permutations: 479001600
NumThreads:   89088
CPU Clock Resolution: 0.000000001.
GPU time: 0.104996840 second(s).
CPU Clock Resolution: 0.000000001.
CGBN reduction: 0.000101231 second(s).
```

Two output lines are new. `Permutations:` is `N!` computed on the host, printed
next to `Iterations:` so that the check that matters - did the rank ranges tile
the whole space exactly once - is visible in the output rather than something
you have to work out. `CGBN reduction:` times the two accumulation kernels
separately from the search, because they are the price of the
arbitrary-precision counter and the point of the variant is to be able to see
what that price is. It is about 0.2 ms, independent of `N`.

The search itself is not measurably slower. `cgbncudaqap` against `cudaqap`:
`chr12a` 0.1047 s vs 0.1049 s, `fldata-144` 0.1058 s vs 0.1051 s, N=13 1.643 s
vs 1.636 s (medians of 5-7 runs), N=14 28.07 s vs 28.43 s and N=15 457.42 s vs
456.92 s (single runs) - deltas of -0.2%, +0.7%, +0.4%, -1.2% and +0.11%, i.e.
run-to-run noise with no consistent sign. N=15 is the tightest measurement of
the set: half a second apart over 1307674368000 permutations. The 128-bit loop
counter costs nothing next to the `O(N^2)` cost evaluation inside it.

Both `N` limits are checked up front and reported:

```
%> ./cgbncudaqap -s ./chr25a.dat
Problem size 25 gives 278577766582812248275 permutations per thread, which does not fit the 64-bit per-thread iteration counter.

%> ./cgbncudaqap -f ./fldata-10000.dat -d ./dstdata-10000.dat
Problem size 100 is too large: the permutation rank does not fit in 128 bits (the limit is 34).
```

The first is the effective cap: the rank type reaches 34, but the *local*
counter is a `uint64_t`, so `N!` divided by the launch's thread count must also
fit in 64 bits. On this GPU that binds at `N = 24` (11143110663312489931
permutations per thread, just inside the limit) and fails at `N = 25`. The cap
is therefore device-dependent, since it moves with the thread count.

qapbb and cgbncudaqapbb
-----------------------

The three programs above enumerate. These two prune, and the difference is not
a constant factor: `chr25a` is 25 x 25, so brute force would have to score
15511210043330985984000000 permutations. `qapbb` proves the optimum in eleven
seconds and `cgbncudaqapbb` in two and a half.

`qapbb` is the CPU reference, a literal translation of the FORTRAN IV program
by T. Boenniger, R. E. Burkard and K.-H. Stratmann, with the linear sum
assignment routine by U. Derigs. Each node's bound is a Gilmore-Lawler bound
obtained by solving an LSAP. It reads standard input and has no options:

```
%> ./qapbb < ./chr25a.dat
 no linear term.



         EXACT SOLUTION:
             1 -->25
             2 -->12
             3 --> 5
             ...
            25 --> 9

         OBJECTIVE FUNCTION VALUE:  3796
```

The layout, the blank lines and the asterisks it prints when a value does not
fit its field are the FORTRAN `FORMAT` statements, reproduced.

`cgbncudaqapbb` takes the same input files and options as the other GPU
solvers, minus `-m` (it uses no shared memory) and plus two of its own:

```
%> ./cgbncudaqapbb --help
Usage: cgbncudaqapbb [-h | --help]
           [-s <input-file> | --single-file <input-file>]
           (use single-file matrix input format).
           [ -f <flow-input-file> | --flinput <flow-input-file>]
           [ -d <distance-input-file> | --dstinput <distance-input-file>]
           [ -t <max-threads> | --max-threads <max-threads>]
           (cap on the launch thread count. default is 0 -
            as many as occupancy and device memory allow).
           [ -k <frontier-size> | --frontier <frontier-size>]
           (how many live nodes to expand the search frontier
            to before the depth-first phase. default is 0 -
            32 per thread).
           [-p | --print] (print vector contents).

%> ./cgbncudaqapbb -f ./fldata-625.dat -d ./dstdata-625.dat
Minimum cost: 41048904
B&B nodes:    8596072
Permutations: 15511210043330985984000000
Accounted:    15511210043330985984000000
Residual:     0
Start bound:  41048904
NumThreads:   14848
Frontier:     0 live node(s) at depth 23 - solved breadth-first
Busiest:      795 node(s), 0 idle thread(s)
CPU Clock Resolution: 0.000000001.
GPU time: 2.735117544 second(s).
CPU Clock Resolution: 0.000000001.
CGBN reduction: 0.000632827 second(s).
```

Three of those lines deserve explanation.

**`Accounted:` and `Residual:` are the correctness check**, and they replace
`Iterations: == N!`. A pruning search never visits `N!` of anything, so the
count that proved the brute-force solvers had covered the space is gone. What
replaces it is exact. Every node the search finishes with - pruned by the
bound, or evaluated as the last two facilities - stands for precisely `(N-L)!`
complete permutations, where `L` is its level. Those numbers must add up to
`N!` and nothing else. `Accounted:` is that sum, computed on the device in
1024-bit CGBN arithmetic because at `N = 30` it is 2.65e32; `Residual:` is
`N! - Accounted`, formed as a **signed** CGBN value so that a search which
covered something twice is as visible as one that missed something. Anything
other than `0` means the search did not tile the space.

**`Frontier:` and `Busiest:` describe how the work was spread.** Under a
pruning bound, subtrees differ in size by five orders of magnitude, so cutting
the tree at a fixed depth and dealing out the pieces does not work: one thread
gets the hard part of the proof and the rest of the device idles. Measured on
this instance with a fixed depth-5 split: 8.68 million nodes, 257352 of them in
a single thread, 62 seconds of wall clock of which 61 were that one thread.
`cgbncudaqapbb` instead grows the frontier - it expands every live node's
children in parallel, one level at a time, until there are enough live nodes to
keep the device busy, and only then switches to depth-first. `Busiest:` is the
node count of the thread that did the most; on the run above it is 795 out of
8596072.

**`B&B nodes:` is not reproducible, and does not need to be.** Every thread
prunes against one shared incumbent, so how much speculative work gets done
depends on timing. `chr25a` measured 7915417, 7939937 and 10001210 nodes on
three consecutive runs - all returning 3796 with a residual of zero. Where the
starting heuristic already found the optimum, the incumbent never moves and the
node count is identical run to run.

Both branch-and-bound solvers do their arithmetic in 32-bit integers, which
they inherit from the FORTRAN, where 10^9 stands in for infinity.
`cgbncudaqapbb` therefore refuses an input whose objective could get close:
`sum(flow) * max(distance)` bounds every intermediate the search forms, and
if that reaches `2^30` the program says so and stops instead of wrapping and
reporting a wrong answer with a straight face. That admits the 32 x 32
`fldata-1024` pair and rejects the 100 x 100 one.

Two smaller things follow from pruning. `Start bound:` is a 2-opt local search
run on the host before the GPU is touched, which gives the search something to
prune against from the first node; where it equals the reported cost, the
heuristic was already optimal and the GPU work was the proof. And ties may
resolve differently from the brute-force solvers: pruning is
`bound >= incumbent`, so a second permutation of equal cost is discarded rather
than compared. In practice the assignment matched `qap` exactly on all eight
small pair-format inputs, but that is a property of these inputs, not a
guarantee.

Performance
-----------

Measured on an RTX 4080 Laptop GPU (sm_89) against a Core i9-14900HX. Both
columns are solve time only, as described above; medians of several runs.

| instance                     | N  | permutations | minimum cost | threads | GPU solve | CPU solve | speedup |
|------------------------------|----|--------------|--------------|---------|-----------|-----------|---------|
| `fldata-9`   / `dstdata-9`   |  3 |            6 |          153 |      32 | 0.000097s | 0.000001s |       - |
| `fldata-16`  / `dstdata-16`  |  4 |           24 |          313 |      32 | 0.000140s | 0.000002s |       - |
| `fldata-25`  / `dstdata-25`  |  5 |          120 |         3654 |     128 | 0.000164s | 0.000004s |       - |
| `fldata-64`  / `dstdata-64`  |  8 |        40320 |       111640 |   40704 | 0.000241s | 0.001578s |      7x |
| `fldata-81`  / `dstdata-81`  |  9 |       362880 |       138868 |   89088 | 0.000301s | 0.017856s |     59x |
| `fldata-100` / `dstdata-100` | 10 |      3628800 |       165908 |   89088 | 0.000810s | 0.140467s |    173x |
| `fldata-144` / `dstdata-144` | 12 |    479001600 |       953450 |   89088 |  0.10415s |  23.2684s |    223x |
| `fldata-12a` / `dstdata-12a` | 12 |    479001600 |         9552 |   89088 |  0.10513s |  23.2753s |    221x |
| `chr12a` (`-s`)              | 12 |    479001600 |         9552 |   89088 |  0.10466s |  23.2784s |    222x |

The GPU loses on the smallest instances, where a kernel launch costs more than
the whole search, and only repays its fixed ~0.13 s of process startup around
N=10 in wall-clock terms. On solve time it climbs to roughly 220x by N=12.

The branch-and-bound pair is measured against each other on the same machine,
since brute force cannot reach the sizes where they are interesting. Both
columns are solve time; `qapbb` includes its whole run, `cgbncudaqapbb` its GPU
time as reported.

| instance                       | N  | minimum cost | B&B nodes  | `qapbb` CPU | `cgbncudaqapbb` GPU | speedup |
|--------------------------------|----|--------------|------------|-------------|---------------------|---------|
| `chr12a` (`-s`)                | 12 |         9552 |       1479 |   0.0016 s  |            0.0267 s |   0.06x |
| `fldata-144` / `dstdata-144`   | 12 |       953450 |       3567 |   0.0016 s  |            0.0319 s |   0.05x |
| `fldata-225` / `dstdata-225`   | 15 |      2135449 |        902 |   0.0028 s  |            0.0622 s |   0.05x |
| `fldata-256` / `dstdata-256`   | 16 |      2942118 |      17162 |   0.0292 s  |            0.1068 s |   0.27x |
| `chr20a` (`-s`)                | 20 |         2192 |     158236 |   0.0548 s  |            0.1677 s |   0.33x |
| `fldata-400` / `dstdata-400`   | 20 |      9779382 |     100978 |   0.164 s   |            0.2887 s |   0.57x |
| `chr25a` (`-s`)                | 25 |         3796 |    ~7.9e6  |  11.2 s     |            2.46 s   |    4.6x |
| `fldata-625` / `dstdata-625`   | 25 |     41048904 |    8596072 |  71.8 s     |            2.74 s   |     26x |
| `fldata-900` / `dstdata-900`   | 30 |    119265331 |   33496281 |  2248 s     |           19.99 s   |    112x |
| `fldata-1024` / `dstdata-1024` | 32 |    189919789 | 1484193064 |  see below  |         1275.31 s   |       - |

The shape is the same as for brute force and for the same reason: below about
`N = 20` the tree is too small to spread over 15000 threads, and the GPU is
paying launch and setup costs against a search a single core finishes in
milliseconds. The crossover is around `N = 20`, and past it the gap widens
quickly, because the node count is what grows and the node count is what
parallelises. The `N = 30` row is thirty-seven minutes against twenty seconds,
and the two agree: `qapbb`'s assignment re-scores to 119265331, the same
optimum.

`N = 32` is the largest instance here, and the only one that behaves
differently. It is the first where the frontier reaches its target before the
tree runs out, so the breadth-first phase stops at depth 7 with 665914 live
nodes and hands them to the depth-first phase - the mode every smaller
instance skips:

```
Minimum cost: 189919789
B&B nodes:    1484193064
Permutations: 263130836933693530167218012160000000
Accounted:    263130836933693530167218012160000000
Residual:     0
Start bound:  189919789
NumThreads:   14848
Frontier:     665914 live node(s) at depth 7 into the depth-first phase
Busiest:      2366935 node(s), 0 idle thread(s)
GPU time: 1275.305643775 second(s).
```

1.48 billion nodes in 21 minutes - 1.16 million bounds a second, each one an
LSAP - and 32! accounted for exactly:
263130836933693530167218012160000000, which is 118 bits, fifty-four past what
a `uint64_t` holds. `Busiest:` is 2366935 against 1484193064 total, so the
thread that drew the largest of the 665914 subtrees carried 0.16% of the
search. `qapbb` was started on the same instance for the CPU column and was
stopped, unfinished, after two hours; a single-threaded host build of the GPU's
own algorithm was stopped at one hour forty. `N = 30` is the last size where
both solvers were run to completion, and the `N = 32` row rests on the
program's own checks: `Accounted:` is exactly `32!`, and the reported
assignment re-scores to the reported cost against the untouched input
matrices.

The two solvers do not explore the same tree, which is worth keeping in mind
when reading that table. `qapbb` branches on the single assignment with maximal
alternative cost - an excellent rule on one core, and one that gives a node no
usable index, so there is no cheap way to name "the k-th subtree" for a thread
to take. `cgbncudaqapbb` places facilities in a fixed order instead, one per
level, which makes a node an ordinary partial permutation at the cost of a
weaker branching rule. On the same instance and the same bound, the GPU tree is
the larger one; it is also the one that can be handed out.

The QAPLIB optima are reproduced by both: 9552 for `chr12a`, 2192 for `chr20a`
and 3796 for `chr25a`.

`NumThreads` is not a fixed figure: `cudaqap` sizes the launch from the problem
and the device, from a single 32-thread block on a 3x3 instance up to full
occupancy (58 SMs x 1536 = 89088 threads) once there is enough work.
`cudaqap.md` has the per-size block/grid breakdown, wall-clock figures, and the
runtime projections beyond N=15. `cgbncudaqap` uses the identical sizing code
and picks the identical geometry at every `N` the two share.

Input data
----------

Two formats are accepted:

1. **Pair format** (`-f` and `-d`) - two files, each a bare `N x N` matrix with one row per line and no header. Values may be separated by commas, spaces, or tabs. Files named `fldata-*` are Flow Graph inputs; files named `dstdata-*` are Distance Graph inputs. The two must agree in size.

2. **Single-file / QAPLIB format** (`-s`) - one file: `N` on the first line, then the `N x N` flow matrix, then the `N x N` distance matrix. `chr12a.dat`, `chr20a.dat`, and `chr25a.dat` are QAPLIB instances in this format.

The `*.dat` files included here are sample inputs. Be aware that the numbering is not consistent: for most of them the number is the cell count, so `fldata-9.dat` is 3x3 and `fldata-900.dat` is 30x30 - but `fldata-20.dat` is 20x20, and `fldata-12a.dat`/`dstdata-12a.dat` are the pair-format form of the 12x12 `chr12a` instance. Every `fldata-*`/`dstdata-*` pair is square and well formed. `fldata-1000.dat` follows the `fldata-20.dat` reading where the number is the *dimension*, so it is 1000x1000; the rest use the number as the cell count. All three of `-1000`, `-10000` and `-65536` are far past every solver's size limit - `N <= 20` for `cudaqap`, `N <= 24` for `cgbncudaqap`, and for the branch-and-bound pair a limit set by how hard the instance is rather than by arithmetic - so they are structurally valid but not solvable; the brute-force solvers report the problem size and exit. The `rawdata-*`, `testdata-*`, and `dstflatdata-*` files are scratch artifacts from an earlier workflow and are not solver inputs.

Problem size
------------

Brute force is `O(N! * N^2)`, so the reach is small. On the GPU, `N <= 12` finishes in well under a second, `N = 13` in under two seconds, `N = 14` in about thirty seconds, `N = 15` in about eight minutes and `N = 16` in about two and a half hours - all measured. Beyond that it runs away fast: `N = 17` projects to nearly two days and `N = 20` to decades. `cudaqap` rejects `N > 20` outright, because the permutation ranking that partitions work across GPU threads requires `N!` to fit in a 64-bit integer:

```
%> ./cudaqap -s ./chr25a.dat
Problem size 25 is too large: the permutation count does not fit in 64 bits (the limit is 20).
```

`cgbncudaqap` moves that cap to `N <= 24`, which is worth being precise about:
it removes an *arithmetic* limit, not a computational one. The reach is set by
`N! * N^2`, and no counter width changes that. The sizes it newly admits are
these:

| N  |                    iterations = N! | 64-bit? | projected time |
|----|------------------------------------|---------|----------------|
| 20 |                2432902008176640000 | yes     | 64 years       |
| 21 |               51090942171709440000 | **no**  | 1.5 thousand years |
| 22 |             1124000727777607680000 | **no**  | 39 thousand years |
| 23 |            25852016738884976640000 | **no**  | 970 thousand years |
| 24 |           620448401733239439360000 | **no**  | 28 million years |

So the honest summary is that `cgbncudaqap` makes `N = 21..24` *expressible* -
the ranks address the space, the counter reports `N!` exactly - and leaves them
just as unreachable as before. The value is in the architecture, not the reach:
nothing in the program now imposes a limit below the physical one.

Branch and bound is the thing that changes the reach rather than the notation.
`cgbncudaqapbb` has no factorial-shaped size limit at all: it never forms a
permutation rank over the whole space, so nothing has to fit in 64 or 128 bits
for a given `N`. What it has instead is a memory limit and a difficulty limit.
Each thread needs its own workspace - the sorted-row tables alone are
`O(N^3)` words, 48 KB per thread at `N = 25` and 618 KB at `N = 60` - so past
about `N = 50` the launch starts shrinking to fit device memory. And how long
a run takes depends on how much of the tree the bound can cut, which is a
property of the instance and not of its size. Two 25 x 25 instances
in the table above differ by a factor of 4.5 in `qapbb` time. The 1024-bit
accumulator caps `N` at 170, which is not a limit anyone will meet.

The largest runs actually carried to completion:

| N  | instance | solver        |        permutations | threads | minimum cost | solve time | verified          |
|----|----------|---------------|---------------------|---------|--------------|------------|-------------------|
| 13 | A        | `cudaqap`     |       6,227,020,800 |   89088 |       440532 |     1.66 s | cost + assignment |
| 14 | A        | `cudaqap`     |      87,178,291,200 |   89088 |       423242 |    28.41 s | cost + assignment |
| 15 | A        | `cudaqap`     |   1,307,674,368,000 |   89088 |       521907 |   458.27 s | iteration count   |
| 15 | B        | `cudaqap`     |   1,307,674,368,000 |   89088 |       362694 |   456.92 s | cost + assignment |
| 15 | B        | `cgbncudaqap` |   1,307,674,368,000 |   89088 |       362694 |   457.42 s | cost + assignment |
| 16 | A        | `cudaqap`     |  20,922,789,888,000 |   83520 |       552418 |  8511.36 s | iteration count   |

Each ran on a random square instance generated for the purpose and iterated
exactly `N!` times. N=15 appears twice because it was run on two different
inputs; instance B went through both GPU solvers with `-p`, and both returned
the same assignment,
`{ 14, 13, 12, 11, 10, 7, 9, 8, 6, 5, 4, 3, 2, 1, 0 }`, which re-scores against
the input matrices to 362694. Where the `verified` column says `iteration
count`, that is all that was checked, since the assignment is printed at the end
of the run and getting it after the fact means running the search again.

Under branch and bound the corresponding list is a different shape, because
these runs finish in seconds rather than hours:

| N  | instance             | solver            | minimum cost | nodes      | solve time |
|----|----------------------|-------------------|--------------|------------|------------|
| 25 | `chr25a`             | `cgbncudaqapbb`   |         3796 |    ~7.9e6  |     2.46 s |
| 25 | `fldata-625` pair    | `cgbncudaqapbb`   |     41048904 |    8596072 |     2.74 s |
| 30 | `fldata-900` pair    | `cgbncudaqapbb`   |    119265331 |   33496281 |    19.99 s |
| 32 | `fldata-1024` pair   | `cgbncudaqapbb`   |    189919789 | 1484193064 |  1275.31 s |

Each accounted for exactly `N!` permutations, with a residual of zero:
15511210043330985984000000 at `N = 25`, 265252859812191058636308480000000 at
`N = 30` and 263130836933693530167218012160000000 at `N = 32`. The `N = 30`
optimum and node count were confirmed against a host build of the same
algorithm - 119265331 in 33496339 nodes, 420.7 s on one core - and
independently by `qapbb`, whose own assignment re-scores to the same value.

`cudaqap.md` has the full projection table by problem size.

genrandomdata
-------------

The auxiliary program `genrandomdata` can be used to generate random data to be used by the `qap` program. It builds from `genrandomdata.cpp` via plain `make` or `make genrandomdata`. Example usage:

```
%> ./genrandomdata -q -f -o ./myfl-10.dat  -u 100 -m 10 -M 10 -n 10
%> ./genrandomdata -q -f -o ./mydst-10.dat -u 100 -m 10 -M 10 -n 10
```

The use case above generates a 10x10 Flow Graph and Distance Graph pair for the
`qap` program. `-o` overwrites without asking, so pick names that do not collide
with the shipped `fldata-*`/`dstdata-*` files.

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

### Practical notes

`genrandomdata` writes one generated set per line, `--nsets` lines of between
`--minsize` and `--maxsize` values drawn from `1 .. --usize`. With `-f` the size
is fixed, so the file is `--nsets` rows by `--minsize` columns.

**`qap` needs a square matrix, so `--nsets` must equal the set size**, which is
why the example above pairs `-n 10` with `-m 10 -M 10`. Historically the shipped
`fldata-1000.dat` / `dstdata-1000.dat` were made with `-n 1000` against a set
size of 10, giving 1000 rows of 10 values - not a valid QAP input. Those files
have since been regenerated as a proper 1000x1000 pair.

An N x N instance is generated with `-f -m N -M N -n N -u <max-value>`, and is
now fast at any size the solvers could plausibly consume:

```
%> ./genrandomdata -q -f -o ./myfl-12.dat  -u 144 -m 12 -M 12 -n 12
%> ./genrandomdata -q -f -o ./mydst-12.dat -u 144 -m 12 -M 12 -n 12
%> ./qap -f ./myfl-12.dat -d ./mydst-12.dat
```

(Pick output names that do not collide with the shipped `fldata-*`/`dstdata-*`
files, which `-o` would overwrite without asking.)

`genrandomdata` used to be extremely slow. It drew from `lrand48()`'s full
`0 .. 2^31-1` range and redrew until the value happened to fall inside the
requested bounds, costing on the order of `2^31 / range` draws per value - and
with `-f`, where exactly one set size is acceptable, about `2^31` draws per set.
`-q -f -u 100 -m 10 -M 10 -n 20` took 117 seconds. It now scales the draw into
range instead, and the same command takes about 3 milliseconds; a 256x256 pair
takes about 20.

Full coverage of the universe `1 .. --usize` is only required when you ask for
it with `--auto` / `-a`, which keeps generating sets until the universe is
covered - at the cost of possibly emitting more than `--nsets` rows, which
breaks squareness. Without `-a` the sets are written as generated. (Earlier the
coverage check was applied to every run, so a plain run that happened not to
cover the universe printed "Generated Universe Sets do not cover the entire
Universe", exited 1 and discarded everything it had produced.)

Invalid size ranges are now rejected up front rather than hanging: `--minsize`
must be non-zero and no greater than `--maxsize`, and `--maxsize` cannot exceed
`--usize`, since a subset of distinct elements cannot be wider than the
universe it is drawn from.
