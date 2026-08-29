cudaqap results
===============

`cudaqap` is the CUDA port of `qap`. Both enumerate the full permutation space
and keep the minimum of `sum_ij FLG[i][j] * DST[AS[i]][AS[j]]`, so for any input
the two must agree on three things: the minimum cost, the iteration count -
which is exactly `N!` - and the minimizing assignment itself. Every run below
was cross-checked against `qap` on the same input, and every assignment was
re-scored independently to confirm it reproduces the reported cost.

`cgbncudaqap` is a third solver, a variant of `cudaqap` that replaces the 64-bit
iteration counter with a per-thread `uint64_t` plus an arbitrary-precision
NVIDIA CGBN total, and the 64-bit permutation rank with a 128-bit one. It is
covered in its own section at the end; everything between here and there
describes `cudaqap`, and `cgbncudaqap` matches it measurement for measurement
except where that section says otherwise.

Build and run:

```
%>> make -f Makefile.cuda
%>> ./cudaqap -f ./fldata-144.dat -d ./dstdata-144.dat
%>> ./cudaqap -s ./chr12a.dat
%>> ./cgbncudaqap -s ./chr12a.dat
```

Sample output
-------------

```
%>> ./cudaqap -s ./chr12a.dat
Minimum cost: 9552
Iterations:   479001600
NumThreads:   89088
CPU Clock Resolution: 0.000000001.
GPU time: 0.103186507 second(s).

%>> ./cudaqap -f ./fldata-144.dat -d ./dstdata-144.dat
Minimum cost: 953450
Iterations:   479001600
NumThreads:   89088
CPU Clock Resolution: 0.000000001.
GPU time: 0.102246301 second(s).
```

`chr12a` reproduces the QAPLIB published optimum of 9552 over all 479001600
permutations. Adding `-p` also prints the assignment that achieves it,
`{ 6, 4, 11, 1, 0, 2, 8, 10, 9, 5, 7, 3 }`, which is the same permutation `qap`
reports - the two agree on the assignment, not just the cost. Where several
permutations tie on cost, both report the lexicographically first.

The `GPU time:`/`CPU time:` line brackets the solve only - kernel launch through
`cudaDeviceSynchronize()` for `cudaqap`, the permutation loop for `qap`. It
excludes input parsing, CUDA context setup, and teardown, which together cost
`cudaqap` a roughly fixed 0.13 s of process wall clock on this machine. Both
columns are given below. If the clock itself fails, the line is omitted and the
search result is still printed.

Timings
-------

Medians: 7 runs for GPU solve, 3-5 for CPU solve, 5 for wall clock.

| instance                     | N  | permutations | minimum cost | threads | GPU solve | CPU solve | speedup | GPU wall |
|------------------------------|----|--------------|--------------|---------|-----------|-----------|---------|----------|
| `fldata-9`   / `dstdata-9`   |  3 |            6 |          153 |      32 | 0.000097s | 0.000001s |       - |  0.129 s |
| `fldata-16`  / `dstdata-16`  |  4 |           24 |          313 |      32 | 0.000140s | 0.000002s |       - |  0.128 s |
| `fldata-25`  / `dstdata-25`  |  5 |          120 |         3654 |     128 | 0.000164s | 0.000004s |       - |  0.129 s |
| `fldata-64`  / `dstdata-64`  |  8 |        40320 |       111640 |   40704 | 0.000241s | 0.001578s |      7x |  0.129 s |
| `fldata-81`  / `dstdata-81`  |  9 |       362880 |       138868 |   89088 | 0.000301s | 0.017856s |     59x |  0.127 s |
| `fldata-100` / `dstdata-100` | 10 |      3628800 |       165908 |   89088 | 0.000810s | 0.140467s |    173x |  0.134 s |
| `fldata-144` / `dstdata-144` | 12 |    479001600 |       953450 |   89088 |  0.10415s |  23.2684s |    223x |  0.274 s |
| `fldata-12a` / `dstdata-12a` | 12 |    479001600 |         9552 |   89088 |  0.10513s |  23.2753s |    221x |  0.274 s |
| `chr12a` (`-s`)              | 12 |    479001600 |         9552 |   89088 |  0.10466s |  23.2784s |    222x |  0.280 s |

The minimizing assignments for the six pair-format instances above are, in
order: `{2,0,1}`, `{3,0,1,2}`, `{4,3,2,1,0}`, `{6,7,5,4,3,2,0,1}`,
`{8,7,6,5,4,2,3,1,0}`, `{9,8,7,6,5,3,4,2,1,0}`.

The GPU loses on the smallest instances because a kernel launch costs more than
the entire search does, and it does not repay its fixed ~0.13 s of process
startup until around N=10 in wall-clock terms. On solve time it climbs from 173x
at N=10 to about 220x at N=12. Recording the minimizing assignment rather than
just the cost is free at this scale.

`fldata-12a`/`dstdata-12a` is the same instance as `chr12a` in pair format, and
both forms agree at 9552 - that cross-check exercises the two input parsers
against each other.

Launch geometry
---------------

The thread count is not fixed. `cudaqap` sizes the launch from the device and
the problem: it walks candidate block widths, asks
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` how many of each would be
resident, and takes the width that keeps the most threads on an SM - preferring
the wider block on a tie, since that means fewer blocks and fewer global atomics
in the final reduction. A width reporting zero resident blocks cannot launch at
all and is never chosen. The grid is then one full wave of that block, capped by
the number of threads there is actually work for. On this GPU that resolves to:

| N  | stride | block | grid | threads | dynamic shmem | blocks/SM |
|----|--------|-------|------|---------|---------------|-----------|
|  3 |      3 |    32 |    1 |      32 |         384 B |        24 |
|  4 |      5 |    32 |    1 |      32 |         640 B |        24 |
|  5 |      5 |   128 |    1 |     128 |        2560 B |        12 |
|  8 |      9 |   768 |   53 |   40704 |       27648 B |         2 |
|  9 |      9 |   768 |  116 |   89088 |       27648 B |         2 |
| 10 |     11 |   768 |  116 |   89088 |       33792 B |         2 |
| 12 |     13 |   768 |  116 |   89088 |       39936 B |         2 |
| 15 |     15 |   768 |  116 |   89088 |       46080 B |         2 |
| 16 |     17 |   480 |  174 |   83520 |       32640 B |         3 |
| 20 |     21 |   384 |  174 |   66816 |       32256 B |         3 |

89088 is 58 SMs x 1536 threads, i.e. full occupancy.

Each thread owns a slice of shared memory at `&MCD[TIB * (N | 1)]`. The stride
is rounded up to odd deliberately: shared-memory banks repeat every 32 words, so
consecutive threads collide with multiplicity `gcd(stride, 32)`. An `N + 1`
stride is odd only when N is even; at N=15 it gave stride 16 and a 16-way
conflict, which made that instance take over 17 minutes instead of 7:38. Making
the stride odd also shrinks the slice at odd N, which buys back occupancy -
N=15 moved from 512-thread blocks to 768, and N=17/N=19 from 77952/66816
threads to 83520/74240.

Sizing the launch means asking the driver about the device, and that has to be
done carefully. `cudaGetDeviceProperties` fills an entire `cudaDeviceProp` and
costs around 700 microseconds on this machine - several times the whole kernel
at small N. Host work immediately before a launch inflates the measured time of
a short kernel, so that one call alone made N=8 and N=9 measurably slower than
a fixed geometry. Querying just the five attributes the sizing needs, via
`cudaDeviceGetAttribute`, costs under a microsecond and removes the effect.

`NumThreads:` reports the launch size chosen for that problem. Each thread takes
a contiguous range of lexicographic permutation ranks; when there are fewer
permutations than threads the surplus threads correctly take an empty range. The
winning thread is identified by a packed `(cost << 32) | threadId` argmin key,
and its best rank is unranked back into a permutation on the host - so only 8
bytes per thread cross the reduction.

Reach
-----

Brute force is `O(N! * N^2)`, so the practical ceiling is low and moves by less
than one problem size per order of magnitude of hardware:

- **N <= 12** finishes in about a tenth of a second of solve time.
- **13 <= N <= 20** is accepted and starts normally, but past about N=16 it
  will not finish in any useful time:

  | N  | permutations | threads | solve time | source    |
  |----|--------------|---------|------------|-----------|
  | 13 |      6.2e+09 |   89088 |     1.66 s | measured  |
  | 14 |      8.7e+10 |   89088 |    28.41 s | measured  |
  | 15 |      1.3e+12 |   89088 |     7.6 min | measured  |
  | 16 |      2.1e+13 |   83520 |     2.4 hrs | measured  |
  | 17 |      3.6e+14 |   83520 |    45 hrs   | projected |
  | 18 |      6.4e+15 |   74240 |    43 days  | projected |
  | 19 |      1.2e+17 |   74240 |     2 years | projected |
  | 20 |      2.4e+18 |   66816 |    61 years | projected |

  The four measured rows are recorded in full under **Large runs** below. The
  remaining rows extrapolate from N=16 as `N! * N^2`, scaled by the drop in
  thread count shown in the table:
  shared memory per thread grows with N, so the launch falls away from full
  occupancy from N=16 onward.

  A plain `N^2` cost model holds up well. Normalising for thread count, the
  constant `k = time * threads / (N! * N^2 * 89088)` measures 1.52, 1.58, 1.66,
  1.56 and 1.49 e-12 at N=12..16 - flat to within about 5% either side, with no
  trend. Taking the median, 1.56e-12, the model reproduces all five anchors to
  within 7%: N=12 +2.6%, N=13 -1.1%, N=14 -6.3%, N=15 +0.0%, N=16 +4.6%. It
  also predicted N=16 at 2.5 hours before that run started, against 2.4 hours
  measured, which is the only out-of-sample test the table has had.

  Earlier drafts claimed the effective exponent was climbing towards `N^2.5` or
  `N^2.9`. That was an artifact of a shared-memory bank conflict inflating N=13
  plus a two-run N=14 sample; neither survived remeasurement.

  Extrapolating four factorial steps from one anchor is still worth only an
  order of magnitude. `chr20a` is N=20, so it sits at the bottom of the table:
  starting it looks like a hang but is not, and nothing guards against it.
- **N > 20** is rejected up front, because ranking the permutation space
  requires `N!` to fit in a `uint64_t`:

```
%>> ./cudaqap -s ./chr25a.dat
Problem size 25 is too large: the permutation count does not fit in 64 bits (the limit is 20).
```

  `cgbncudaqap` carries this to `N <= 24`; see **cgbncudaqap** below.

The same check turns away the three largest sample pairs. `fldata-10000` and
`fldata-65536` are 100x100 and 256x256, and `fldata-1000` is 1000x1000 - all
square and well formed, all far past the limit, so `cudaqap` reports the size
and exits rather than starting a search that could never finish.

Large runs
----------

| N  | instance | solver        |        permutations | threads | minimum cost | solve time | verified          |
|----|----------|---------------|---------------------|---------|--------------|------------|-------------------|
| 13 | A        | `cudaqap`     |       6,227,020,800 |   89088 |       440532 |     1.66 s | cost + assignment |
| 14 | A        | `cudaqap`     |      87,178,291,200 |   89088 |       423242 |    28.41 s | cost + assignment |
| 15 | A        | `cudaqap`     |   1,307,674,368,000 |   89088 |       521907 |   458.27 s | iteration count   |
| 15 | B        | `cudaqap`     |   1,307,674,368,000 |   89088 |       362694 |   456.92 s | cost + assignment |
| 15 | B        | `cgbncudaqap` |   1,307,674,368,000 |   89088 |       362694 |   457.42 s | cost + assignment |
| 16 | A        | `cudaqap`     |  20,922,789,888,000 |   83520 |       552418 |  8511.36 s | iteration count   |

These are the large runs behind the projections. Each ran on a random square
instance generated for the purpose (they are not in the repo), and each iterated
exactly `N!` times, which is what confirms the rank ranges tiled the whole
permutation space with no gaps or overlap.

The `instance` column matters at N=15, which has been run twice on two different
inputs - hence the two different minimum costs. Instance B is the later pair,
put through both GPU solvers with `-p`, and it is the stronger record: both
returned the same assignment,
`{ 14, 13, 12, 11, 10, 7, 9, 8, 6, 5, 4, 3, 2, 1, 0 }`, and re-scoring that
permutation against the raw matrices reproduces 362694. The two solve times,
456.92 s and 457.42 s, are also the tightest speed comparison in this document.

The `verified` column is deliberate. Where it says `iteration count`, that is
all that was checked: the assignment is printed at the end of the run with `-p`,
so obtaining it after the fact means running the search again. That is what
instance B bought at N=15, for two more 7.6-minute runs. N=16 would cost 2.4
hours a side and remains unverified on assignment.

cgbncudaqap
-----------

`cgbncudaqap.cu` is a variant of `cudaqap.cu`, not a rewrite. The search kernel,
the launch-geometry selection, the odd shared-memory stride, the packed argmin
key and the host-side unranking of the winner are all unchanged. Three things
differ.

### What changed

**The iteration count is now per-thread.** `cudaqap` did a single
`atomicAdd(&IT, Hi - Lo)` per thread into one device-wide `uint64_t`.
`cgbncudaqap` gives each thread a private `uint64_t LIT`, increments it once per
permutation actually scored, and stores it to `GITC[GTID]` when the thread
finishes. Counting what the loop did rather than what it was asked to do is a
slightly stronger check, and it is what the arbitrary-precision total is built
from.

**The global count is a CGBN accumulator.** `N!` passes `2^64` at `N = 21`:
`21!` is 51090942171709440000 against a `uint64_t` ceiling of
18446744073709551615, so there is no 64-bit value that could hold the answer.
The total is instead summed into a 1024-bit `cgbn_mem_t` by two kernels:

- `AccumulateIterations` runs 512 CGBN instances (64 blocks x 256 threads, one
  warp per instance at `TPI = 32`). Each instance sums a strided share of
  `GITC`. Striding rather than blocking avoids a division and keeps every
  instance busy; integer addition is exact and associative, so visiting order
  does not affect the result.
- `CombineIterations` runs a single instance that folds the 512 partials into
  one grand total.

There is no lock and no host arithmetic in the reduction. `blockDim.x` must be a
whole number of instances, because CGBN lays an instance's lanes out along
`threadIdx.x` and derives its warp sync mask from that.

**Permutation ranks are `__uint128_t`.** This, not the counter, is what lifts
the size cap, and the reason it is not also CGBN is worth stating plainly: a
CGBN value is not a scalar. Its limbs live across TPI cooperating lanes, so a
single thread can neither hold one nor update one atomically - which rules CGBN
out for anything per-thread, and a permutation rank is the most per-thread
quantity in the program. The widest type one thread can own is `__uint128_t`,
and `34!` (2.95e38) is the largest factorial under `2^128`.

The division of labor follows from that constraint rather than from taste:
per-thread state gets the widest scalar available, and the one genuinely global
quantity gets CGBN.

### Limits

Two, both checked on the host before the launch:

```
%>> ./cgbncudaqap -f ./fldata-10000.dat -d ./dstdata-10000.dat
Problem size 100 is too large: the permutation rank does not fit in 128 bits (the limit is 34).

%>> ./cgbncudaqap -s ./chr25a.dat
Problem size 25 gives 278577766582812248275 permutations per thread, which does not fit the 64-bit per-thread iteration counter.
```

The second is the binding one. `N! / threads` has to fit the local `uint64_t`,
and on this GPU that admits `N = 24` at 11143110663312489931 permutations per
thread - just inside the limit - and rejects `N = 25`. The cap is therefore
device-dependent: it moves with the launch's thread count, which is itself
derived from occupancy. Widening `LIT` to `__uint128_t` would take the cap to
the rank ceiling of 34, but see the reach table below for why that would be
decoration.

Summarised: `cudaqap` stops at 20, `cgbncudaqap` at 24 in practice and 34 by
construction, and the CGBN accumulator itself would not run out until `170!`.

### Cost

The search is not measurably slower. Same instances, same geometry; medians of
7 runs at N=12, 5 at N=13, single runs at N=14 and N=15:

| instance     | N  | `cudaqap`  solve  | `cgbncudaqap` solve | delta  | CGBN reduction |
|--------------|----|-------------------|---------------------|--------|----------------|
| `chr12a`     | 12 |         0.10494 s |           0.10474 s |  -0.2% |      0.00011 s |
| `fldata-144` | 12 |         0.10508 s |           0.10577 s |  +0.7% |      0.00012 s |
| generated    | 13 |         1.63598 s |           1.64306 s |  +0.4% |      0.00021 s |
| generated    | 14 |        28.42631 s |           28.0748 s |  -1.2% |      0.00021 s |
| generated    | 15 |       456.91564 s |          457.4166 s | +0.11% |      0.00021 s |

N=15 is the tightest of these: 7.6 minutes of kernel apiece and a 0.11% difference, or half a second across 1307674368000 permutations. The deltas have no consistent sign and the largest is 1.2%, on one of the single-run rows; they are run-to-run noise. The 128-bit loop counter costs a handful of instructions against an `O(N^2)` cost evaluation, and `Chunk` and `Rem` are computed on the host and passed in as arguments, so no thread pays for a 128-bit division to find its own range. The reduction is about 0.2 ms and does not scale with `N` - it scales with the thread count, which is bounded by the device.

Two output lines are new:

```
%>> ./cgbncudaqap -s ./chr12a.dat
Minimum cost: 9552
Iterations:   479001600
Permutations: 479001600
NumThreads:   89088
CPU Clock Resolution: 0.000000001.
GPU time: 0.104996840 second(s).
CPU Clock Resolution: 0.000000001.
CGBN reduction: 0.000101231 second(s).
```

`Permutations:` is `N!` computed on the host. Printing it beside `Iterations:`
puts the check that matters - did the rank ranges tile the space exactly once -
in the output itself. `CGBN reduction:` is timed separately from the search
because it is the price of the arbitrary-precision counter, and the variant is
only interesting if that price is visible.

### Reach, with the counter no longer in the way

Projected as `k * N! * N^2 / threads` with `k = 1.38e-7` fitted to the five
measured anchors (N=12..16, spread +6.3%/-4.9%), and the thread count taken from
the geometry the program actually selects at each size:

| N  |                    iterations = N! | digits | 64-bit? | threads | time            |
|----|------------------------------------|--------|---------|---------|-----------------|
| 15 |                      1307674368000 |     13 | yes     |   89088 | 7.6 min (measured) |
| 16 |                     20922789888000 |     14 | yes     |   83520 | 2.4 hrs (measured) |
| 17 |                    355687428096000 |     15 | yes     |   83520 | 2.0 days        |
| 18 |                   6402373705728000 |     16 | yes     |   74240 | 45 days         |
| 19 |                 121645100408832000 |     18 | yes     |   74240 | 2.6 years       |
| 20 |                2432902008176640000 |     19 | yes     |   66816 | 64 years        |
| 21 |               51090942171709440000 |     20 | **no**  |   66816 | 1.5 thousand years |
| 22 |             1124000727777607680000 |     22 | **no**  |   61248 | 39 thousand years |
| 23 |            25852016738884976640000 |     23 | **no**  |   61248 | 970 thousand years |
| 24 |           620448401733239439360000 |     24 | **no**  |   55680 | 28 million years |

The four rows `cudaqap` cannot express at all are the four that need more than
64 bits to state their own iteration count. `cgbncudaqap` states them exactly.
It does not make them runnable: `N = 21` is about 1500 years on this GPU, and
`N = 24` is 28 million. What the variant removes is an *arithmetic* limit that
sat below the physical one - after this change nothing in the program stops
before the hardware does.

### Verification

`cgbncudaqap` was checked the same three ways as `cudaqap` - cost, iteration
count, assignment - against both `qap` and `cudaqap`:

- All three agree on cost and assignment at N = 1, 2, 3, 4, 5, 8, 9, 10, 12, 13,
  14, and `cudaqap` and `cgbncudaqap` also agree at N = 15. `Iterations` equals
  `Permutations` equals `N!` exactly at every size.
- `chr12a` reproduces QAPLIB's 9552 with the assignment
  `{ 6, 4, 11, 1, 0, 2, 8, 10, 9, 5, 7, 3 }`, matching `cudaqap`.
- The N=12, N=13, N=14, N=15 and `chr12a` assignments were re-scored
  independently against the raw matrices and reproduced their reported costs.
- Clean under `compute-sanitizer` at N=10, and re-run at N=13 - a size where
  every thread walks a long private loop rather than a handful of permutations,
  so the shared-memory slices and the reduction are under sustained load:

  | tool         | N  | GPU time    | vs native | verdict                        |
  |--------------|----|-------------|-----------|--------------------------------|
  | (none)       | 13 |    1.643 s  |      1.0x | -                              |
  | `synccheck`  | 13 |    1.664 s  |      1.0x | 0 errors                       |
  | `initcheck`  | 13 |  332.943 s  |      203x | 0 errors                       |
  | `memcheck`   | 13 |  414.840 s  |      253x | 0 errors                       |
  | `racecheck`  | 13 |           - |    >6400x | killed after 2h55m, unfinished |
  | (none)       | 10 |  0.000957 s |      1.0x | -                              |
  | `racecheck`  | 10 |    7.258 s  |     7584x | 0 hazards, 0 errors, 0 warnings |

  Every run that completed returned the same cost, iteration count and assignment
  as its uninstrumented counterpart: 368897 / 6227020800 (exactly 13!) /
  `{ 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }` at N=13, and 165908 / 3628800
  (exactly 10!) / `{ 9, 8, 7, 6, 5, 3, 4, 2, 1, 0 }` at N=10. `synccheck` is
  effectively free; `initcheck` and `memcheck` cost about 200-250x, the same
  ratios measured at N=12 (209x and 263x).

  `racecheck` is in a different class at roughly 7600x, and it is run at N=10 by
  choice rather than by concession. A larger N adds it no coverage: the race
  surface - the per-thread `MCD` slices, the shared `BKEY`, and the block and
  global `atomicMin`s - is identical at every problem size, and N only changes
  how many times each thread goes round its own private loop. The N=13 attempt
  bears that out from the other direction: it was killed after two hours and
  fifty-five minutes without finishing, against 7.6 seconds for the N=10 run
  that reports the same thing.

The two things that cannot be reached by running the solver were tested
directly, since no `N >= 21` search will ever terminate:

- **The 128-bit ranking.** The `qap` namespace was extracted verbatim from
  `cgbncudaqap.cu` into a harness that unranks a given rank on the device and
  steps it forward, and its output was compared against an independent
  arbitrary-precision reference. 224 ranks across N=21..34 - including 0, 1,
  `N!-1`, `N!-2`, `N!/2` and random ranks - matched exactly.
- **The CGBN reduction past 2^64.** `AccumulateIterations`, `CombineIterations`
  and `BigToString` were extracted verbatim and fed counts summing to as much as
  25 decimal digits, including the instance-count boundaries (511/512/513
  threads) and 100000 threads. All exact.

Environment
-----------

```
GPU     NVIDIA GeForce RTX 4080 Laptop GPU, sm_89, 12282 MiB, driver 575.64.03
CPU     Intel(R) Core(TM) i9-14900HX
nvcc    release 12.9, V12.9.86   (Makefile.cuda: CUDA_VERSION = 12.9, CUDA_CC = 89)
g++     14.3.1 20250523 (Red Hat 14.3.1-1)
```

`cudaqap` is verified clean under `compute-sanitizer` for `memcheck`,
`racecheck`, `synccheck`, and `initcheck`. `cgbncudaqap` is clean under all four:
`synccheck`, `initcheck` and `memcheck` at N=13, and `racecheck` at N=10, where
it reports 0 hazards, 0 errors and 0 warnings. See the table in
**cgbncudaqap > Verification** for the per-tool costs and for why `racecheck` is
run at N=10.

