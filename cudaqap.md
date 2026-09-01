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
covered in its own section; everything between here and there describes
`cudaqap`, and `cgbncudaqap` matches it measurement for measurement except
where that section says otherwise.

`cgbncudaqapbb` is a fourth solver and the only one that does not enumerate.
It searches the same space with the Gilmore-Lawler branch and bound of
`qapbb.c`, so `N!` is no longer its runtime and no longer its correctness
check; both are replaced, and the last section covers what by.

Build and run:

```
%>> make -f Makefile.cuda
%>> ./cudaqap -f ./fldata-144.dat -d ./dstdata-144.dat
%>> ./cudaqap -s ./chr12a.dat
%>> ./cgbncudaqap -s ./chr12a.dat
%>> ./cgbncudaqapbb -s ./chr25a.dat
%>> cc -O2 -fwrapv -o qapbb qapbb.c && ./qapbb < ./chr25a.dat
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

The division of labour follows from that constraint rather than from taste:
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

| instance     | N  | `cudaqap` solve | `cgbncudaqap` solve | delta  | CGBN reduction |
|--------------|----|-----------------|---------------------|--------|----------------|
| `chr12a`     | 12 |       0.10494 s |           0.10474 s |  -0.2% |      0.00011 s |
| `fldata-144` | 12 |       0.10508 s |           0.10577 s |  +0.7% |      0.00012 s |
| generated    | 13 |        1.63598 s |            1.64306 s |  +0.4% |      0.00021 s |
| generated    | 14 |       28.4263 s |           28.0748 s |  -1.2% |      0.00021 s |
| generated    | 15 |      456.9156 s |          457.4166 s | +0.11% |      0.00021 s |

N=15 is the tightest of these: 7.6 minutes of kernel apiece and a 0.11%
difference, or half a second across 1307674368000 permutations. The deltas have
no consistent sign and the largest is 1.2%, on one of the single-run rows; they
are run-to-run noise. The 128-bit loop counter costs a handful of instructions
against an `O(N^2)` cost evaluation, and
`Chunk` and `Rem` are computed on the host and passed in as arguments, so no
thread pays for a 128-bit division to find its own range. The reduction is about
0.2 ms and does not scale with `N` - it scales with the thread count, which is
bounded by the device.

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

cgbncudaqapbb
-------------

`cgbncudaqapbb.cu` keeps `cgbncudaqap.cu`'s scaffolding - option parsing, the
input readers, `Verify()`, the launch-geometry search, the packed argmin key,
128-bit ranks, the CGBN reduction kernels, `BigToString` - and replaces the
search. The bound is `qapbb.c`'s, ported function for function into
`namespace qapbb` with the FORTRAN label numbers and 1-based arrays intact.

### The tree had to change

`qapbb` branches on the single assignment `(i,j)` with maximal alternative
cost: one child assigns it, the other forbids it. That is a strong rule on one
core and unusable on a GPU, because a node's identity is then a set of
forbidden cells. There is no cheap index for it, so there is no cheap way to
say "thread 8000 takes that subtree".

`cgbncudaqapbb` places the facilities in a fixed order instead - one per level,
in an order chosen at the root by decreasing total flow - so a node at level
`L` *is* an injective `L`-tuple of locations, with an ordinary mixed-radix
rank. A thread can be handed one 128-bit number and rebuild the node from it.
`ALTKOS` and the `ZUL`/`IKAP` bookkeeping that served the old rule are gone,
and `PROGNO`'s `CSPEI` cache went with them: the n-ary tree never revisits a
level's matrix, so the cache had nothing to give back. That also removed
`N(N+1)(2N+1)/6` words per thread, 11 KB at `N = 20`.

The root relabelling is not cosmetic. Index order is branching order, so
placing the most heavily connected facility first tightens the bound early.
Measured on a single-threaded host build of this same search, where the node
count is not perturbed by the incumbent race: `chr20a` fell from 32462 nodes
to 11859 and `chr25a` from 17.8 million to 827 thousand when it was added.

### Growing the frontier instead of splitting the tree

The first working version cut the tree at a fixed depth, indexed the nodes at
that depth, and let threads claim ranges of them from a work queue. It was
correct and it was slower than one CPU core. The reason is in its own output:

```
N = 25, fixed depth-5 split
B&B nodes:    8675014
Busiest:      257352 node(s)
GPU time:     61.960711364 second(s).
```

8.68 million nodes across the whole launch, and one thread held 257352 of
them. Every other thread finished in the first second. Under a pruning bound
the subtree sizes span five orders of magnitude, and no fixed depth fixes that: a
finer cut makes the queue longer without making the largest piece smaller,
because the piece that matters is deep inside one branch.

So the frontier is grown rather than guessed. `ExpandFrontier` takes the live
nodes at level `L`, rebuilds each one's state from its rank, bounds all of its
children in parallel, accounts for the dead ones and appends the survivors to
the next level's buffer. The host repeats that until the frontier holds `32`
live nodes per thread, or until the next level would not fit, or until the
128-bit rank would overflow. `BranchAndBound` then hands the frontier out one
subtree per task. Small instances never leave the breadth-first phase at all -
the `Frontier:` line says `- solved breadth-first` when the whole search
finished there.

Same instance, same machine, after:

```
B&B nodes:    8596072
Busiest:      795 node(s), 0 idle thread(s)
GPU time:     2.735117544 second(s).
```

Rebuilding a node's state from its rank costs `O(L*N^2)` against a node's own
`O(N^3)`, which is what buys the expansion its freedom: no frontier node
depends on any other, so there is no synchronisation inside a level.

### The free incumbent

The LSAP that produces a node's bound also produces a complete, feasible
assignment of every facility still free to a location still free. Scoring it
costs `O(m^2)` against the bound's `O(m^3)` and needs nothing the workspace
does not already hold, so `TryCompletion` scores it and offers it to the
incumbent.

This matters more on a GPU than it would on a CPU. A sequential search reaches
a leaf almost immediately and prunes with a real solution from then on. Tens of
thousands of threads starting at once have nothing but the host's 2-opt bound
until one of them gets to the bottom, and everything they do in the meantime is
speculative. On `chr25a`, where the host heuristic comes in at 5062 against an
optimum of 3796, adding it took the node count from 19.7 million to 6.0
million.

That figure is for running it on every live node. The shipped version runs it
on every eighth: unthrottled it cost 28% on `fldata-400`, whose starting
heuristic was already optimal and had nothing to gain. Sampled, `chr25a`
settles at about 7.9 million - most of the benefit for a few percent, because
with fifteen thousand threads the incumbent still improves in bursts.

### N! is still the check, differently

There is no `Iterations: == N!` line any more, because a pruning search does
not visit `N!` of anything. The replacement is exact and stronger, in that it
constrains the pruning as well as the enumeration.

Every node the search disposes of - pruned by the bound, or evaluated as the
closed-form pair of last-two-facility completions - stands for exactly `(N-L)!`
complete permutations, where `L` is its level. A node that is *not* disposed of
passes its `(N-L)!` down to its `N-L` children, each worth `(N-L-1)!`, which
sum to the same thing. So the disposal counts, weighted by factorials, must
come to precisely `N!`.

Each thread keeps one `uint64_t` count per level. `SumCounters` folds those
into one count per level, and `CombinePermutations` - a single CGBN instance -
forms `sum count_L * (N-L)!` in 1024-bit arithmetic and subtracts it from `N!`:

```
Permutations: 265252859812191058636308480000000
Accounted:    265252859812191058636308480000000
Residual:     0
```

The counts stay 64-bit and only the products are wide, which is what makes the
per-thread side of it free. `cgbn_mul` returns the low half of the product,
which is the whole product here: a count below `2^64` times a factorial below
`2^1024` could only need more than 1024 bits at sizes whose workspace could
never be allocated.

The residual is formed as a **signed** value, using `cgbn_signed_is_negative`
and `cgbn_signed_abs` from this CGBN fork's `cgbn_signed.h`. Unsigned, a search
that covered something twice would wrap `N! - accounted` into an enormous
positive number and read as a wildly wrong under-count; signed, the sign says
which way the error went. This is not hypothetical - see below.

### What the residual did and did not catch

Two real bugs got through it during development, and between them they are the
argument for reading all three of the cost, the residual and the node count.

**The 128-bit rank.** The frontier initially held 64-bit ranks. A rank at level
`L` runs to `N!/(N-L)!`, which at `N = 25` passes `2^64` at depth 16; a wrapped
rank decodes to a different node, so the search covered some subtrees twice and
missed others. It returned 3856 for `chr25a` instead of 3796 - and it reported
`Residual: 0` while doing it, because the accounting faithfully counts what the
search disposed of, not what it should have. What caught this was the QAPLIB
optimum, not the residual. The frontier now holds `__uint128_t`, and the host
stops deepening one level before `N!/(N-L)!` would leave 128 bits.

**The missing root tables.** Both kernels must call `wegspe` once at the root
to build the level-0 sorted rows that the Gilmore-Lawler scalar products are
read from. An early version of the rewrite dropped that call. The bound then
fell back to its linear part - still a valid lower bound, so still the right
answer, and still `Residual: 0` - and simply stopped pruning: `fldata-400` went
from 101 thousand nodes to 88 million. Nothing but the node count showed it.

The lesson both times is that the reported cost and the residual are necessary
and not sufficient. The node count is the third instrument, and it is why it is
printed.

### Measurements

Against `qapbb` on the same machine. `qapbb` times are whole-run; the GPU
column is solve time as reported. Medians of 5 for the fast instances.

| instance             | N  | minimum cost | GPU nodes  | `qapbb` | `cgbncudaqapbb` | speedup |
|----------------------|----|--------------|------------|---------|-----------------|---------|
| `chr12a`             | 12 |         9552 |       1479 | 0.0016s |         0.0267s |   0.06x |
| `fldata-144` pair    | 12 |       953450 |       3567 | 0.0016s |         0.0319s |   0.05x |
| `fldata-225` pair    | 15 |      2135449 |        902 | 0.0028s |         0.0622s |   0.05x |
| `fldata-256` pair    | 16 |      2942118 |      17162 | 0.0292s |         0.1068s |   0.27x |
| `chr20a`             | 20 |         2192 |     158236 | 0.0548s |         0.1677s |   0.33x |
| `fldata-400` pair    | 20 |      9779382 |     100978 | 0.164s  |         0.2887s |   0.57x |
| `chr25a`             | 25 |         3796 |     ~7.9e6 | 11.2s   |         2.46s   |    4.6x |
| `fldata-625` pair    | 25 |     41048904 |    8596072 | 71.8s   |         2.74s   |     26x |
| `fldata-900` pair    | 30 |    119265331 |   33496281 | 2248s   |        19.99s   |    112x |
| `fldata-1024` pair   | 32 |    189919789 | 1484193064 | -       |      1275.31s   |       - |

The crossover is around `N = 20`. Below it the tree is too small to spread over
15000 threads and the GPU is paying setup costs against a search one core
finishes in milliseconds; above it the node count grows, and the node count is
the thing that parallelises. `qapbb` at `N = 30` took 2248 seconds - thirty-seven
minutes - and returned an assignment that re-scores to 119265331, the same
optimum the GPU reached in twenty seconds. It was started on `N = 32` as well
and stopped, unfinished, after two hours - as was a single-threaded host build
of the GPU's own algorithm, at one hour forty against the GPU's twenty-one
minutes. That is why the `N = 32` row has no CPU figure: `N = 30` is the last
size where both were run to completion, and `N = 32` rests on the program's own
checks instead.

`N = 32` is worth its own look, because it is the only instance measured here
that uses the whole design. Every smaller one is finished by the breadth-first
phase before the frontier reaches its target; this one stops the expansion at
depth 7 with 665914 live nodes and hands them to the depth-first phase:

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

Three things in that output are the design working. 1.48 billion nodes in 21
minutes is 1.16 million bounds a second, each one an LSAP. `Busiest:` is
2366935 against 1484193064, so the thread that drew the largest of the 665914
subtrees carried 0.16% of the search - the tail that made the first version of
this program useless is gone. And 32! is
263130836933693530167218012160000000, an exact 118-bit answer arrived at by
summing 64-bit counts: fifty-four bits more than any counter a thread could
have carried. It is still inside 128 bits - 34! is where that runs out - so
this size does not yet need the full 1024, but it is well past the point where
`cudaqap`'s and `cgbncudaqap`'s `Iterations:` line could have been written down
at all.

The two do not explore the same tree, which the table cannot show. A cleaner
parallel-speedup figure comes from a host build of `cgbncudaqapbb`'s own
algorithm, single-threaded: `fldata-625` 54.0 s against 2.74 s, and
`fldata-900` 420.7 s against 19.99 s - 20x and 21x, on the same tree, the same
bound and very nearly the same node counts (8596072 against 8596159 at N=25;
33496281 against 33496339 at N=30).

`NumThreads` is 14848 at every size measured, which is what the occupancy API
gives for this kernel - register pressure, not shared memory, since there is
none. The second cap, three fifths of free device memory divided by the
per-thread workspace, does not bind below about `N = 50`: the sorted-row
tables are `O(N^3)` words, 48 KB per thread at `N = 25` but 618 KB at
`N = 60`. `-t` lowers the count by hand; it is worth reaching for only to
reproduce a measurement, since fewer threads is strictly less parallelism.

### Determinism

The optimum, `Accounted:` and `Residual:` are reproducible. The node count is
not, and cannot be: every thread prunes against one shared incumbent, so how
much speculative work gets done before someone lowers it depends on timing.
`chr25a` measured 7915417, 7939937 and 10001210 nodes on three consecutive
runs, all returning 3796 with a residual of zero. Instances whose starting
heuristic is already optimal are reproducible, because the incumbent then never
moves: `fldata-625` gave 8596072 nodes on all three runs.

Ties may also resolve differently from the three brute-force solvers. Pruning
is `bound >= incumbent`, so a second permutation of equal cost is discarded
rather than compared, and there is no equivalent of "lexicographically first
among ties". In practice the assignment matched `qap` exactly on all eight
small pair-format inputs tested, but that is a property of those inputs.

### Verification

Costs match `qapbb` on every instance tested and QAPLIB on all three `chr`
instances (9552 / 2192 / 3796). Assignments match `qap` on `fldata-9`, `-16`,
`-25`, `-64`, `-81`, `-100`, `-144` and `-12a`. `Residual:` is 0 everywhere.
The program re-scores its own reported assignment against the untouched input
matrices and complains on stderr if that disagrees with the search's figure,
since the search works in reduced arithmetic; it never has.

A correct run always has a residual of zero, so the signed arithmetic is a path
the solver cannot reach. It was validated by extraction, as `cgbncudaqap`'s
`N >= 21` paths were: `CombinePermutations`, `CgbnSetUi64` and `BigToString`
were pulled verbatim into a scratch harness and driven with hand-built level
counts at `N = 25`.

| level counts                        | accounted                  | residual                  |
|-------------------------------------|----------------------------|---------------------------|
| 25 nodes at level 1 (exact)          | 15511210043330985984000000 | 0                         |
| one level-1 node lost                | 14890761641597746544640000 | 620448401733239439360000  |
| one extra level-2 node               | 15537062060069870960640000 | -25852016738884976640000  |
| one extra level-1 node               | 16131658445064225423360000 | -620448401733239439360000 |

The exact case is `25 * 24! == 25!`. The three errors come back as `+24!`,
`-23!` and `-24!` respectively - the right magnitude and, for the two
over-counts, the right sign. Repeat this if the accounting or the reduction is
touched.

`compute-sanitizer` is far cheaper here than on `cgbncudaqap`, and cheap enough
that all four tools can be run on the same instance without planning around
them:

| tool         | N  | GPU time   | vs native | verdict                          |
|--------------|----|------------|-----------|----------------------------------|
| (none)       | 12 |   0.0316 s |      1.0x | -                                |
| `synccheck`  | 12 |   0.0333 s |      1.1x | 0 errors                         |
| `initcheck`  | 12 |   0.4367 s |       14x | 0 errors                         |
| `memcheck`   | 12 |   0.6704 s |       21x | 0 errors                         |
| (none)       | 10 |   0.0145 s |      1.0x | -                                |
| `racecheck`  | 10 |   0.2322 s |       16x | 0 hazards, 0 errors, 0 warnings  |

All four returned the same cost, node count and residual as the uninstrumented
run: 953450 in 3567 nodes at `N = 12`, 165908 in 104 nodes at `N = 10`, residual
zero throughout. The costs are worth comparing against `cgbncudaqap`, where
`initcheck` runs 203x and `memcheck` 253x: those instrument every global access,
and the enumerating kernel makes `O(N^2)` of them per permutation over `N!`
permutations. `racecheck` is the sharper contrast - 16x here against roughly
7600x there - and that one is the shared memory. `cgbncudaqap` gives every
thread an `MCD` slice to instrument; this kernel has no shared memory at all,
so all `racecheck` has to watch is the argmin `atomicMin` and the frontier's
append counter.

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
run at N=10. `cgbncudaqapbb` is clean under all four as well, at a small
fraction of the cost; see **cgbncudaqapbb > Verification**.
