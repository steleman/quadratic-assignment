// Copyright (c) 2025-2026 Stefan Teleman.
//
// Licensed under the MIT License.
// See https://opensource.org/license/mit
// SPDX-License-Identifier: MIT
//
// cgbncudaqapbb -- branch and bound on the GPU.
//
// This is cgbncudaqap.cu with the search replaced. The other three solvers in
// this directory enumerate all N! permutations; this one prunes, using the
// Gilmore-Lawler bound machinery of qapbb.c (the C translation of the
// Boenniger / Burkard / Stratmann FORTRAN QAP, with U. Derigs' LSAP). What
// carries over from cgbncudaqap.cu is the shape of the program: the option
// parsing, the input readers, the launch-geometry search, the packed argmin
// key, the 128-bit ranks and the CGBN accumulators.
//
// Four things about the port are worth stating up front.
//
// 1. The branching rule changed, deliberately. qapbb branches on a single
//    pair (i,j) chosen by maximal alternative cost, which makes each node's
//    two children "assign (i,j)" and "forbid (i,j)". That tree is excellent
//    on one CPU and useless to partition across 10^4 threads: a node's
//    identity is a set of forbidden cells, not a prefix, so there is no cheap
//    index for "the k-th subtree". Here the DFS instead places the facilities
//    in a root-chosen order, one per level, so a node at level L IS an
//    injective L-tuple of locations -- a partial permutation with an ordinary
//    mixed-radix rank, which is the whole of what a thread needs to be handed
//    to rebuild the node and search it. ALTKOS and the ZUL/IKAP bookkeeping
//    that served the old rule are gone; the bound itself (LSAP, WEGSPE,
//    PROGNO, and the root reduction) is unchanged, and so is the closed-form
//    evaluation of the last two facilities.
//
// 2. The work is not split, it is grown. Under a pruning bound, subtree sizes
//    differ by five orders of magnitude, so cutting the tree at a fixed depth
//    and dealing out the pieces leaves one thread holding the hard part of
//    the proof while every other thread finishes in the first second: at
//    N = 25 that cost 62 seconds of which 61 were one thread. Instead the
//    search runs breadth-first from the root, expanding every live node's
//    children in parallel one level at a time, until the frontier is large
//    enough to keep the device busy -- and only then hands those nodes out,
//    one subtree per task, to a depth-first phase. Small instances never
//    leave the breadth-first phase at all.
//
// 3. The count that CGBN accumulates is no longer the iteration count. A
//    pruning search does not visit N! anything, so "iterations == N!" can no
//    longer be the check that the work was tiled correctly. What replaces it
//    is exact and just as strong: every node the search disposes of stands
//    for a known number of complete permutations -- (N - L)! for a node at
//    level L -- and those numbers must sum to exactly N!. That sum is what
//    the CGBN accumulator holds. At N = 30 it is 2.65e32, which no 64-bit
//    counter could express, and it is checked against N! on the device.
//
// 4. That check is where the signed CGBN comes in. The residual
//    N! - accounted is formed as a signed 1024-bit value, so a search that
//    over-counted is as visible as one that under-counted; in unsigned
//    arithmetic an over-count would wrap into an enormous positive number and
//    read as a wildly wrong under-count.
//
// The per-thread workspace is far too large for shared memory (tens of KB per
// thread), so it lives in global memory, one contiguous slice per thread.
// cgbncudaqap's -m/--shmem-size therefore has nothing to size and is replaced
// by -t/--max-threads, which caps the launch, and -k/--frontier, which
// overrides how many live nodes the breadth-first phase stops at.

#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <numeric>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <cerrno>
#include <cassert>
#include <getopt.h>

#include <cuda.h>
#include <cuda_runtime.h>

// GMP must precede cgbn.h: the header is compiled on the host pass too, and
// selects its host backend from whether <gmp.h> has been seen. Without it the
// host pass stops at "#error You must use GMP for now". GMP is also what
// builds the factorial table and formats the accumulators in decimal, below.
#include <gmp.h>
#include <cgbn/cgbn.h>

// One warp per CGBN instance, one 32-bit limb per lane, no padding
// (BITS/32 % TPI == 0). 1024 bits holds every factorial up to 170!, and the
// accounting arithmetic is (count <= 2^64) * (N - L)!, so the accumulator is
// comfortable anywhere the memory for the workspace is.
static const uint32_t CGBN_TPI  = 32U;
static const uint32_t CGBN_BITS = 1024U;

// 170! < 2^1024 < 171!. This is the only hard size ceiling in the program;
// device memory bites long before it.
static const uint32_t MAX_ORDER = 170U;

typedef cgbn_context_t<CGBN_TPI> cgbn_ctx_t;
typedef cgbn_env_t<cgbn_ctx_t, CGBN_BITS> cgbn_env;
typedef cgbn_mem_t<CGBN_BITS> cgbn_big_t;

// The reduced root matrices, shared read-only by every thread: A is the flow
// matrix and B the distance matrix, both column-major with leading dimension
// N, both destroyed by the root reduction; C is the linear term the reduction
// leaves behind. VEKSUM[L] is the offset of level L's sorted-row segment in
// ASPEI/BSPEI.
__managed__ int32_t* GA;
__managed__ int32_t* GB;
__managed__ int32_t* GC0;
__managed__ int32_t* GVEKSUM;

// Packed argmin key: (cost << 32) | winning global thread id. Its high half
// doubles as the shared incumbent every thread prunes against, so a single
// 64-bit atomicMin both publishes a new best solution and tightens the bound
// for the whole device. The initial value is (host heuristic cost << 32) with
// a thread id of ~0, which is how "no thread beat the heuristic" is spelled.
__device__ uint64_t MCK = ~0ULL;

// Hand-out cursor for the frontier work queue.
__device__ unsigned long long GNEXT = 0ULL;
__device__ uint64_t TOT = 0UL;

// Per-thread global-memory workspace, and the per-thread outputs.
__managed__ int32_t* GWS;
__managed__ uint64_t* GLPR;
__managed__ uint64_t* GITC;
__managed__ uint32_t* GSOL;

// A frontier entry is a node's mixed-radix rank at the current level -- one
// number, from which Descend() rebuilds the node's whole state. It has to be
// 128 bits: the rank at depth L runs to N!/(N-L)!, which at N = 25 passes
// 2^64 at depth 16, and a wrapped rank decodes into a different node
// entirely. That is not a crash; it is a search that quietly covers some
// subtrees twice and others not at all -- and it still reports a residual of
// zero, because the accounting counts what the search disposed of, not what
// it should have. cgbncudaqap widened its permutation ranks to 128 bits for
// the same reason. Here the ceiling is not a fixed problem size: the host
// stops deepening the frontier one level before N!/(N-L)! would leave 128
// bits, and the depth-first phase takes over from wherever that is.
typedef __uint128_t rank_t;

// The two frontier buffers, ping-ponged one level at a time, and the counter
// the expansion appends through.
__managed__ rank_t* GF0;
__managed__ rank_t* GF1;
__managed__ unsigned long long* GFCNT;

// Per-level counter totals: [L] is the number of nodes disposed of at
// level L.
__managed__ uint64_t* GSUM;

// Factorials 0! .. N! as 1024-bit values, and the CGBN outputs.
__managed__ cgbn_big_t* GFACT;
__managed__ cgbn_big_t* GPART;
__managed__ cgbn_big_t* GITOT;
__managed__ cgbn_big_t* GACCT;
__managed__ cgbn_big_t* GRESID;
__managed__ int32_t* GRSIGN;

std::vector<std::vector<uint32_t>> HFLG;
std::vector<std::vector<uint32_t>> HDST;
std::vector<uint32_t> HAS;

template<typename _Ty>
void checkCudaReturnValue(_Ty R, const char* FN, const char* FL, int32_t LN) {
  if (R) {
    (void) fprintf(stderr, "CUDA error at %s [%s/%i]: error code=%u (%s)\n",
                   FN, FL, LN, static_cast<uint32_t>(R),
                   cudaGetErrorName(static_cast<cudaError_t>(R)));
    exit(EXIT_FAILURE);
  }
}

#define checkCudaError(X) checkCudaReturnValue((X), #X, __FILE__, __LINE__)

// The bound machinery, ported from qapbb.c. The Fortran subprogram names
// (SSORT, LSAP, WEGSPE, PROGNO) are kept, lower-cased, so the device code
// still diffs by eye against qapbb.c and against the FORTRAN it came from --
// the same reason namespace qap in cgbncudaqap.cu keeps next_permutation's
// name. Every array is 1-based, element 0 allocated and unused, exactly as
// in the original.
namespace qapbb {

// Fortran SUBROUTINE SSORT. Shell sort of A(1..L) increasing, carrying B along.
__device__
void ssort(int32_t* a, int32_t* b, int32_t l) {
  int32_t f, n2, s, t, ls, i, is, ah, bh, j, js;

  f = 1;
  if (l <= f)
    return;

  n2 = (l - f + 1) / 2;
  s = 1023;

  for (t = 1; t <= 10; t++) {
    if (s > n2)
      goto L90;
    ls = l - s;
    for (i = f; i <= ls; i++) {
      is = i + s;
      ah = a[is];
      bh = b[is];
      j = i;
      js = is;
    L5:
      if (ah >= a[j])
        goto L10;
      a[js] = a[j];
      b[js] = b[j];
      js = j;
      j = j - s;
      if (j >= f)
        goto L5;
    L10:
      a[js] = ah;
      b[js] = bh;
    }
  L90:
    s = s / 2;
  }
}

// Fortran SUBROUTINE LSAP (U. Derigs). Linear sum assignment by shortest augmenting
// path, on the n x n row-wise cost matrix c. Returns the optimal value in
// *z; the assignment and duals fall out in spalte / ys / yt. This is the
// node bound, and it is where nearly all of the search's time goes.
__device__
void lsap(int32_t n, int32_t sup, const int32_t* c, int32_t* z, int32_t* zeile,
          int32_t* spalte, int32_t* dminus, int32_t* dplus, int32_t* ys,
          int32_t* yt, int32_t* vor, int32_t* label) {
  int32_t i, j, ik, cc, ui, jo, vj, u, us, usi, d, indexv, w, ws, wsi, vgl, ind;
  int32_t is, isj;

  ui = 0;
  jo = 0;
  d = 0;
  indexv = 0;

  for (i = 1; i <= n; i++) {
    zeile[i] = 0;
    spalte[i] = 0;
    vor[i] = 0;
    ys[i] = 0;
    yt[i] = 0;
  }

  ik = 0;
  for (i = 1; i <= n; i++) {
    for (j = 1; j <= n; j++) {
      ik = ik + 1;
      cc = c[ik];
      if (j == 1)
        goto L4;
      if ((cc - ui) >= 0)
        continue;
    L4:
      ui = cc;
      jo = j;
    }

    ys[i] = ui;
    if (zeile[jo] != 0)
      continue;

    zeile[jo] = i;
    spalte[i] = jo;
  }

  for (j = 1; j <= n; j++) {
    yt[j] = 0;
    if (zeile[j] == 0)
      yt[j] = sup;
  }

  ik = 0;
  for (i = 1; i <= n; i++) {
    ui = ys[i];
    for (j = 1; j <= n; j++) {
      ik = ik + 1;
      vj = yt[j];
      if (vj <= 0)
        continue;
      cc = c[ik] - ui;
      if (cc >= vj)
        continue;
      yt[j] = cc;
      vor[j] = i;
    }
  }

  for (j = 1; j <= n; j++) {
    i = vor[j];
    if (i == 0)
      continue;
    if (spalte[i] != 0)
      continue;
    spalte[i] = j;
    zeile[j] = i;
  }

  for (i = 1; i <= n; i++) {
    if (spalte[i] != 0)
      continue;
    ui = ys[i];
    ik = (i - 1) * n;
    for (j = 1; j <= n; j++) {
      ik = ik + 1;
      if (zeile[j] != 0)
        continue;
      cc = c[ik];
      if ((cc - ui - yt[j]) > 0)
        continue;
      spalte[i] = j;
      zeile[j] = i;
      break;
    }
  }

  for (u = 1; u <= n; u++) {
    if (spalte[u] > 0)
      continue;

    us = (u - 1) * n;
    for (i = 1; i <= n; i++) {
      vor[i] = u;
      label[i] = 0;
      dplus[i] = sup;
      usi = us + i;
      dminus[i] = c[usi] - ys[u] - yt[i];
    }
    dplus[u] = 0;

  L105:
    d = sup;
    for (i = 1; i <= n; i++) {
      if (label[i])
        continue;
      if (dminus[i] >= d)
        continue;
      d = dminus[i];
      indexv = i;
    }

    if (zeile[indexv] <= 0)
      goto L400;
    label[indexv] = 1;
    w = zeile[indexv];
    ws = (w - 1) * n;
    dplus[w] = d;
    for (i = 1; i <= n; i++) {
      if (label[i])
        continue;
      wsi = ws + i;
      vgl = d + c[wsi] - ys[w] - yt[i];
      if (dminus[i] <= vgl)
        continue;
      dminus[i] = vgl;
      vor[i] = w;
    }
    goto L105;

  L400:
    w = vor[indexv];
    zeile[indexv] = w;
    ind = spalte[w];
    spalte[w] = indexv;
    if (w == u)
      goto L500;
    indexv = ind;
    goto L400;

  L500:
    for (i = 1; i <= n; i++) {
      if (dplus[i] == sup)
        goto L505;
      ys[i] = ys[i] + d - dplus[i];
    L505:
      if (dminus[i] >= d)
        continue;
      yt[i] = yt[i] + dminus[i] - d;
    }
  }

  *z = 0;
  for (i = 1; i <= n; i++) {
    is = (i - 1) * n;
    j = spalte[i];
    isj = is + j;
    *z = *z + c[isj];
  }
}

// Fortran SUBROUTINE WEGSPE. Rows of A sorted decreasingly and rows of B increasingly
// (diagonal excluded), stored row-wise in aspei / bspei. At the root
// (nmk == n) both are built from scratch; below it, level k's segment is
// copied to level k+1's with the just-fixed row's and column's contributions
// deleted. Level k's segment is left intact, which is what lets the DFS walk
// back up without recomputing anything.
__device__
void wegspe(int32_t n, int32_t k, int32_t nmk, int32_t izaehl, int32_t jzaehl,
            const int32_t* a, const int32_t* b, const int32_t* veksum,
            int32_t* vekt, const int32_t* boolv, const int32_t* bool1,
            int32_t* aspei, int32_t* bspei, int32_t* h1, int32_t ld) {
#define A(i, j) a[((j) - 1) * ld + ((i) - 1)]
#define B(i, j) b[((j) - 1) * ld + ((i) - 1)]

  int32_t nmkm2, j1, i, iz, j, nmkj, nsum1, j2, t, logi;

  nmkm2 = nmk - 1;
  if (nmk != n)
    goto L2900;

  j1 = 1;
  for (i = 1; i <= n; i++) {
    iz = 0;
    for (j = 1; j <= n; j++) {
      if (j == i)
        continue;
      iz = iz + 1;
      vekt[iz] = A(i, j);
    }
    ssort(vekt, h1, nmkm2);
    for (j = 1; j <= nmkm2; j++) {
      nmkj = nmk - j;
      aspei[j1] = vekt[nmkj];
      j1 = j1 + 1;
    }
  }

  j1 = 1;
  for (i = 1; i <= n; i++) {
    iz = 0;
    for (j = 1; j <= n; j++) {
      if (j == i)
        continue;
      iz = iz + 1;
      vekt[iz] = B(i, j);
    }

    ssort(vekt, h1, nmkm2);

    for (j = 1; j <= nmkm2; j++) {
      bspei[j1] = vekt[j];
      j1 = j1 + 1;
    }
  }

  return;

L2900:
  nsum1 = veksum[k + 1];
  j1 = nsum1;
  j2 = nsum1 - (nmk + 1) * nmk;

  for (i = 1; i <= n; i++) {
    if (boolv[i])
      goto L2930;
    t = A(i, izaehl);
    logi = 1;

    for (j = 1; j <= nmk; j++) {
      j2 = j2 + 1;
      if (aspei[j2] == t && logi) {
        logi = 0; /* 2980 */
        continue;
      }
      j1 = j1 + 1;
      aspei[j1] = aspei[j2];
    }
    continue;
  L2930:
    if (i == izaehl)
      j2 = j2 + nmk;
  }

  j1 = nsum1;
  j2 = nsum1 - (nmk + 1) * nmk;

  for (i = 1; i <= n; i++) {
    if (bool1[i])
      goto L2960;
    t = B(i, jzaehl);
    logi = 1;
    for (j = 1; j <= nmk; j++) {
      j2 = j2 + 1;
      if (bspei[j2] == t && logi) {
        logi = 0; /* 2970 */
        continue;
      }
      j1 = j1 + 1;
      bspei[j1] = bspei[j2];
    }
    continue;
  L2960:
    if (i == jzaehl)
      j2 = j2 + nmk;
  }

#undef A
#undef B
}

// Fortran SUBROUTINE PROGNO, minus its cache. The original stores each level's
// minimal scalar products in CSPEI so that backtracking can read them back
// instead of recomputing them. That cache pays only when a level's matrix is
// revisited unchanged, which is what qapbb's binary "forbid (i,j)" branch
// does and what this n-ary tree never does: coming back up, the next child
// is a different assignment and needs a different matrix. CSPEI is therefore
// dropped outright, which is also the largest single saving in the
// per-thread workspace -- N(N+1)(2N+1)/6 words, 11 KB at N = 20.
__device__
void progno_level(int32_t n, int32_t L, const int32_t* veksum,
                  const int32_t* aspei, const int32_t* bspei,
                  int32_t* umspei) {
  int32_t nmk = n - L;
  int32_t nmkm1 = nmk - 1;
  int32_t nsum = (L == 0) ? 0 : veksum[L];
  int32_t j2 = nsum;
  int32_t ju = 0;
  int32_t i, j, iz, j3, t;

  for (i = 1; i <= nmk; i++) {
    j3 = nsum;
    for (j = 1; j <= nmk; j++) {
      t = 0;
      for (iz = 1; iz <= nmkm1; iz++)
        t = t + aspei[j2 + iz] * bspei[j3 + iz];
      j3 = j3 + nmkm1;
      umspei[++ju] = t;
    }
    j2 = j2 + nmkm1;
  }
}

} // namespace qapbb

// One thread's scratch. Every field points into that thread's own contiguous
// slice of GWS; nothing here is shared, so nothing here needs an atomic. The
// slice is in global memory rather than shared because it is tens of KB per
// thread -- 48 KB at N = 25 -- which is half a block's entire shared budget
// for a single thread.
struct BBWork {
  int32_t* C;
  int32_t* umspei;
  int32_t* aspei;
  int32_t* bspei;
  int32_t* vekt;
  int32_t* h2;
  int32_t* ys;
  int32_t* yt;
  int32_t* zeile;
  int32_t* spalte;
  int32_t* dminus;
  int32_t* dplus;
  int32_t* vor;
  int32_t* label;
  int32_t* boolv;
  int32_t* bool1;
  int32_t* cur;
  int32_t* colsel;
  int32_t* sol;
  int32_t* frow;
  int32_t* fcol;
};

// Lays the slice out and returns its size in 32-bit words. Called with
// W == NULL on the host to size the allocation and with a real W on the
// device to carve it up, so the two can never disagree.
__host__ __device__ __forceinline__
size_t LayoutWork(BBWork* W, int32_t* Base, uint32_t N) {
  size_t NN = static_cast<size_t>(N) * N;
  size_t NAB = static_cast<size_t>(N) * (N + 1U) * (2U * N - 2U) / 6U + 2U;
  size_t V = static_cast<size_t>(N) + 3U;
  size_t O = 0;

#define BBFIELD(FLD, CNT)                 \
  do {                                    \
    if (W)                                \
      W->FLD = Base + O;                  \
    O += (CNT);                           \
  } while (0)

  BBFIELD(C, NN);
  BBFIELD(umspei, NN + 2U);
  BBFIELD(aspei, NAB);
  BBFIELD(bspei, NAB);
  BBFIELD(vekt, V);
  BBFIELD(h2, V);
  BBFIELD(ys, V);
  BBFIELD(yt, V);
  BBFIELD(zeile, V);
  BBFIELD(spalte, V);
  BBFIELD(dminus, V);
  BBFIELD(dplus, V);
  BBFIELD(vor, V);
  BBFIELD(label, V);
  BBFIELD(boolv, V);
  BBFIELD(bool1, V);
  BBFIELD(cur, V);
  BBFIELD(colsel, V);
  BBFIELD(sol, V);
  BBFIELD(frow, V);
  BBFIELD(fcol, V);

#undef BBFIELD

  return O;
}

static const int32_t BB_UNENDL = 1000000000;

// The shared incumbent. It is the high half of the argmin key, so the same
// 64-bit atomicMin that publishes a better solution also tightens the bound
// every other thread prunes against -- one variable, one atomic, no second
// broadcast. The load is volatile because a thread that keeps reading a
// cached incumbent would simply do more work, silently.
__device__ __forceinline__
uint32_t CurrentBound() {
  return static_cast<uint32_t>(*(volatile unsigned long long*) &MCK >> 32);
}

// One thread's view of the search: the read-only root data, its own
// workspace, its own counters.
struct Search {
  int32_t N;
  int32_t LD;
  const int32_t* A;
  const int32_t* B;
  const int32_t* C0;
  const int32_t* VEKSUM;
  BBWork W;
  uint64_t* Lpr;
  uint32_t* Sol;
  uint64_t Tid;
  uint64_t Nodes;
};

#define SA(i, j) S.A[((j) - 1) * S.LD + ((i) - 1)]
#define SB(i, j) S.B[((j) - 1) * S.LD + ((i) - 1)]
#define SC(i, j) S.W.C[((j) - 1) * S.LD + ((i) - 1)]

// Deepest frontier the expansion will build. A node's rank at level L is a
// mixed-radix number with L digits, and Descend decodes them into a local
// array of this size. The 128-bit rank runs out first at every size where
// both limits are reachable, so this is a backstop, not the binding one.
#define MAX_FRONTIER_DEPTH 32

// The m-th still-free location, counting from zero in increasing order. This
// fixed order is what makes a node's identity a mixed-radix rank.
// The level-0 sorted rows: A's rows descending and B's rows ascending, minus
// the diagonal. They are the same for every node and every thread, and
// nothing below level 0 ever writes over them, so each thread builds them
// once at kernel entry. Forgetting to is not an error the answer will show:
// the bound falls back to its linear part, stays valid, and simply stops
// pruning -- which at N = 20 was the difference between 101 thousand nodes
// and 88 million.
__device__ __forceinline__
void BuildRootTables(Search& S) {
  for (int32_t I = 1; I <= S.N; ++I) {
    S.W.boolv[I] = 0;
    S.W.bool1[I] = 0;
  }

  qapbb::wegspe(S.N, 0, S.N, 0, 0, S.A, S.B, S.VEKSUM, S.W.vekt, S.W.boolv,
                S.W.bool1, S.W.aspei, S.W.bspei, S.W.h2, S.LD);
}

__device__ __forceinline__
int32_t FreeCol(const Search& S, int32_t m) {
  for (int32_t j = 1; j <= S.N; j++)
    if (!S.W.bool1[j] && m-- == 0)
      return j;

  return 0;
}

// Every node the search disposes of -- pruned, or evaluated as a pair of leaf
// completions -- stands for exactly (N - L)! complete permutations. Recording
// the counts by level rather than the products keeps the per-thread counters
// 64-bit; the products are formed once, in CGBN, after the search.
__device__ __forceinline__
void Account(Search& S, int32_t L) {
  S.Lpr[L] += 1UL;
}

// Publish a complete assignment. The path down to level L is in W.sol; the
// caller has already written the rest into S.Sol.
__device__ __forceinline__
void PublishSolution(Search& S, int32_t L, int32_t V) {
  for (int32_t i = 1; i < L; i++)
    S.Sol[i - 1] = static_cast<uint32_t>(S.W.sol[i] - 1);

  // The solution has to be visible before the key that advertises it.
  __threadfence();

  uint64_t Key = (static_cast<uint64_t>(static_cast<uint32_t>(V)) << 32) |
                 static_cast<uint64_t>(static_cast<uint32_t>(S.Tid));

  (void) atomicMin((unsigned long long*) &MCK,
                   static_cast<unsigned long long>(Key));
}

__device__ __forceinline__
void RecordSolution(Search& S, int32_t L, int32_t r, int32_t c, int32_t iz,
                    int32_t jz, int32_t i1, int32_t j1, int32_t V) {
  S.Sol[r - 1] = static_cast<uint32_t>(c - 1);
  S.Sol[iz - 1] = static_cast<uint32_t>(jz - 1);
  S.Sol[i1 - 1] = static_cast<uint32_t>(j1 - 1);
  PublishSolution(S, L, V);
}

// Fold the pair (r -> c) into C, either way. Descending, the interactions
// between the newly placed facility and every facility still free move into
// the linear term; backing out subtracts exactly what was added, so C is
// restored bit for bit. This is FORTRAN label 5220, both directions.
__device__ __forceinline__
void ShiftC(Search& S, int32_t r, int32_t c, bool Forward) {
  const int32_t N = S.N;

  for (int32_t i = 1; i <= N; i++) {
    if (S.W.boolv[i])
      continue;

    int32_t t1 = SA(r, i);
    int32_t t2 = SA(i, r);

    for (int32_t j = 1; j <= N; j++) {
      if (S.W.bool1[j])
        continue;

      int32_t d = t1 * SB(c, j) + t2 * SB(j, c);
      SC(i, j) += Forward ? d : -d;
    }
  }
}

// The LSAP that produced the bound also produced an assignment of every
// facility still free to a location still free, and that assignment is a
// complete, feasible solution to the original problem. Scoring it costs
// O(m^2) against the bound's O(m^3) and needs nothing the workspace does not
// already hold.
//
// It matters because a parallel search has no other way to get an incumbent
// early. Thousands of threads start at once from whatever bound the host
// heuristic found; until one of them reaches a leaf, every one of them is
// pruning against a stale number and exploring subtrees a sequential search
// would already have discarded. On chr25a, where the host heuristic comes in
// at 5062 against an optimum of 3796, that speculative work was 24 times the
// sequential node count.
//
// The score is computed in the reduced arithmetic the search runs in, which
// gives the true objective: for a complete assignment phi, the objective is
// sum C(i, phi(i)) + sum A(i,j) B(phi(i), phi(j)) over the reduced matrices,
// and zpc already holds that sum over the facilities already placed.
__device__ __forceinline__
void TryCompletion(Search& S, int32_t L, int32_t r, int32_t c, int32_t zpc) {
  const int32_t N = S.N;
  int32_t m = 0;
  int32_t a, b, i, j;

  for (i = 1; i <= N; i++)
    if (!S.W.boolv[i])
      S.W.frow[++m] = i;

  j = 0;
  for (i = 1; i <= N; i++)
    if (!S.W.bool1[i])
      S.W.fcol[++j] = i;

  int32_t V = zpc;

  for (a = 1; a <= m; a++) {
    i = S.W.frow[a];
    j = S.W.fcol[S.W.spalte[a]];
    V += SC(i, j) + SA(r, i) * SB(c, j) + SA(i, r) * SB(j, c);
  }

  for (a = 1; a <= m; a++) {
    int32_t ia = S.W.frow[a];
    int32_t ja = S.W.fcol[S.W.spalte[a]];

    for (b = 1; b <= m; b++) {
      if (b == a)
        continue;
      V += SA(ia, S.W.frow[b]) * SB(ja, S.W.fcol[S.W.spalte[b]]);
    }
  }

  if (static_cast<uint32_t>(V) >= CurrentBound())
    return;

  S.Sol[r - 1] = static_cast<uint32_t>(c - 1);

  for (a = 1; a <= m; a++)
    S.Sol[S.W.frow[a] - 1] =
      static_cast<uint32_t>(S.W.fcol[S.W.spalte[a]] - 1);

  PublishSolution(S, L, V);
}

// The bound of the child that places facility r at location c, given a
// workspace already holding the parent's state. Returns zpc + zstern, and
// leaves the child's sorted rows in aspei/bspei ready to descend into. C is
// deliberately NOT updated: a child that fails this test never pays for it.
__device__ __forceinline__
int32_t ChildBound(Search& S, int32_t k, int32_t r, int32_t c, int32_t zpc) {
  const int32_t N = S.N;
  int32_t L = k + 1;
  int32_t zstern = 0;
  int32_t i, j, iz, jz, t1, t2;

  qapbb::wegspe(N, k, N - L, r, c, S.A, S.B, S.VEKSUM, S.W.vekt, S.W.boolv,
                S.W.bool1, S.W.aspei, S.W.bspei, S.W.h2, S.LD);
  qapbb::progno_level(N, L, S.VEKSUM, S.W.aspei, S.W.bspei, S.W.umspei);

  // FORTRAN label 5490's tail: fold C and the new pair's interactions into
  // the linear part of the child's bound matrix, without touching C itself.
  iz = 0;
  for (i = 1; i <= N; i++) {
    if (S.W.boolv[i])
      continue;
    t1 = SA(i, r);
    t2 = SA(r, i);
    jz = iz;
    for (j = 1; j <= N; j++) {
      if (S.W.bool1[j])
        continue;
      jz++;
      S.W.umspei[jz] += SC(i, j) + t1 * SB(j, c) + t2 * SB(c, j);
    }
    iz += N - L;
  }

  qapbb::lsap(N - L, BB_UNENDL, S.W.umspei, &zstern, S.W.zeile, S.W.spalte,
              S.W.dminus, S.W.dplus, S.W.ys, S.W.yt, S.W.vor, S.W.label);
  S.Nodes += 1UL;

  // Every eighth live node rather than every one. Scoring the completion is
  // O(m^2) against the bound's O(m^3), which sounds free and measures at 28%
  // of the search on an instance whose incumbent was already optimal and had
  // nothing to gain. Sampling keeps nearly all of the benefit -- with tens of
  // thousands of threads the incumbent still improves in bursts -- for a few
  // percent.
  if ((S.Nodes & 7UL) == 0UL &&
      static_cast<uint32_t>(zpc + zstern) < CurrentBound())
    TryCompletion(S, L, r, c, zpc);

  return zpc + zstern;
}

// Two facilities and two locations left: both completions, in closed form.
// This is FORTRAN label 5330. C has not been updated for this child, so the
// four entries the completions need are formed on the spot.
//
// L and r are always the same number here, since the facility placed at
// level L is facility L; they are kept apart because they mean different
// things -- L is how much of W.sol is valid, r is which facility this is.
__device__ __forceinline__
void LeafPair(Search& S, int32_t L, int32_t r, int32_t c, int32_t zpc) {
  const int32_t N = S.N;
  int32_t iz = 0, i1 = 0, jz = 0, j1 = 0, V;

  for (int32_t i = 1; i <= N; i++)
    if (!S.W.boolv[i]) {
      if (!iz)
        iz = i;
      else {
        i1 = i;
        break;
      }
    }

  for (int32_t j = 1; j <= N; j++)
    if (!S.W.bool1[j]) {
      if (!jz)
        jz = j;
      else {
        j1 = j;
        break;
      }
    }

#define SCC(i, j) (SC(i, j) + SA(r, i) * SB(c, j) + SA(i, r) * SB(j, c))
  S.Nodes += 1UL;

  V = zpc + SCC(iz, jz) + SCC(i1, j1) +
      SA(iz, i1) * SB(jz, j1) + SA(i1, iz) * SB(j1, jz);
  if (static_cast<uint32_t>(V) < CurrentBound())
    RecordSolution(S, L, r, c, iz, jz, i1, j1, V);

  V = zpc + SCC(iz, j1) + SCC(i1, jz) +
      SA(iz, i1) * SB(j1, jz) + SA(i1, iz) * SB(jz, j1);
  if (static_cast<uint32_t>(V) < CurrentBound())
    RecordSolution(S, L, r, c, iz, j1, i1, jz, V);
#undef SCC
}

// Rebuild the workspace so that it holds the state of the level-Depth node
// with the given rank: C shifted for every pair on the path, boolv/bool1
// marking them, aspei/bspei sorted for that level. Returns zpart.
//
// Nothing on this path needs a bound -- the node is known to be live, or is
// being expanded -- so the LSAP is never called here, and the walk down costs
// O(Depth * N^2) rather than O(Depth * N^3).
__device__
int32_t Descend(Search& S, rank_t Rank, int32_t Depth) {
  const int32_t N = S.N;
  int32_t M[MAX_FRONTIER_DEPTH];
  int32_t zpart = 0;
  rank_t R = Rank;

  // The rank is mixed-radix with radix (N - k) at level k; peel the digits
  // off the bottom. nvcc compiles a 128-bit division as a called helper,
  // which is fine: this runs once per frontier node, not once per bound.
  for (int32_t k = Depth - 1; k >= 0; --k) {
    M[k] = static_cast<int32_t>(R % static_cast<rank_t>(N - k));
    R /= static_cast<rank_t>(N - k);
  }

  for (int32_t i = 0; i < N * N; i++)
    S.W.C[i] = S.C0[i];

  for (int32_t i = 1; i <= N; i++) {
    S.W.boolv[i] = 0;
    S.W.bool1[i] = 0;
  }

  for (int32_t k = 0; k < Depth; ++k) {
    int32_t L = k + 1;
    int32_t r = L;
    int32_t c = FreeCol(S, M[k]);

    S.W.boolv[r] = 1;
    S.W.bool1[c] = 1;
    S.W.sol[r] = c;
    S.W.colsel[k] = c;
    zpart += SC(r, c);

    qapbb::wegspe(N, k, N - L, r, c, S.A, S.B, S.VEKSUM, S.W.vekt, S.W.boolv,
                  S.W.bool1, S.W.aspei, S.W.bspei, S.W.h2, S.LD);
    ShiftC(S, r, c, true);
  }

  return zpart;
}

// One level of breadth-first expansion, and the reason this program is a GPU
// program at all.
//
// A static split of the tree cannot work here. Subtree sizes under a pruning
// bound differ by five orders of magnitude, so whatever depth the prefixes
// are cut at, one thread draws the subtree that contains the hard part of the
// proof and every other thread finishes in the first second and waits for it.
// Measured at N = 25 with a fixed depth-5 split: 8.68 million nodes in total,
// 257352 of them in one thread, and 62 seconds of wall clock of which 61 were
// that one thread.
//
// So the frontier is grown instead of guessed. Level by level, every live
// node's children are bounded in parallel and the survivors collected; the
// host stops when there are enough of them to keep the device busy. Dead
// children are accounted for and dropped here, so the frontier that reaches
// the search kernel contains only nodes that are actually worth searching --
// and it contains a lot of them, which is what makes the tail short.
//
// Each thread rebuilds its parent's state from the parent's rank. Rebuilding
// is O(L * N^2) against a node's O(N^3), and it is what buys the expansion
// its complete freedom from shared state: no frontier node depends on any
// other, so there is no synchronisation inside a level at all.
__global__ void ExpandFrontier(uint32_t N, uint32_t L, const rank_t* Cur,
                               uint64_t NC, rank_t* Next, uint64_t NCap,
                               unsigned long long* NCnt, uint32_t NTH,
                               uint64_t WPT) {
  uint32_t TIB = threadIdx.y * blockDim.x + threadIdx.x;
  uint32_t BTH = blockDim.x * blockDim.y;
  uint32_t BID = blockIdx.y * gridDim.x + blockIdx.x;
  uint64_t GTID = static_cast<uint64_t>(BID) * BTH + TIB;

  if (GTID >= NTH)
    return;

  Search S;

  S.N = static_cast<int32_t>(N);
  S.LD = static_cast<int32_t>(N);
  S.A = GA;
  S.B = GB;
  S.C0 = GC0;
  S.VEKSUM = GVEKSUM;
  S.Lpr = GLPR + GTID * (N + 3U);
  S.Sol = GSOL + GTID * N;
  S.Tid = GTID;
  S.Nodes = 0UL;

  (void) LayoutWork(&S.W, GWS + GTID * WPT, N);
  BuildRootTables(S);

  const int32_t N32 = S.N;
  const int32_t Lv = static_cast<int32_t>(L);

  for (uint64_t Idx = GTID; Idx < NC; Idx += NTH) {
    rank_t Rank = Cur[Idx];
    int32_t zpart = Descend(S, Rank, Lv);

    for (int32_t m = 0; m < N32 - Lv; ++m) {
      int32_t r = Lv + 1;
      int32_t c = FreeCol(S, m);
      int32_t zpc;

      S.W.boolv[r] = 1;
      S.W.bool1[c] = 1;
      zpc = zpart + SC(r, c);

      if (r == N32 - 2) {
        LeafPair(S, r, r, c, zpc);
        Account(S, r);
      } else if (static_cast<uint32_t>(ChildBound(S, Lv, r, c, zpc)) >=
                 CurrentBound()) {
        Account(S, r);
      } else {
        rank_t Child = Rank * static_cast<rank_t>(N32 - Lv) +
                       static_cast<rank_t>(m);
        // The host only expands a level whose children are guaranteed to
        // fit, so this can only be false if that invariant is ever broken.
        // Counting past the end anyway is what lets the host say so instead
        // of writing over whatever follows the buffer.
        unsigned long long Slot = atomicAdd(NCnt, 1ULL);

        if (Slot < NCap)
          Next[Slot] = Child;
      }

      S.W.boolv[r] = 0;
      S.W.bool1[c] = 0;
    }
  }

  GITC[GTID] += S.Nodes;
}

// Depth-first search of the whole subtree below one frontier node.
//
// The C matrix carries the interactions between placed and unplaced
// facilities. Descending folds the new pair into it and backing out
// subtracts the same quantity, so C is restored exactly; that is the one
// piece of state the DFS mutates in place, and it is why every thread needs
// a private C rather than sharing the root's.
__device__
void RunSubtree(Search& S, rank_t Rank, int32_t Depth) {
  const int32_t N = S.N;
  int32_t k, r, c, m, L;
  int32_t zpart = Descend(S, Rank, Depth);

  k = Depth;
  S.W.cur[k] = 0;

  for (;;) {
    if (S.W.cur[k] >= N - k) {
      if (k == Depth)
        break;

      // Undo the assignment that took the search from level k - 1 to k.
      k--;
      r = k + 1;
      c = S.W.colsel[k];
      ShiftC(S, r, c, false);
      zpart -= SC(r, c);
      S.W.boolv[r] = 0;
      S.W.bool1[c] = 0;
      continue;
    }

    m = S.W.cur[k]++;
    c = FreeCol(S, m);
    r = k + 1;
    L = k + 1;

    S.W.boolv[r] = 1;
    S.W.bool1[c] = 1;

    int32_t zpc = zpart + SC(r, c);

    if (L == N - 2) {
      LeafPair(S, L, r, c, zpc);
      Account(S, L);
      S.W.boolv[r] = 0;
      S.W.bool1[c] = 0;
      continue;
    }

    if (static_cast<uint32_t>(ChildBound(S, k, r, c, zpc)) >=
        CurrentBound()) {
      Account(S, L);
      S.W.boolv[r] = 0;
      S.W.bool1[c] = 0;
      continue;
    }

    // Entering the child for good: now C earns its update.
    ShiftC(S, r, c, true);
    zpart = zpc;
    S.W.colsel[k] = c;
    S.W.sol[r] = c;
    k = L;
    S.W.cur[k] = 0;
  }
}

// The search kernel. Threads are persistent: each one repeatedly claims a
// chunk of consecutive frontier nodes from a single device-wide cursor and
// searches their subtrees. A static assignment would have been simpler, and
// wrong, for the reason given above ExpandFrontier.
__global__ void BranchAndBound(uint32_t N, uint32_t Depth, const rank_t* Front,
                               uint64_t NF, uint64_t QChunk, uint32_t NTH,
                               uint64_t WPT) {
  uint32_t TIB = threadIdx.y * blockDim.x + threadIdx.x;
  uint32_t BTH = blockDim.x * blockDim.y;
  uint32_t BID = blockIdx.y * gridDim.x + blockIdx.x;
  uint64_t GTID = static_cast<uint64_t>(BID) * BTH + TIB;

  if (GTID == 0UL)
    (void) atomicAdd((unsigned long long*) &TOT,
                     static_cast<unsigned long long>(NTH));

  if (GTID >= NTH)
    return;

  Search S;

  S.N = static_cast<int32_t>(N);
  S.LD = static_cast<int32_t>(N);
  S.A = GA;
  S.B = GB;
  S.C0 = GC0;
  S.VEKSUM = GVEKSUM;
  S.Lpr = GLPR + GTID * (N + 3U);
  S.Sol = GSOL + GTID * N;
  S.Tid = GTID;
  S.Nodes = 0UL;

  (void) LayoutWork(&S.W, GWS + GTID * WPT, N);
  BuildRootTables(S);

  for (;;) {
    unsigned long long Lo =
      atomicAdd(&GNEXT, static_cast<unsigned long long>(QChunk));

    if (Lo >= NF)
      break;

    uint64_t Hi = static_cast<uint64_t>(Lo) + QChunk;

    if (Hi > NF)
      Hi = NF;

    for (uint64_t I = Lo; I < Hi; ++I)
      RunSubtree(S, Front[I], static_cast<int32_t>(Depth));
  }

  GITC[GTID] += S.Nodes;
}

// Widen a uint64_t into a CGBN register. There is no cgbn_set_ui64, and the
// accumulator is wider than 64 bits, so the value is assembled from its two
// 32-bit halves.
__device__ __forceinline__
void CgbnSetUi64(cgbn_env& E, cgbn_env::cgbn_t& R, uint64_t V) {
  cgbn_set_ui32(E, R, static_cast<uint32_t>(V >> 32));
  cgbn_shift_left(E, R, R, 32U);
  (void) cgbn_add_ui32(E, R, R, static_cast<uint32_t>(V & 0xffffffffULL));
}

// Stage one of the node total: NInst CGBN instances, each one warp, sum a
// strided share of the per-thread counters.
__global__ void AccumulateIterations(const uint64_t* ITC, uint32_t NTH,
                                     cgbn_big_t* Part, uint32_t NInst,
                                     cgbn_error_report_t* Rpt) {
  uint32_t Inst = (blockIdx.x * blockDim.x + threadIdx.x) / CGBN_TPI;

  // Warp-uniform: every lane of an instance takes the same branch, which is
  // what CGBN's cooperative operations require.
  if (Inst >= NInst)
    return;

  cgbn_ctx_t Ctx(cgbn_report_monitor, Rpt, Inst);
  cgbn_env E(Ctx.env<cgbn_env>());
  cgbn_env::cgbn_t A, T;

  cgbn_set_ui32(E, A, 0U);

  for (uint32_t I = Inst; I < NTH; I += NInst) {
    CgbnSetUi64(E, T, ITC[I]);
    (void) cgbn_add(E, A, A, T);
  }

  cgbn_store(E, &Part[Inst], A);
}

// Stage two: a single instance folds the partials into the one global count.
__global__ void CombineIterations(cgbn_big_t* Part, uint32_t NInst,
                                  cgbn_big_t* Tot, cgbn_error_report_t* Rpt) {
  if (blockIdx.x != 0U || threadIdx.x >= CGBN_TPI)
    return;

  cgbn_ctx_t Ctx(cgbn_report_monitor, Rpt, 0U);
  cgbn_env E(Ctx.env<cgbn_env>());
  cgbn_env::cgbn_t A, T;

  cgbn_set_ui32(E, A, 0U);

  for (uint32_t I = 0; I < NInst; ++I) {
    cgbn_load(E, T, &Part[I]);
    (void) cgbn_add(E, A, A, T);
  }

  cgbn_store(E, Tot, A);
}

// Collapse the per-thread disposal counts to one count per level. Ordinary
// 64-bit atomics: there are at most N + 1 distinct counters and no count can
// approach 2^64, so nothing here needs to be wide. Sum must be zeroed first.
__global__ void SumCounters(const uint64_t* Lpr, uint32_t NTH, uint32_t N,
                            uint64_t* Sum) {
  uint32_t T = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t Stride = gridDim.x * blockDim.x;

  for (uint32_t I = T; I < NTH; I += Stride) {
    for (uint32_t L = 0; L <= N; ++L) {
      uint64_t V = Lpr[static_cast<size_t>(I) * (N + 3U) + L];

      if (V)
        (void) atomicAdd((unsigned long long*) &Sum[L],
                         static_cast<unsigned long long>(V));
    }
  }
}

// The permutation accounting, and the check it exists for.
//
// A node disposed of at level L -- pruned, or evaluated as a pair of leaf
// completions -- stands for exactly (N - L)! complete permutations. Summing
// count * factorial over the levels must give exactly N!, because the levels
// partition the permutation space: that is this program's equivalent of
// cgbncudaqap's "iterations == N!", and it is the only thing that can catch a
// frontier that lost a node or accounted for one twice.
//
// N! passes 2^64 at N = 21 and is 1.55e25 at N = 25, so the sum has to be
// arbitrary precision; the counts stay 64-bit and only the products are wide.
// cgbn_mul returns the low half of the product, which is the whole product
// here -- a count below 2^64 times a factorial below 2^1024 could only need
// more than 1024 bits at sizes the workspace could never be allocated for.
//
// The residual is formed as a SIGNED 1024-bit value. Unsigned, an over-count
// would wrap N! - accounted into an enormous positive number and read as a
// wildly wrong under-count; signed, its sign says which way the error went.
__global__ void CombinePermutations(const uint64_t* Sum, cgbn_big_t* Fact,
                                    uint32_t N, cgbn_big_t* Acct,
                                    cgbn_big_t* Resid, int32_t* Sign,
                                    cgbn_error_report_t* Rpt) {
  if (blockIdx.x != 0U || threadIdx.x >= CGBN_TPI)
    return;

  cgbn_ctx_t Ctx(cgbn_report_monitor, Rpt, 0U);
  cgbn_env E(Ctx.env<cgbn_env>());
  cgbn_env::cgbn_t A, T, F;

  cgbn_set_ui32(E, A, 0U);

  for (uint32_t L = 1U; L + 2U <= N; ++L) {
    if (Sum[L] == 0UL)
      continue;

    cgbn_load(E, F, &Fact[N - L]);
    CgbnSetUi64(E, T, Sum[L]);
    cgbn_mul(E, T, T, F);
    (void) cgbn_add(E, A, A, T);
  }

  cgbn_store(E, Acct, A);

  cgbn_load(E, F, &Fact[N]);
  (void) cgbn_sub(E, T, F, A);

  int32_t Neg = cgbn_signed_is_negative(E, T) ? 1 : 0;
  (void) cgbn_signed_abs(E, T, T);
  cgbn_store(E, Resid, T);

  if (threadIdx.x == 0U)
    *Sign = Neg;
}

static struct timespec tp_start;
static struct timespec tp_end;
static struct timespec tp_acc_start;
static struct timespec tp_acc_end;

// [[nodiscard]]: a failed timestamp leaves *ts untouched, which would make
// PrintTimediff report a difference against uninitialized storage. Callers
// must decide what to do rather than drop the status on the floor.
[[nodiscard]] static int Timestamp(struct timespec* ts) {
  if (clock_gettime(CLOCK_REALTIME, ts) != 0) {
    std::cerr << "clock_gettime(2) failed: " << strerror(errno)
      << std::endl;
    return -1;
  }

  return 0;
}

// Deliberately NOT the verbatim twin of the PrintTimediff in qap.cpp and
// cudaqap.cu: this variant takes a label, because there are two intervals
// worth reporting here -- the search itself and the CGBN reduction that
// follows it. The arithmetic below is identical to the other two.
static void PrintTimediff(const char* Label, const struct timespec* S,
                          const struct timespec* E) {
  struct timespec R;

  if (clock_getres(CLOCK_REALTIME, &R) != 0) {
    std::cerr << "clock_getres(2) failed: " << strerror(errno)
      << std::endl;
    return;
  }

  // Take the difference as one nanosecond count so that the nanosecond field
  // borrows from the second field. Differencing the two fields separately and
  // then taking labs() of the remainder loses the borrow entirely, and is only
  // correct when the interval does not cross a second boundary.
  int64_t Nns = ((int64_t) E->tv_sec - (int64_t) S->tv_sec) * 1000000000L +
                ((int64_t) E->tv_nsec - (int64_t) S->tv_nsec);
  const char* Sgn = "";

  // Only reachable if the clock stepped backwards mid-run; show it rather
  // than hiding it in an absolute value.
  if (Nns < 0) {
    Nns = -Nns;
    Sgn = "-";
  }

  char Fill = std::cout.fill();

  std::cout << "CPU Clock Resolution: " << (int64_t) R.tv_sec << '.'
    << std::setfill('0') << std::setw(9) << (int64_t) R.tv_nsec
    << '.' << std::endl;
  std::cout << Label << ": " << Sgn << Nns / 1000000000L << '.'
    << std::setfill('0') << std::setw(9) << Nns % 1000000000L << " second(s)."
    << std::endl;

  std::cout.fill(Fill);
}

void ReadFromFile(const std::string& FileName,
                  std::vector<std::vector<uint32_t>>& IV) {
  std::ifstream IFS(FileName);
  if (IFS.good()) {
    std::string Line;
    const char* Sep = ", \t\n";
    char* End;

    while (std::getline(IFS, Line)) {
      if (!Line.empty()) {
        std::vector<uint32_t> V;
        if (char* Tok = strtok_r(const_cast<char*>(Line.data()), Sep, &End))
          V.push_back(static_cast<uint32_t>(std::stoul(Tok)));

        while (char* Tok = strtok_r(NULL, Sep, &End))
          V.push_back(static_cast<uint32_t>(std::stoul(Tok)));

        IV.push_back(V);
      }
    }

    IFS.close();
  } else {
    std::cerr << "Invalid input file " << FileName << " was specified."
      << std::endl;
    std::exit(1);
  }
}

void ReadFromFile(const std::string& FileName) {
  std::ifstream IFS(FileName);

  if (IFS.good()) {
    std::string Line;
    const char* Sep = ", \t\n";
    char* End;
    uint32_t N = 0U;
    uint32_t C;

    while (std::getline(IFS, Line)) {
      if (!Line.empty()) {
        if (char* Tok = strtok_r(const_cast<char*>(Line.data()), Sep, &End))
          N = static_cast<uint32_t>(std::stoul(Tok));
        while (strtok_r(NULL, Sep, &End));
        break;
      }
    }

    std::vector<uint32_t> V;
    C = 0U;

    while (std::getline(IFS, Line) && C++ <= N) {
      if (!Line.empty()) {
        V.clear();
        if (char* Tok = strtok_r(const_cast<char*>(Line.data()), Sep, &End))
          V.push_back(static_cast<uint32_t>(std::stoul(Tok)));

        while (char* Tok = strtok_r(NULL, Sep, &End))
          V.push_back(static_cast<uint32_t>(std::stoul(Tok)));

        HFLG.push_back(V);
      }
    }


    V.clear();
    C = 0U;

    while (std::getline(IFS, Line) && C++ <= N) {
      if (!Line.empty()) {
        V.clear();
        if (char* Tok = strtok_r(const_cast<char*>(Line.data()), Sep, &End))
          V.push_back(static_cast<uint32_t>(std::stoul(Tok)));

        while (char* Tok = strtok_r(NULL, Sep, &End))
          V.push_back(static_cast<uint32_t>(std::stoul(Tok)));

        HDST.push_back(V);
      }
    }

  } else {
    std::cerr << "Invalid input file " << FileName << " was specified."
      << std::endl;
    std::exit(1);
  }

}

void ReadFromFile(const std::string& FLGName,
                  const std::string& DSTName) {
  ReadFromFile(FLGName, HFLG);
  ReadFromFile(DSTName, HDST);
}

// GFLG/GDST are __managed__ and read directly by the kernel, so nothing is
// handed back to the caller. N is the matrix order; Verify() has already

void PrintHostVector(const std::vector<uint32_t>& HV) {
  std::cout << "{ ";
  if (!HV.empty()) {
    std::vector<uint32_t>::const_iterator I = HV.begin();
    std::vector<uint32_t>::const_iterator E = HV.end();

    std::cout << *I;
    while (++I != E) {
      std::cout << ", " << *I;
    }
  }

  std::cout << " }";
}

void PrintGraphVector(const std::vector<std::vector<uint32_t>>& V) {
  std::cout << "{ ";
  if (!V.empty()) {
    std::vector<std::vector<uint32_t>>::const_iterator I = V.begin();
    std::vector<std::vector<uint32_t>>::const_iterator E = V.end();

    PrintHostVector(*I);

    while (++I != E) {
      std::cout << ',' << std::endl << "  ";
      PrintHostVector(*I);
    }
  }

  std::cout << " }" << std::endl;
}

bool Verify() {
  if (HFLG.size() != HDST.size()) {
    std::cerr << "Outer Host Vectors are not the same size: "
      << HFLG.size() << '/' << HDST.size() << '.' << std::endl;
    return false;
  }

  std::vector<std::vector<uint32_t>>::const_iterator FI;
  std::vector<std::vector<uint32_t>>::const_iterator FE = HFLG.end();
  std::vector<std::vector<uint32_t>>::const_iterator DI;
  std::vector<std::vector<uint32_t>>::const_iterator DE = HDST.end();

  for (FI = HFLG.begin(), DI = HDST.begin(); FI != FE && DI != DE;
       ++FI, ++DI) {
    if ((*FI).size() != (*DI).size()) {
      std::cerr << "Inner Host Vectors are not the same size: "
        << (*FI).size() << '/' << (*DI).size() << '.' << std::endl;
      return false;
    }
  }

  return true;
}


// The objective on the untouched input matrices. Everything the search does
// happens in reduced arithmetic, where costs are shifted by the reduction
// constants; this is the only place the true objective is computed, and it
// is what the program reports. A disagreement between it and the search's
// own figure would mean the reduction accounting is wrong, so it is checked
// rather than assumed.
static int64_t HostObjective(uint32_t N, const std::vector<uint32_t>& P) {
  int64_t Z = 0;

  for (uint32_t I = 0; I < N; ++I)
    for (uint32_t J = 0; J < N; ++J)
      Z += static_cast<int64_t>(HFLG[I][J]) * HDST[P[I]][P[J]];

  return Z;
}

// Objective in the relabelled, pre-reduction space, on 1-based column-major
// matrices. Used by the starting heuristic only.
static int64_t WorkObjective(int32_t N, int32_t LD, const int32_t* A,
                             const int32_t* B, const int32_t* C,
                             const int32_t* P) {
  int64_t Z = 0;

  for (int32_t I = 1; I <= N; ++I) {
    Z += C[(P[I] - 1) * LD + (I - 1)];

    for (int32_t J = 1; J <= N; ++J)
      Z += static_cast<int64_t>(A[(J - 1) * LD + (I - 1)]) *
           B[(P[J] - 1) * LD + (P[I] - 1)];
  }

  return Z;
}

// A deterministic starting incumbent: the identity and its cyclic shifts,
// each driven to a 2-opt local optimum. Branch and bound with no incumbent
// spends its first minutes proving things it could have been told, and the
// GPU makes that worse rather than better -- every thread would explore the
// same unpruned top of the tree at once. There is no randomness here: two
// runs on the same input start from the same bound.
static int64_t StartBound(int32_t N, int32_t LD, const int32_t* A,
                          const int32_t* B, const int32_t* C, int32_t* Best) {
  std::vector<int32_t> P(static_cast<size_t>(N) + 2U);
  int32_t Starts = N < 32 ? N : 32;
  int64_t BZ = -1;

  for (int32_t S = 0; S < Starts; ++S) {
    for (int32_t I = 1; I <= N; ++I)
      P[I] = ((I - 1 + S) % N) + 1;

    int64_t Z = WorkObjective(N, LD, A, B, C, P.data());
    bool Improved;

    do {
      Improved = false;

      for (int32_t I = 1; I <= N && !Improved; ++I) {
        for (int32_t J = I + 1; J <= N && !Improved; ++J) {
          int32_t T = P[I];
          P[I] = P[J];
          P[J] = T;

          int64_t Y = WorkObjective(N, LD, A, B, C, P.data());

          if (Y < Z) {
            Z = Y;
            Improved = true;
          } else {
            T = P[I];
            P[I] = P[J];
            P[J] = T;
          }
        }
      }
    } while (Improved);

    if (BZ < 0 || Z < BZ) {
      BZ = Z;

      for (int32_t I = 1; I <= N; ++I)
        Best[I] = P[I];
    }
  }

  return BZ;
}

// Root relabelling of the facilities. The DFS fixes facilities in index
// order, so index order IS branching order, and branching on the most
// heavily connected facility first tightens the bound far earlier. This is
// the one place the n-ary tree can recover some of what it gave up by
// dropping qapbb's alternative-cost branching rule, and it is worth a great
// deal: chr20a falls from 32462 nodes to 11859, chr25a from 17.8 million to
// 827 thousand. B is untouched -- only the facilities are relabelled, and
// the locations they map to are unaffected.
static void ReorderFacilities(int32_t N, int32_t LD, int32_t* A, int32_t* C,
                              std::vector<int32_t>& Ord) {
  std::vector<int64_t> W(static_cast<size_t>(N) + 2U, 0);
  std::vector<int32_t> T(static_cast<size_t>(N) * N);

  Ord.assign(static_cast<size_t>(N) + 2U, 0);

  for (int32_t I = 1; I <= N; ++I) {
    Ord[I] = I;

    for (int32_t J = 1; J <= N; ++J)
      W[I] += A[(J - 1) * LD + (I - 1)] + A[(I - 1) * LD + (J - 1)];
  }

  for (int32_t I = 1; I <= N; ++I)
    for (int32_t J = I + 1; J <= N; ++J)
      if (W[Ord[J]] > W[Ord[I]])
        std::swap(Ord[I], Ord[J]);

  for (int32_t I = 1; I <= N; ++I)
    for (int32_t J = 1; J <= N; ++J)
      T[(J - 1) * LD + (I - 1)] = A[(Ord[J] - 1) * LD + (Ord[I] - 1)];

  (void) memcpy(A, T.data(), T.size() * sizeof(int32_t));

  for (int32_t I = 1; I <= N; ++I)
    for (int32_t J = 1; J <= N; ++J)
      T[(J - 1) * LD + (I - 1)] = C[(J - 1) * LD + (Ord[I] - 1)];

  (void) memcpy(C, T.data(), T.size() * sizeof(int32_t));
}

// The preamble of Fortran SUBROUTINE QAP: the first reduction C = A(i,i)*B(j,j),
// then the row-wise and column-wise reductions of A and B with their
// compensating terms accumulated into C. Verbatim from qapbb.c, and run once
// on the host because its result is the same for every thread.
static void RootReduce(int32_t N, int32_t LD, int32_t* a, int32_t* b,
                       int32_t* c, int32_t Unendl, int32_t* u, int32_t* v) {
#define A(i, j) a[((j) - 1) * LD + ((i) - 1)]
#define B(i, j) b[((j) - 1) * LD + ((i) - 1)]
#define C(i, j) c[((j) - 1) * LD + ((i) - 1)]
  int32_t i, j, kk, ra, rb, raa, rbb, ca1, bmj, bm, ca, am, zstern, ccc;

  for (i = 1; i <= N; i++) {
    zstern = A(i, i);
    for (j = 1; j <= N; j++)
      C(i, j) = zstern * B(j, j) + C(i, j);
  }

  for (j = 1; j <= N; j++) {
    A(j, j) = Unendl;
    B(j, j) = Unendl;
  }

  for (i = 1; i <= N; i++) {
    ra = A(i, 1);
    rb = B(i, 1);
    for (j = 2; j <= N; j++) {
      raa = A(i, j);
      rbb = B(i, j);
      if (raa < ra)
        ra = raa;
      if (rbb < rb)
        rb = rbb;
    }

    for (j = 1; j <= N; j++) {
      A(i, j) = A(i, j) - ra;
      B(i, j) = B(i, j) - rb;
    }

    u[i] = ra;
    v[i] = rb;
  }

  for (i = 1; i <= N; i++) {
    ca1 = u[i];
    for (j = 1; j <= N; j++) {
      bmj = v[j];
      bm = (N - 1) * bmj;
      for (kk = 1; kk <= N; kk++)
        if (kk != j)
          bm = bm + B(j, kk);
      ca = ca1 * bm;
      am = 0;
      for (kk = 1; kk <= N; kk++)
        if (kk != i)
          am = am + A(i, kk);
      C(i, j) = ca + bmj * am + C(i, j);
    }
  }

  for (i = 1; i <= N; i++) {
    ra = A(1, i);
    rb = B(1, i);
    for (j = 2; j <= N; j++) {
      raa = A(j, i);
      rbb = B(j, i);
      if (raa < ra)
        ra = raa;
      if (rbb < rb)
        rb = rbb;
    }

    for (j = 1; j <= N; j++) {
      A(j, i) = A(j, i) - ra;
      B(j, i) = B(j, i) - rb;
    }

    u[i] = ra;
    v[i] = rb;
  }

  for (i = 1; i <= N; i++) {
    A(i, i) = 0;
    B(i, i) = 0;
    ca1 = u[i];
    for (j = 1; j <= N; j++) {
      bmj = v[j];
      bm = (N - 1) * bmj;
      for (kk = 1; kk <= N; kk++)
        if (kk != j)
          bm = bm + B(kk, j);
      ca = ca1 * bm;
      am = 0;
      for (kk = 1; kk <= N; kk++)
        if (kk != i)
          am = am + A(kk, i);
      ccc = C(i, j) + ca + bmj * am;
      C(i, j) = ccc;
    }
  }
#undef A
#undef B
#undef C
}

// CGBN stores a value as little-endian 32-bit limbs; mpz_import reads exactly
// that layout. GMP is already a dependency of the CGBN host pass, so this
// costs nothing extra and avoids hand-rolling a bignum-to-decimal conversion.
std::string BigToString(const cgbn_big_t* M) {
  mpz_t Z;
  (void) mpz_init(Z);
  mpz_import(Z, CGBN_BITS / 32U, -1, sizeof(uint32_t), 0, 0, M->_limbs);

  char* S = mpz_get_str(NULL, 10, Z);
  std::string R(S);

  void (*FreeFn)(void*, size_t) = NULL;
  mp_get_memory_functions(NULL, NULL, &FreeFn);
  FreeFn(S, std::strlen(S) + 1U);
  mpz_clear(Z);

  return R;
}

// Fill GFACT[0 .. N] with 0! .. N! in CGBN's limb layout, and hand back N!
// in decimal for the Permutations: line. mpz_export writes only the
// significant limbs, so the buffer is zeroed first.
static std::string SetupFactorials(uint32_t N, cgbn_big_t* Fact) {
  mpz_t F;
  (void) mpz_init_set_ui(F, 1UL);

  for (uint32_t I = 0; I <= N; ++I) {
    if (I)
      mpz_mul_ui(F, F, I);

    (void) memset(&Fact[I], 0, sizeof(cgbn_big_t));

    size_t Count = 0;
    (void) mpz_export(Fact[I]._limbs, &Count, -1, sizeof(uint32_t), 0, 0, F);
  }

  char* S = mpz_get_str(NULL, 10, F);
  std::string R(S);

  void (*FreeFn)(void*, size_t) = NULL;
  mp_get_memory_functions(NULL, NULL, &FreeFn);
  FreeFn(S, std::strlen(S) + 1U);
  mpz_clear(F);

  return R;
}

uint64_t MinimumCostKey() {
  uint64_t LMC;
  checkCudaError(cudaMemcpyFromSymbol(&LMC, MCK, sizeof(uint64_t), 0,
                                      cudaMemcpyDeviceToHost));
  return LMC;
}

uint32_t NumThreads() {
  uint64_t LTOT;
  checkCudaError(cudaMemcpyFromSymbol(&LTOT, TOT, sizeof(uint64_t), 0,
                                      cudaMemcpyDeviceToHost));
  return static_cast<uint32_t>(LTOT);
}

// The incumbent starts at the heuristic's cost with a thread id of ~0: no
// thread can own that id, so a key still carrying it at the end means no
// thread beat the heuristic, and the heuristic's own permutation is the
// answer.
void SetupInitialCounters(uint32_t UB) {
  uint64_t Key = (static_cast<uint64_t>(UB) << 32) | 0xffffffffULL;
  uint64_t Zero = 0UL;
  unsigned long long ZeroL = 0ULL;

  checkCudaError(cudaMemcpyToSymbol(MCK, &Key, sizeof(uint64_t), 0,
                                    cudaMemcpyHostToDevice));
  checkCudaError(cudaMemcpyToSymbol(TOT, &Zero, sizeof(uint64_t), 0,
                                    cudaMemcpyHostToDevice));
  checkCudaError(cudaMemcpyToSymbol(GNEXT, &ZeroL, sizeof(unsigned long long),
                                    0, cudaMemcpyHostToDevice));
}

void PrintFlowGraphVector() {
  PrintGraphVector(HFLG);
}

void PrintDistanceVector() {
  PrintGraphVector(HDST);
}

void PrintAssignmentVector() {
  PrintHostVector(HAS);
}

void Print() {
  std::cout << "Flow Graph Vector:" << std::endl;
  PrintFlowGraphVector();
  std::cout << std::endl;

  std::cout << "Distance Vector:" << std::endl;
  PrintDistanceVector();
  std::cout << std::endl;

  std::cout << "Assignment Vector:" << std::endl;
  PrintAssignmentVector();
  std::cout << std::endl << std::endl;
}

static bool ValidateArguments(const std::string& SName,
                              const std::string& FLName,
                              const std::string& DSTName) {
  if (!SName.empty())
    return true;

  if (FLName.empty()) {
    std::cerr << "Flows input file name is invalid." << std::endl;
    return false;
  }

  if (DSTName.empty()) {
    std::cerr << "Distances input file name is invalid." << std::endl;
    return false;
  }

  return true;
}

static void PrintHelp() {
  std::cerr << "Usage: cgbncudaqapbb [-h | --help]" << std::endl;
  std::cerr << "           [-s <input-file> | --single-file <input-file>]"
    << std::endl;
  std::cerr << "           (use single-file matrix input format)." << std::endl;
  std::cerr << "           [ -f <flow-input-file> | --flinput <flow-input-file>]"
    << std::endl;
  std::cerr << "           [ -d <distance-input-file> | --dstinput <distance-input-file>]"
    << std::endl;
  std::cerr << "           [ -t <max-threads> | --max-threads <max-threads>]"
    << std::endl;
  std::cerr << "           (cap on the launch thread count. default is 0 -"
    << std::endl;
  std::cerr << "            as many as occupancy and device memory allow)."
    << std::endl;
  std::cerr << "           [ -k <frontier-size> | --frontier <frontier-size>]"
    << std::endl;
  std::cerr << "           (how many live nodes to expand the search frontier"
    << std::endl;
  std::cerr << "            to before the depth-first phase. default is 0 -"
    << std::endl;
  std::cerr << "            32 per thread)." << std::endl;
  std::cerr << "           [-p | --print] (print vector contents)." << std::endl;
}

static struct option long_options[] = {
  { "help",             no_argument,        0,  'h' },
  { "print",            no_argument,        0,  'p' },
  { "single-file",      required_argument,  0,  's' },
  { "flinput",          required_argument,  0,  'f' },
  { "dstinput",         required_argument,  0,  'd' },
  { "max-threads",      required_argument,  0,  't' },
  { "frontier",         required_argument,  0,  'k' },
  { 0,                  0,                  0,   0  }
};

int main(int argc, char* const argv[]) {
  int C;
  int OIx = 0;
  std::string FLGFile;
  std::string DSTFile;
  std::string SFile;
  uint64_t MaxThreads = 0UL;
  uint64_t UserFrontier = 0UL;
  bool DoPrint = false;

  while (1) {
    C = getopt_long(argc, argv, "hps:f:d:t:k:", long_options, &OIx);
    if (C == -1)
      break;

    switch (C) {
    case 'h':
      PrintHelp();
      return 0;
    case 'p':
      DoPrint = true;
      break;
    case 's':
      SFile = optarg;
      break;
    case 'f':
      FLGFile = optarg;
      break;
    case 'd':
      DSTFile = optarg;
      break;
    case 't':
      MaxThreads = static_cast<uint64_t>(std::stoull(optarg));
      break;
    case 'k':
      UserFrontier = static_cast<uint64_t>(std::stoull(optarg));
      break;
    default:
      PrintHelp();
      return 1;
    }
  }

  if (!ValidateArguments(SFile, FLGFile, DSTFile)) {
    PrintHelp();
    return 1;
  }

  if (!SFile.empty())
    ReadFromFile(SFile);
  else
    ReadFromFile(FLGFile, DSTFile);

  if (!Verify())
    return 1;

  uint32_t N = static_cast<uint32_t>(HFLG.size());

  if (N == 0U) {
    std::cerr << "Empty input." << std::endl;
    return 1;
  }

  // The accumulators are 1024 bits, and 170! is the last factorial that fits.
  // Nothing else in the program imposes a factorial-shaped limit: the search
  // never enumerates permutations, and no rank is ever formed. In practice
  // device memory for the per-thread workspace, which grows as N^3, bites
  // long before this does.
  if (N > MAX_ORDER) {
    std::cerr << "Problem size " << N << " is too large: " << N
      << "! does not fit in the " << CGBN_BITS
      << "-bit accumulator (the limit is " << MAX_ORDER << ")." << std::endl;
    return 1;
  }

  HAS.resize(N);
  std::iota(HAS.begin(), HAS.end(), 0U);

  // The bound machinery is 32-bit throughout, inherited from the FORTRAN, and
  // 10^9 is its infinity. sum(flow) * max(distance) is an upper bound on any
  // objective this instance can produce, and so on every intermediate the
  // search forms; refuse the input if that reaches 2^30, rather than let the
  // arithmetic wrap silently and report a wrong answer with a straight face.
  int64_t SumA = 0;
  int64_t MaxB = 0;

  for (uint32_t I = 0; I < N; ++I) {
    for (uint32_t J = 0; J < N; ++J) {
      SumA += static_cast<int64_t>(HFLG[I][J]);

      if (static_cast<int64_t>(HDST[I][J]) > MaxB)
        MaxB = static_cast<int64_t>(HDST[I][J]);
    }
  }

  if (MaxB && SumA > 1073741824L / MaxB) {
    std::cerr << "Input is too large for the 32-bit bound arithmetic: the "
      << "objective can reach " << SumA << " * " << MaxB
      << ", which exceeds 2^30." << std::endl;
    return 1;
  }

  int32_t LD = static_cast<int32_t>(N);
  std::vector<int32_t> HA(static_cast<size_t>(N) * N);
  std::vector<int32_t> HB(static_cast<size_t>(N) * N);
  std::vector<int32_t> HC(static_cast<size_t>(N) * N, 0);

  for (uint32_t I = 0; I < N; ++I) {
    for (uint32_t J = 0; J < N; ++J) {
      HA[static_cast<size_t>(J) * LD + I] = static_cast<int32_t>(HFLG[I][J]);
      HB[static_cast<size_t>(J) * LD + I] = static_cast<int32_t>(HDST[I][J]);
    }
  }

  // The branch and bound proper needs three free facilities at the root: its
  // leaves are the nodes with two left. One and two are settled here.
  if (N <= 2U) {
    std::vector<uint32_t> P(N);
    std::iota(P.begin(), P.end(), 0U);
    int64_t Z = HostObjective(N, P);

    if (N == 2U) {
      std::vector<uint32_t> Q(2);
      Q[0] = 1U;
      Q[1] = 0U;

      if (HostObjective(N, Q) < Z) {
        Z = HostObjective(N, Q);
        P = Q;
      }
    }

    HAS = P;
    std::cout << "Minimum cost: " << Z << std::endl;
    std::cout << "B&B nodes:    0" << std::endl;
    std::cout << "Permutations: " << (N == 2U ? 2 : 1) << std::endl;

    if (DoPrint)
      Print();

    return 0;
  }

  std::vector<int32_t> Ord;
  ReorderFacilities(LD, LD, HA.data(), HC.data(), Ord);

  std::vector<int32_t> HBest(static_cast<size_t>(N) + 2U, 0);
  int64_t Heur = StartBound(LD, LD, HA.data(), HB.data(), HC.data(),
                            HBest.data());

  std::vector<int32_t> HU(static_cast<size_t>(N) + 3U, 0);
  std::vector<int32_t> HV(static_cast<size_t>(N) + 3U, 0);

  RootReduce(LD, LD, HA.data(), HB.data(), HC.data(), BB_UNENDL, HU.data(),
             HV.data());

  // VEKSUM[L] is where level L's segment of the sorted rows begins, one
  // 1-based offset per level, exactly as in the FORTRAN.
  std::vector<int32_t> HVEKSUM(static_cast<size_t>(N) + 3U, 0);

  {
    int32_t I2 = LD + 1;
    int32_t Acc = 0;

    for (int32_t I = 1; I <= LD - 2; ++I) {
      int32_t Nmk = I2 - I;
      Acc += Nmk * Nmk - Nmk;
      HVEKSUM[I] = Acc;
    }
  }

  // Query the attributes individually rather than filling a whole
  // cudaDeviceProp: cudaGetDeviceProperties costs ~700us on this machine,
  // and host work immediately before a launch inflates the measured time of
  // a short kernel.
  int Dev = 0;
  int SMCount = 0;
  int MaxThreadsPerBlock = 0;
  int WarpSize = 0;

  checkCudaError(cudaGetDevice(&Dev));
  checkCudaError(cudaDeviceGetAttribute(&SMCount,
                   cudaDevAttrMultiProcessorCount, Dev));
  checkCudaError(cudaDeviceGetAttribute(&MaxThreadsPerBlock,
                   cudaDevAttrMaxThreadsPerBlock, Dev));
  checkCudaError(cudaDeviceGetAttribute(&WarpSize,
                   cudaDevAttrWarpSize, Dev));

  // Unlike the three enumerating solvers, this kernel asks for no dynamic
  // shared memory at all: the per-thread workspace is tens of KB and lives
  // in global memory. Occupancy here is decided by registers alone, so the
  // block-width search passes 0 for the shared-memory argument.
  size_t Warp = static_cast<size_t>(WarpSize);
  size_t BlockThreads = 0;
  int BlocksPerSM = 0;

  for (size_t B = Warp; B <= static_cast<size_t>(MaxThreadsPerBlock);
       B += Warp) {
    int Resident = 0;
    checkCudaError(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                     &Resident, BranchAndBound, static_cast<int>(B), 0));

    // Resident == 0 means this width cannot launch at all, so it is never a
    // candidate however good its arithmetic looks.
    if (Resident > 0 &&
        static_cast<size_t>(Resident) * B >=
        static_cast<size_t>(BlocksPerSM) * BlockThreads) {
      BlockThreads = B;
      BlocksPerSM = Resident;
    }
  }

  if (BlockThreads == 0UL) {
    std::cerr << "No usable launch geometry: no block width between " << Warp
      << " and " << MaxThreadsPerBlock << " threads can be resident."
      << std::endl;
    return 1;
  }

  // Words of workspace per thread, laid out by the same function the kernel
  // uses to carve it up, so the two cannot disagree.
  // Frontier capacity. The expansion stops before a level whose children
  // could not fit, so this is a hard bound and not a guess: 8 million ranks
  // is 128 MB per buffer, against a workspace that is larger still.
  size_t FCap = 8388608UL;

  size_t WPT = LayoutWork(NULL, NULL, N);
  // GWS, GLPR, GITC and GSOL are the four allocations that scale with the
  // thread count; everything else is fixed or scales with N alone.
  size_t PerThread = WPT * sizeof(int32_t) +
                     (static_cast<size_t>(N) + 3U) * sizeof(uint64_t) +
                     sizeof(uint64_t) +
                     static_cast<size_t>(N) * sizeof(uint32_t);

  size_t FreeBytes = 0;
  size_t TotalBytes = 0;
  checkCudaError(cudaMemGetInfo(&FreeBytes, &TotalBytes));

  // Three fifths of what is free, less the two frontier buffers, which are
  // sized in nodes rather than threads and so are not part of PerThread. What
  // is left over covers the read-only root data, the CGBN buffers and the
  // driver's own allocations.
  size_t FrontierBytes = 2UL * FCap * sizeof(rank_t);
  size_t Budget = FreeBytes / 5UL * 3UL;

  Budget = Budget > FrontierBytes ? Budget - FrontierBytes : 0UL;

  size_t ByMemory = PerThread ? Budget / PerThread : 0UL;
  size_t Wave = static_cast<size_t>(BlocksPerSM) *
                static_cast<size_t>(SMCount) * BlockThreads;
  size_t NTH = Wave < ByMemory ? Wave : ByMemory;

  if (MaxThreads && NTH > MaxThreads)
    NTH = static_cast<size_t>(MaxThreads);

  NTH = NTH / BlockThreads * BlockThreads;

  if (NTH == 0UL) {
    std::cerr << "Problem size " << N << " needs " << PerThread
      << " byte(s) of workspace per thread; " << Budget
      << " byte(s) of device memory will not hold one block of "
      << BlockThreads << "." << std::endl;
    return 1;
  }

  size_t GridBlocks = NTH / BlockThreads;

  size_t TargetF = UserFrontier ? static_cast<size_t>(UserFrontier)
                                : 32UL * NTH;

  if (TargetF > FCap)
    TargetF = FCap;

  size_t VecBytes = (static_cast<size_t>(N) + 3U) * sizeof(uint64_t);
  size_t MatBytes = static_cast<size_t>(N) * N * sizeof(int32_t);

  // 64 blocks of 256 threads is 512 CGBN instances of one warp each.
  // blockDim.x must be a whole number of instances: CGBN lays an instance's
  // lanes out along threadIdx.x and derives its sync mask from that.
  const uint32_t AccThreads = 256U;
  const uint32_t AccBlocks = 64U;
  const uint32_t NInst = AccBlocks * (AccThreads / CGBN_TPI);

  checkCudaError(cudaMallocManaged((void**) &GA, MatBytes));
  checkCudaError(cudaMallocManaged((void**) &GB, MatBytes));
  checkCudaError(cudaMallocManaged((void**) &GC0, MatBytes));
  checkCudaError(cudaMallocManaged((void**) &GVEKSUM,
                   (static_cast<size_t>(N) + 3U) * sizeof(int32_t)));
  checkCudaError(cudaMallocManaged((void**) &GWS, NTH * WPT * sizeof(int32_t)));
  checkCudaError(cudaMallocManaged((void**) &GLPR, NTH * VecBytes));
  checkCudaError(cudaMallocManaged((void**) &GITC, NTH * sizeof(uint64_t)));
  checkCudaError(cudaMallocManaged((void**) &GSOL,
                   NTH * N * sizeof(uint32_t)));
  checkCudaError(cudaMallocManaged((void**) &GF0, FCap * sizeof(rank_t)));
  checkCudaError(cudaMallocManaged((void**) &GF1, FCap * sizeof(rank_t)));
  checkCudaError(cudaMallocManaged((void**) &GFCNT,
                   sizeof(unsigned long long)));
  checkCudaError(cudaMallocManaged((void**) &GSUM, VecBytes));
  checkCudaError(cudaMallocManaged((void**) &GFACT,
                   (static_cast<size_t>(N) + 1U) * sizeof(cgbn_big_t)));
  checkCudaError(cudaMallocManaged((void**) &GPART,
                   NInst * sizeof(cgbn_big_t)));
  checkCudaError(cudaMallocManaged((void**) &GITOT, sizeof(cgbn_big_t)));
  checkCudaError(cudaMallocManaged((void**) &GACCT, sizeof(cgbn_big_t)));
  checkCudaError(cudaMallocManaged((void**) &GRESID, sizeof(cgbn_big_t)));
  checkCudaError(cudaMallocManaged((void**) &GRSIGN, sizeof(int32_t)));

  (void) memcpy(GA, HA.data(), MatBytes);
  (void) memcpy(GB, HB.data(), MatBytes);
  (void) memcpy(GC0, HC.data(), MatBytes);
  (void) memcpy(GVEKSUM, HVEKSUM.data(),
                (static_cast<size_t>(N) + 3U) * sizeof(int32_t));
  (void) memset(GSUM, 0, VecBytes);
  (void) memset(GLPR, 0, NTH * VecBytes);
  (void) memset(GITC, 0, NTH * sizeof(uint64_t));

  // The expansion starts from the root, whose rank at level 0 is 0. The
  // root's own bound is never tested: a child's bound is never below its
  // parent's, so if the root could be pruned every child prunes, and the
  // accounting comes out the same either way.
  GF0[0] = static_cast<rank_t>(0);

  std::string Permutations = SetupFactorials(N, GFACT);

  SetupInitialCounters(static_cast<uint32_t>(Heur));
  cgbn_error_report_t* Report = NULL;
  checkCudaError(cgbn_error_report_alloc(&Report));

  dim3 ThreadsPerBlock(static_cast<unsigned int>(BlockThreads));
  dim3 NumBlocks(static_cast<unsigned int>(GridBlocks));

  // Order matters: call Timestamp first so it is never short-circuited away.
  bool Timed = Timestamp(&tp_start) == 0;

  // Grow the frontier one level at a time until it is big enough to keep the
  // device busy, or until the next level could not fit, or until the level
  // below would be the leaf level and there is nothing left to expand.
  rank_t* Cur = GF0;
  rank_t* Next = GF1;
  uint64_t NF = 1UL;
  uint32_t Depth = 0U;
  // The largest rank the next level could produce: N * (N-1) * ... down to
  // the level below. Once that would leave 128 bits the frontier stops
  // deepening and the depth-first phase takes over from wherever it is.
  rank_t RankMax = 1;
  const rank_t RankCeil = ~static_cast<rank_t>(0);

  while (NF > 0UL && NF < TargetF && Depth + 3U <= N &&
         NF * (N - Depth) <= FCap && Depth + 1U < MAX_FRONTIER_DEPTH &&
         RankMax <= RankCeil / static_cast<rank_t>(N - Depth)) {
    RankMax *= static_cast<rank_t>(N - Depth);

    *GFCNT = 0ULL;
    ExpandFrontier<<<NumBlocks, ThreadsPerBlock>>>(N, Depth, Cur, NF, Next,
                                                   static_cast<uint64_t>(FCap),
                                                   GFCNT,
                                                   static_cast<uint32_t>(NTH),
                                                   static_cast<uint64_t>(WPT));
    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());

    NF = static_cast<uint64_t>(*GFCNT);

    if (NF > FCap) {
      std::cerr << "Frontier overflow at depth " << (Depth + 1U) << ": "
        << NF << " live node(s) against a capacity of " << FCap << "."
        << std::endl;
      return 1;
    }

    std::swap(Cur, Next);
    ++Depth;
  }

  // One frontier node per task. They are all live, so the queue is handing
  // out real work rather than mostly-dead subtrees, and there are enough of
  // them that the largest one is a small fraction of the whole.
  uint64_t QChunk = 1UL;

  BranchAndBound<<<NumBlocks, ThreadsPerBlock>>>(N, Depth, Cur, NF, QChunk,
                                                 static_cast<uint32_t>(NTH),
                                                 static_cast<uint64_t>(WPT));
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
  Timed = (Timestamp(&tp_end) == 0) && Timed;

  // The reduction kernels are timed separately rather than folded into the
  // search: they are the price of the arbitrary-precision accounting, and
  // the point of this variant is to be able to see what that price is.
  bool AccTimed = Timestamp(&tp_acc_start) == 0;
  AccumulateIterations<<<AccBlocks, AccThreads>>>(GITC,
                                                  static_cast<uint32_t>(NTH),
                                                  GPART, NInst, Report);
  checkCudaError(cudaGetLastError());
  CombineIterations<<<1, CGBN_TPI>>>(GPART, NInst, GITOT, Report);
  checkCudaError(cudaGetLastError());
  SumCounters<<<AccBlocks, AccThreads>>>(GLPR, static_cast<uint32_t>(NTH), N,
                                         GSUM);
  checkCudaError(cudaGetLastError());
  CombinePermutations<<<1, CGBN_TPI>>>(GSUM, GFACT, N, GACCT, GRESID,
                                       GRSIGN, Report);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
  AccTimed = (Timestamp(&tp_acc_end) == 0) && AccTimed;

  if (cgbn_error_report_check(Report)) {
    std::cerr << "CGBN error: " << cgbn_error_string(Report) << std::endl;
    return 1;
  }

  uint64_t Key = MinimumCostKey();
  uint32_t Win = static_cast<uint32_t>(Key & 0xffffffffULL);
  std::vector<int32_t> Loc(static_cast<size_t>(N) + 2U, 0);

  // A thread id of ~0 still in the key means no thread improved on the
  // starting heuristic -- which, the search having proved nothing better
  // exists, makes the heuristic's own permutation the optimum.
  if (Win == 0xffffffffU) {
    for (uint32_t I = 1U; I <= N; ++I)
      Loc[I] = HBest[I];
  } else {
    for (uint32_t I = 1U; I <= N; ++I)
      Loc[I] = static_cast<int32_t>(GSOL[static_cast<size_t>(Win) * N +
                                         (I - 1U)]) + 1;
  }

  // Undo the root relabelling: the search worked on facility Ord[I].
  for (uint32_t I = 1U; I <= N; ++I)
    HAS[Ord[I] - 1] = static_cast<uint32_t>(Loc[I] - 1);

  int64_t Cost = HostObjective(N, HAS);

  // How lopsided the search was. Subtree sizes under a pruning bound differ
  // by orders of magnitude, so the busiest thread, not the average one, sets
  // the wall clock; printing it is the only way to see when the work queue
  // has run out of ways to hide that.
  uint64_t Busiest = 0UL;
  uint64_t Idle = 0UL;

  for (size_t I = 0; I < NTH; ++I) {
    if (GITC[I] > Busiest)
      Busiest = GITC[I];
    if (GITC[I] == 0UL)
      ++Idle;
  }

  std::string Nodes = BigToString(GITOT);
  std::string Accounted = BigToString(GACCT);
  std::string Residual = BigToString(GRESID);
  bool NegResid = (*GRSIGN != 0);
  uint32_t Threads = NumThreads();

  // The search works in reduced arithmetic; this is the same number computed
  // from the untouched input matrices. They must agree.
  if (Cost != static_cast<int64_t>(Key >> 32)) {
    std::cerr << "Reduced objective " << (Key >> 32)
      << " disagrees with the assignment's actual cost " << Cost << "."
      << std::endl;
  }

  checkCudaError(cudaFree(GA));
  checkCudaError(cudaFree(GB));
  checkCudaError(cudaFree(GC0));
  checkCudaError(cudaFree(GVEKSUM));
  checkCudaError(cudaFree(GWS));
  checkCudaError(cudaFree(GLPR));
  checkCudaError(cudaFree(GITC));
  checkCudaError(cudaFree(GSOL));
  checkCudaError(cudaFree(GF0));
  checkCudaError(cudaFree(GF1));
  checkCudaError(cudaFree(GFCNT));
  checkCudaError(cudaFree(GSUM));
  checkCudaError(cudaFree(GFACT));
  checkCudaError(cudaFree(GPART));
  checkCudaError(cudaFree(GITOT));
  checkCudaError(cudaFree(GACCT));
  checkCudaError(cudaFree(GRESID));
  checkCudaError(cudaFree(GRSIGN));
  checkCudaError(cgbn_error_report_free(Report));
  checkCudaError(cudaDeviceReset());

  std::cout << "Minimum cost: " << Cost << std::endl;
  std::cout << "B&B nodes:    " << Nodes << std::endl;
  std::cout << "Permutations: " << Permutations << std::endl;
  std::cout << "Accounted:    " << Accounted << std::endl;
  std::cout << "Residual:     " << (NegResid ? "-" : "") << Residual
    << std::endl;
  std::cout << "Start bound:  " << Heur << std::endl;
  std::cout << "NumThreads:   " << Threads << std::endl;
  std::cout << "Frontier:     " << NF << " live node(s) at depth " << Depth
    << (NF ? " into the depth-first phase" : " - solved breadth-first")
    << std::endl;
  std::cout << "Busiest:      " << Busiest << " node(s), " << Idle
    << " idle thread(s)" << std::endl;

  if (DoPrint)
    Print();

  // A clock failure must not discard the result of the search itself;
  // Timestamp has already said what went wrong on stderr.
  if (Timed)
    PrintTimediff("GPU time", &tp_start, &tp_end);

  if (AccTimed)
    PrintTimediff("CGBN reduction", &tp_acc_start, &tp_acc_end);

  return 0;
}
