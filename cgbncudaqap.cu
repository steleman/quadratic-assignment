// Copyright (c) 2025-2026 Stefan Teleman.
//
// Licensed under the MIT License.
// See https://opensource.org/license/mit
// SPDX-License-Identifier: MIT
//

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
// formats the final accumulator in decimal, below.
#include <gmp.h>
#include <cgbn/cgbn.h>

// A CGBN value is not a scalar: its limbs live across TPI cooperating lanes,
// so an individual thread can neither hold one nor update one atomically.
// That splits the arithmetic of this program in two, and the split is the
// whole design:
//
//   - Per-thread quantities (permutation ranks, the local iteration counter)
//     must stay in types a single thread can own. The widest such type is
//     __uint128_t, and 34! (2.95e38) is the largest factorial below 2^128 --
//     so the rank ceiling rises from cudaqap's 20 to 34.
//   - The one quantity that is genuinely global -- the total iteration count,
//     which is N! and passes 2^64 at N = 21 -- is accumulated in CGBN by a
//     separate cooperative kernel once the search is done.
typedef __uint128_t rank_t;

// One warp per CGBN instance, one 32-bit limb per lane, no padding
// (BITS/32 % TPI == 0). 1024 bits holds every factorial up to 170!, so the
// accumulator is never the binding constraint -- the 128-bit rank is.
static const uint32_t CGBN_TPI  = 32U;
static const uint32_t CGBN_BITS = 1024U;

typedef cgbn_context_t<CGBN_TPI> cgbn_ctx_t;
typedef cgbn_env_t<cgbn_ctx_t, CGBN_BITS> cgbn_env;
typedef cgbn_mem_t<CGBN_BITS> cgbn_big_t;

__managed__ uint32_t* GFLG;
__managed__ uint32_t* GDST;

// Packed argmin key: (cost << 32) | winning global thread id. Packing the
// thread id into the low bits makes a plain 64-bit atomicMin pick the lowest
// cost and break ties deterministically on the lowest thread.
__device__  uint64_t MCK = ~0ULL;

// One slot per thread, holding that thread's best permutation rank. The
// winning thread's slot is what identifies the minimizing assignment.
__managed__ rank_t* GRNK;

// One slot per thread, holding that thread's own uint64_t iteration count.
// This is the per-thread half of the counter; the CGBN reduction below is the
// global half. There is no device-wide 64-bit iteration total any more,
// because at N >= 21 there is no 64-bit value that could hold one.
__managed__ uint64_t* GITC;

// Per-instance partial sums, then the single grand total.
__managed__ cgbn_big_t* GPART;
__managed__ cgbn_big_t* GITOT;
__device__ uint64_t TOT = 0UL;

// Per-thread permutation scratch. Each thread owns the slice
// [TIB * (LSS | 1), ...). The stride must be ODD, not merely padded: banks
// repeat every 32 uint32, so consecutive threads collide with multiplicity
// gcd(stride, 32). A plain LSS + 1 is odd only when LSS is even -- at LSS=15
// it gives stride 16 and a 16-way conflict. LSS | 1 rounds up to odd and is
// never smaller than LSS, so gcd(stride, 32) == 1 for every problem size.
extern __shared__ uint32_t MCD[];

// FACT[I] == I!, in the 128-bit rank type. 34! is the largest factorial that
// fits, which is also the ceiling on the permutation ranking below.
__constant__ rank_t FACT[35];


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

namespace qap {

template<typename _Ty>
__device__ __forceinline__
void swap(_Ty* A, _Ty* B) {
  _Ty T = *A;
  *A = *B;
  *B = T;
}

template<typename _Ty>
__device__
void reverse(_Ty* AX, int64_t B, int64_t E) {
  while (B < E) {
    swap(&AX[B], &AX[E]);
    ++B;
    --E;
  }
}

template<typename _Ty>
__device__ __forceinline__
_Ty array_index(const _Ty* AX, uint32_t IXW, uint32_t IXH, uint32_t W) {
  const _Ty* AXI = static_cast<const _Ty*>(AX + IXW * W);
  const _Ty* AXP = AXI + IXH;
  return *AXP;
}

// Thread-private, exactly as std::next_permutation. AX points at this
// thread's own slice, so no atomics or fences are involved.
template<typename _Ty>
__device__
bool next_permutation(_Ty* AX, uint32_t N) {
  if (N < 2U)
    return false;

  int64_t K = static_cast<int64_t>(N - 2U);
  int64_t J = static_cast<int64_t>(N - 1U);

  while (K >= 0 && AX[K] >= AX[K + 1])
    --K;

  if (K < 0)
    return false;

  while (J >= 0 && AX[J] <= AX[K])
    --J;

  swap(&AX[K], &AX[J]);
  reverse(AX, K + 1, N - 1);
  return true;
}

// Lexicographic unranking via the factorial number system: fill P with the
// Rank-th permutation of [0, N) counting from 0. This is what lets a thread
// jump straight to the start of its slice of the permutation space.
template<typename _Ty>
__device__
void unrank(_Ty* P, uint32_t N, rank_t Rank) {
  for (uint32_t I = 0; I < N; ++I)
    P[I] = static_cast<_Ty>(I);

  for (uint32_t I = 0; I < N; ++I) {
    rank_t F = FACT[N - 1U - I];
    uint32_t D = static_cast<uint32_t>(Rank / F);
    Rank %= F;

    _Ty T = P[I + D];
    for (uint32_t K = I + D; K > I; --K)
      P[K] = P[K - 1];

    P[I] = T;
  }
}

} // namespace qap

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
// established that both matrices are N x N.
void SetupDeviceVectors(uint32_t N,
                        const std::vector<std::vector<uint32_t>>& HFLG,
                        const std::vector<std::vector<uint32_t>>& HDST) {
  size_t Bytes = static_cast<size_t>(N) * N * sizeof(uint32_t);

  checkCudaError(cudaMallocManaged((void**) &GFLG, Bytes));
  checkCudaError(cudaMallocManaged((void**) &GDST, Bytes));

  uint32_t* XA = (uint32_t*) malloc(Bytes);
  assert(XA && "malloc(3C) failed: could not allocate array");

  uint32_t* XAP = XA;
  uint32_t XAS = 0U;
  for (std::vector<std::vector<uint32_t>>::const_iterator I = HFLG.begin();
       I != HFLG.end(); ++I) {
    const std::vector<uint32_t>& HV = *I;
    (void) memcpy(XAP, HV.data(), HV.size() * sizeof(uint32_t));
    XAP += HV.size();
    XAS += HV.size() * sizeof(uint32_t);
  }

  checkCudaError(cudaMemcpy(GFLG, XA, XAS, cudaMemcpyHostToDevice));

  XAS = 0U;
  (void) memset(XA, 0, Bytes);
  XAP = XA;

  for (std::vector<std::vector<uint32_t>>::const_iterator I = HDST.begin();
       I != HDST.end(); ++I) {
    const std::vector<uint32_t>& HV = *I;
    (void) memcpy(XAP, HV.data(), HV.size() * sizeof(uint32_t));
    XAP += HV.size();
    XAS += HV.size() * sizeof(uint32_t);
  }

  checkCudaError(cudaMemcpy(GDST, XA, XAS, cudaMemcpyHostToDevice));

  // HAS is sized here and later overwritten with the minimizing assignment.
  HAS.resize(HFLG.size());
  std::iota(HAS.begin(), HAS.end(), 0U);

  free(XA);
}

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

// Scores the permutation P. This is the direct analogue
// of QAP::ComputeCost in qap.cpp and must agree with it element for element.
__device__ __forceinline__
uint32_t ComputeCost(const uint32_t* P, uint32_t LSS) {
  uint32_t Cost = 0U;

  for (uint32_t I = 0; I < LSS; ++I) {
    uint32_t PI = P[I];

    for (uint32_t J = 0; J < LSS; ++J) {
      Cost += qap::array_index(GFLG, I, J, LSS) *
              qap::array_index(GDST, PI, P[J], LSS);
    }
  }

  return Cost;
}

// Each thread takes a contiguous half-open range [Lo, Hi) of lexicographic
// permutation ranks, unranks Lo into its own shared slice, and walks forward
// with next_permutation. The ranges tile [0, Total) exactly, so every
// permutation is visited by exactly one thread.
//
// Chunk and Rem come in as arguments rather than being recomputed here: a
// 128-bit division is a called helper routine, not an instruction, and there
// is no reason to pay for one per thread when the host already has the answer.
//
// All shared state is initialized here, inside the kernel: shared memory is
// per-block and does not survive a launch, so it cannot be seeded elsewhere.
__global__ void QuadraticAssignment(uint32_t LSS, rank_t Chunk, uint64_t Rem,
                                    rank_t* RNK, uint64_t* ITC) {
  __shared__ uint64_t __align__(8) BKEY;

  uint32_t TIB = threadIdx.y * blockDim.x + threadIdx.x;
  uint32_t BTH = blockDim.x * blockDim.y;

  if (TIB == 0U)
    BKEY = ~0ULL;

  __syncthreads();

  // Uniform across the block, so no thread diverges past the barriers.
  if (LSS >= 1U) {
    uint32_t BID = blockIdx.y * gridDim.x + blockIdx.x;
    uint64_t NTH = static_cast<uint64_t>(gridDim.x) * gridDim.y * BTH;
    uint64_t GTID = static_cast<uint64_t>(BID) * BTH + TIB;

    if (GTID == 0UL)
      (void) atomicAdd((unsigned long long*) &TOT,
                       static_cast<unsigned long long>(NTH));

    rank_t Lo = static_cast<rank_t>(GTID) * Chunk +
                static_cast<rank_t>(GTID < Rem ? GTID : Rem);
    rank_t Hi = Lo + Chunk + static_cast<rank_t>(GTID < Rem ? 1UL : 0UL);

    RNK[GTID] = ~static_cast<rank_t>(0);
    ITC[GTID] = 0UL;

    // Threads past the end of the space when Total < NTH.
    if (Lo < Hi) {
      uint32_t* P = &MCD[TIB * (LSS | 1U)];
      uint32_t TMC = ~0U;
      rank_t TRK = Lo;

      // The thread-local iteration counter. Incremented once per permutation
      // actually scored -- not derived from Hi - Lo -- so that it counts what
      // the loop did rather than what it was asked to do.
      uint64_t LIT = 0UL;

      qap::unrank(P, LSS, Lo);

      for (rank_t R = Lo; R < Hi; ++R) {
        uint32_t CC = ComputeCost(P, LSS);
        ++LIT;

        // Record the rank alongside the cost. The permutation itself is
        // recovered from the rank on the host, so nothing wide has to be
        // carried through the reduction.
        if (CC < TMC) {
          TMC = CC;
          TRK = R;
        }

        if (!qap::next_permutation(P, LSS))
          break;
      }

      RNK[GTID] = TRK;

      // The thread is finished: hand its local count to the global counter.
      // The handoff is a plain store rather than an atomic add, because the
      // global counter is a CGBN value and no single thread can add into one;
      // AccumulateIterations below performs the addition cooperatively.
      ITC[GTID] = LIT;

      uint64_t Key = (static_cast<uint64_t>(TMC) << 32) |
                     static_cast<uint64_t>(static_cast<uint32_t>(GTID));

      (void) atomicMin((unsigned long long*) &BKEY,
                       static_cast<unsigned long long>(Key));
    }
  }

  __syncthreads();

  // One global atomic per block rather than one per permutation.
  if (TIB == 0U && BKEY != ~0ULL)
    (void) atomicMin((unsigned long long*) &MCK,
                     static_cast<unsigned long long>(BKEY));
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

// Stage one of the global iteration count: NINST CGBN instances, each one
// warp, sum a strided share of the per-thread counters. Striding rather than
// blocking keeps every instance busy without a division; integer addition is
// associative, so the order the counts are visited in does not affect the sum.
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
// NInst is a few hundred at most, so this is a handful of microseconds and
// needs no lock, no second reduction level, and no host arithmetic.
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

// Same job for the 128-bit rank type, which iostreams cannot print either.
std::string RankToString(rank_t V) {
  if (V == static_cast<rank_t>(0))
    return std::string("0");

  char B[64];
  size_t I = sizeof(B);

  B[--I] = '\0';

  while (V != static_cast<rank_t>(0)) {
    B[--I] = static_cast<char>('0' + static_cast<uint32_t>(V % 10U));
    V /= 10U;
  }

  return std::string(&B[I]);
}

uint64_t MinimumCostKey() {
  uint64_t LMC;
  checkCudaError(cudaMemcpyFromSymbol(&LMC, MCK, sizeof(uint64_t), 0,
                                      cudaMemcpyDeviceToHost));
  return LMC;
}

// Host mirror of qap::unrank, used once to turn the winning thread's rank
// back into the assignment vector.
void HostUnrank(std::vector<uint32_t>& P, uint32_t N, rank_t Rank,
                const rank_t* Fact) {
  P.resize(N);

  for (uint32_t I = 0; I < N; ++I)
    P[I] = I;

  for (uint32_t I = 0; I < N; ++I) {
    rank_t F = Fact[N - 1U - I];
    uint32_t D = static_cast<uint32_t>(Rank / F);
    Rank %= F;

    uint32_t T = P[I + D];
    for (uint32_t K = I + D; K > I; --K)
      P[K] = P[K - 1];

    P[I] = T;
  }
}

uint32_t NumThreads() {
  uint64_t LTOT;
  checkCudaError(cudaMemcpyFromSymbol(&LTOT, TOT, sizeof(uint64_t), 0,
                                      cudaMemcpyDeviceToHost));
  return LTOT;
}

void SetupInitialCounters() {
  uint64_t IP = 0UL;
  uint64_t CP = std::numeric_limits<uint64_t>::max();

  checkCudaError(cudaMemcpyToSymbol(MCK, &CP, sizeof(uint64_t), 0,
                                    cudaMemcpyHostToDevice));
  checkCudaError(cudaMemcpyToSymbol(TOT, &IP, sizeof(uint64_t), 0,
                                    cudaMemcpyHostToDevice));
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
  std::cerr << "Usage: cgbncudaqap [-h | --help]" << std::endl;
  std::cerr << "           [-s <input-file> | --single-file <input-file>]"
    << std::endl;
  std::cerr << "           (use single-file matrix input format)." << std::endl;
  std::cerr << "           [ -f <flow-input-file> | --flinput <flow-input-file>]"
    << std::endl;
  std::cerr << "           [ -d <distance-input-file> | --dstinput <distance-input-file>]"
    << std::endl;
  std::cerr << "           [ -m <gpu-shared-memory-size> | --shmem-size <gpu-shared-memory-size>]" << std::endl;
  std::cerr << "           (minimum GPU dynamic shared memory per block, in KB."
    << std::endl;
  std::cerr << "            default is 0 - allocate exactly what the assignment"
    << std::endl;
  std::cerr << "            vector requires)." << std::endl;
  std::cerr << "           [-p | --print] (print vector contents)." << std::endl;
}

static struct option long_options[] = {
  { "help",             no_argument,        0,  'h' },
  { "print",            no_argument,        0,  'p' },
  { "single-file",      required_argument,  0,  's' },
  { "flinput",          required_argument,  0,  'f' },
  { "dstinput",         required_argument,  0,  'd' },
  { "shmem-size",       required_argument,  0,  'm' },
  { 0,                  0,                  0,   0  }
};

int main(int argc, char* const argv[]) {
  int C;
  int OIx = 0;
  std::string FLGFile;
  std::string DSTFile;
  std::string SFile;
  uint32_t ShmemSize = 0U;
  bool DoPrint = false;

  while (1) {
    C = getopt_long(argc, argv, "hps:f:d:m:", long_options, &OIx);
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
    case 'm':
      ShmemSize = static_cast<uint32_t>(std::stoul(optarg));
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

  SetupDeviceVectors(static_cast<uint32_t>(HFLG.size()), HFLG, HDST);
  SetupInitialCounters();

  uint32_t LSS = static_cast<uint32_t>(HAS.size());

  // Ranking the permutation space needs LSS! to fit in a rank_t. That is the
  // one place a 128-bit scalar is unavoidable: a rank is per-thread state, and
  // CGBN values cannot be per-thread. 34! is 2.95e38, just under 2^128.
  if (LSS > 34U) {
    std::cerr << "Problem size " << LSS << " is too large: the permutation "
      << "rank does not fit in 128 bits (the limit is 34)." << std::endl;
    return 1;
  }

  rank_t HFACT[35];
  HFACT[0] = static_cast<rank_t>(1);

  for (uint32_t I = 1U; I <= 34U; ++I)
    HFACT[I] = HFACT[I - 1U] * static_cast<rank_t>(I);

  checkCudaError(cudaMemcpyToSymbol(FACT, HFACT, sizeof(HFACT), 0,
                                    cudaMemcpyHostToDevice));

  rank_t Total = HFACT[LSS];

  // Query the five attributes individually rather than filling a whole
  // cudaDeviceProp: cudaGetDeviceProperties costs ~700us on this machine,
  // which is several times the entire kernel at small N, and host work
  // immediately before a launch inflates the measured time of a short kernel.
  // These five attribute queries together cost well under a microsecond.
  int Dev = 0;
  int SMCount = 0;
  int MaxThreadsPerBlock = 0;
  int ShmemPerBlock = 0;
  int WarpSize = 0;
  int ReservedShmem = 0;

  checkCudaError(cudaGetDevice(&Dev));
  checkCudaError(cudaDeviceGetAttribute(&SMCount,
                   cudaDevAttrMultiProcessorCount, Dev));
  checkCudaError(cudaDeviceGetAttribute(&MaxThreadsPerBlock,
                   cudaDevAttrMaxThreadsPerBlock, Dev));
  checkCudaError(cudaDeviceGetAttribute(&ShmemPerBlock,
                   cudaDevAttrMaxSharedMemoryPerBlock, Dev));
  checkCudaError(cudaDeviceGetAttribute(&WarpSize,
                   cudaDevAttrWarpSize, Dev));
  checkCudaError(cudaDeviceGetAttribute(&ReservedShmem,
                   cudaDevAttrReservedSharedMemoryPerBlock, Dev));

  // Each thread owns one padded LSS-element slice of MCD, so shared memory
  // scales with the block size and the problem size together. That coupling
  // is what makes a single hardcoded geometry wrong: a wide block starves
  // occupancy at large N, and a large grid is pure overhead at small N.
  size_t SliceBytes = static_cast<size_t>(LSS | 1U) * sizeof(uint32_t);
  size_t ShmemFloor = static_cast<size_t>(ShmemSize) * 1024UL;

  // A block cannot use all of sharedMemPerBlock as dynamic shared memory:
  // the driver reserves a slice of it, so asking for the full figure makes
  // the block unlaunchable rather than merely tight.
  size_t MaxShmem = static_cast<size_t>(ShmemPerBlock) -
                    static_cast<size_t>(ReservedShmem);

  if (ShmemFloor > MaxShmem) {
    std::cerr << "Requested " << ShmemFloor << " byte(s) of dynamic shared "
      << "memory per block, but this device provides at most " << MaxShmem
      << "." << std::endl;
    return 1;
  }

  // Widest block shared memory and the device will carry, capped by the
  // amount of work actually available, rounded to whole warps.
  size_t Warp = static_cast<size_t>(WarpSize);
  size_t MaxBlock = MaxShmem / SliceBytes;

  if (MaxBlock > static_cast<size_t>(MaxThreadsPerBlock))
    MaxBlock = static_cast<size_t>(MaxThreadsPerBlock);

  if (static_cast<rank_t>(MaxBlock) > Total)
    MaxBlock = static_cast<size_t>(Total);

  MaxBlock = ((MaxBlock + Warp - 1UL) / Warp) * Warp;

  if (MaxBlock > static_cast<size_t>(MaxThreadsPerBlock))
    MaxBlock = static_cast<size_t>(MaxThreadsPerBlock);

  // Pick the block width that keeps the most threads resident per SM. The
  // occupancy API accounts for registers and shared memory together, which
  // is not something that can be predicted from SliceBytes alone.
  size_t BlockThreads = 0;
  size_t ShmemBytes = 0;
  int BlocksPerSM = 0;

  for (size_t B = Warp; B <= MaxBlock; B += Warp) {
    size_t Sh = B * SliceBytes;

    if (Sh < ShmemFloor)
      Sh = ShmemFloor;

    if (Sh > MaxShmem)
      break;

    int Resident = 0;
    checkCudaError(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                     &Resident, QuadraticAssignment,
                     static_cast<int>(B), Sh));

    // Resident == 0 means this width cannot launch at all - typically the
    // register budget rather than shared memory - so it is never a candidate.
    // Otherwise >= rather than >: on a tie in resident threads take the wider
    // block, since it needs fewer blocks and so fewer scheduled blocks and
    // fewer global atomics in the final reduction.
    if (Resident > 0 &&
        static_cast<size_t>(Resident) * B >=
        static_cast<size_t>(BlocksPerSM) * BlockThreads) {
      BlockThreads = B;
      ShmemBytes = Sh;
      BlocksPerSM = Resident;
    }
  }

  if (BlockThreads == 0UL) {
    std::cerr << "No usable launch geometry for problem size " << LSS
      << ": no block width between " << Warp << " and " << MaxBlock
      << " threads can be resident, given a " << SliceBytes
      << "-byte slice per thread";

    if (ShmemFloor > 0UL)
      std::cerr << " and the " << ShmemFloor << "-byte floor from -m";

    std::cerr << "." << std::endl;
    return 1;
  }

  // One full wave: every thread resident at once, each with an equal share of
  // the permutation space, so they all finish together. More blocks than a
  // wave would only queue behind the first, and fewer would idle SMs.
  size_t WaveBlocks = static_cast<size_t>(BlocksPerSM) *
                      static_cast<size_t>(SMCount);
  // Computed in rank_t and only then narrowed: at large LSS the block count a
  // one-permutation-per-thread tiling would need overflows size_t long before
  // it overflows the rank type.
  rank_t NeedBlocks = (Total + static_cast<rank_t>(BlockThreads) -
                       static_cast<rank_t>(1)) /
                      static_cast<rank_t>(BlockThreads);
  size_t GridBlocks = NeedBlocks < static_cast<rank_t>(WaveBlocks)
                        ? static_cast<size_t>(NeedBlocks) : WaveBlocks;

  if (GridBlocks == 0UL)
    GridBlocks = 1UL;

  size_t NTH = GridBlocks * BlockThreads;

  // The argmin key packs the thread id into its low 32 bits.
  if (NTH > 0xffffffffUL) {
    std::cerr << "Launch geometry of " << NTH << " threads exceeds the "
      << "32-bit thread id packed into the argmin key." << std::endl;
    return 1;
  }

  // Chunk is what one thread walks. It is handed to the per-thread uint64_t
  // iteration counter, so it has to fit in one; that is a far looser bound
  // than the 128-bit rank (it first bites near LSS = 25) and it is checked
  // rather than left to wrap silently.
  rank_t Chunk = Total / static_cast<rank_t>(NTH);
  uint64_t Rem = static_cast<uint64_t>(Total % static_cast<rank_t>(NTH));

  if (Chunk + static_cast<rank_t>(1) >
      static_cast<rank_t>(std::numeric_limits<uint64_t>::max())) {
    std::cerr << "Problem size " << LSS << " gives " << RankToString(Chunk)
      << " permutations per thread, which does not fit the 64-bit per-thread "
      << "iteration counter." << std::endl;
    return 1;
  }

  dim3 ThreadsPerBlock(static_cast<unsigned int>(BlockThreads));
  dim3 NumBlocks(static_cast<unsigned int>(GridBlocks));

  // 64 blocks of 256 threads is 512 CGBN instances of one warp each. blockDim.x
  // must be a whole number of instances: CGBN lays an instance's lanes out
  // along threadIdx.x and derives its sync mask from that. Instances past the
  // end of ITC simply sum nothing and store zero.
  const uint32_t AccThreads = 256U;
  const uint32_t AccBlocks = 64U;
  const uint32_t NInst = AccBlocks * (AccThreads / CGBN_TPI);

  checkCudaError(cudaMallocManaged((void**) &GRNK, NTH * sizeof(rank_t)));
  checkCudaError(cudaMallocManaged((void**) &GITC, NTH * sizeof(uint64_t)));
  checkCudaError(cudaMallocManaged((void**) &GPART, NInst * sizeof(cgbn_big_t)));
  checkCudaError(cudaMallocManaged((void**) &GITOT, sizeof(cgbn_big_t)));

  cgbn_error_report_t* Report = NULL;
  checkCudaError(cgbn_error_report_alloc(&Report));

  // Order matters: call Timestamp first so it is never short-circuited away.
  bool Timed = Timestamp(&tp_start) == 0;
  QuadraticAssignment<<<NumBlocks, ThreadsPerBlock, ShmemBytes>>>(LSS, Chunk,
                                                                 Rem, GRNK,
                                                                 GITC);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
  Timed = (Timestamp(&tp_end) == 0) && Timed;

  // The two reduction kernels are timed separately rather than folded into the
  // search: they are the price of the arbitrary-precision counter, and the
  // point of this variant is to be able to see what that price is.
  bool AccTimed = Timestamp(&tp_acc_start) == 0;
  AccumulateIterations<<<AccBlocks, AccThreads>>>(GITC,
                                                  static_cast<uint32_t>(NTH),
                                                  GPART, NInst, Report);
  checkCudaError(cudaGetLastError());
  CombineIterations<<<1, CGBN_TPI>>>(GPART, NInst, GITOT, Report);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
  AccTimed = (Timestamp(&tp_acc_end) == 0) && AccTimed;

  if (cgbn_error_report_check(Report)) {
    std::cerr << "CGBN error: " << cgbn_error_string(Report) << std::endl;
    return 1;
  }

  uint64_t Key = MinimumCostKey();

  std::cout << "Minimum cost: " << (Key >> 32) << std::endl;
  std::cout << "Iterations:   " << BigToString(GITOT) << std::endl;
  std::cout << "Permutations: " << RankToString(Total) << std::endl;
  std::cout << "NumThreads:   " << NumThreads() << std::endl;

  // Recover the minimizing assignment from the winning thread's best rank,
  // before cudaDeviceReset() takes the managed buffer away.
  if (Key != ~0ULL) {
    uint32_t Win = static_cast<uint32_t>(Key & 0xffffffffULL);

    if (Win < NTH && GRNK[Win] != ~static_cast<rank_t>(0))
      HostUnrank(HAS, LSS, GRNK[Win], HFACT);
  }

  checkCudaError(cudaFree(GRNK));
  checkCudaError(cudaFree(GITC));
  checkCudaError(cudaFree(GPART));
  checkCudaError(cudaFree(GITOT));
  checkCudaError(cgbn_error_report_free(Report));
  checkCudaError(cudaDeviceReset());

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

