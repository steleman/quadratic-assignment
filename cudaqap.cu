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

__managed__ uint32_t* GFLG;
__managed__ uint32_t* GDST;
__device__  uint64_t IT = 0UL;
// Packed argmin key: (cost << 32) | winning global thread id. Packing the
// thread id into the low bits makes a plain 64-bit atomicMin pick the lowest
// cost and break ties deterministically on the lowest thread.
__device__  uint64_t MCK = ~0ULL;

// One slot per thread, holding that thread's best permutation rank. The
// winning thread's slot is what identifies the minimizing assignment.
__managed__ uint64_t* GRNK;
__device__ uint64_t TOT = 0UL;
// Per-thread permutation scratch. Each thread owns the slice
// [TIB * (LSS | 1), ...). The stride must be ODD, not merely padded: banks
// repeat every 32 uint32, so consecutive threads collide with multiplicity
// gcd(stride, 32). A plain LSS + 1 is odd only when LSS is even -- at LSS=15
// it gives stride 16 and a 16-way conflict. LSS | 1 rounds up to odd and is
// never smaller than LSS, so gcd(stride, 32) == 1 for every problem size.
extern __shared__ uint32_t MCD[];

// FACT[I] == I!. 20! is the largest factorial that fits in a uint64_t, which
// is also the ceiling on the permutation ranking below.
__constant__ uint64_t FACT[21];


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
void unrank(_Ty* P, uint32_t N, uint64_t Rank) {
  for (uint32_t I = 0; I < N; ++I)
    P[I] = static_cast<_Ty>(I);

  for (uint32_t I = 0; I < N; ++I) {
    uint64_t F = FACT[N - 1U - I];
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

static void PrintTimediff(const struct timespec* S,
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
  std::cout << "GPU time: " << Sgn << Nns / 1000000000L << '.'
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
// All shared state is initialized here, inside the kernel: shared memory is
// per-block and does not survive a launch, so it cannot be seeded elsewhere.
__global__ void QuadraticAssignment(uint32_t LSS, uint64_t Total,
                                    uint64_t* RNK) {
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

    uint64_t Chunk = Total / NTH;
    uint64_t Rem = Total % NTH;
    uint64_t Lo = GTID * Chunk + (GTID < Rem ? GTID : Rem);
    uint64_t Hi = Lo + Chunk + (GTID < Rem ? 1UL : 0UL);

    RNK[GTID] = ~0ULL;

    // Threads past the end of the space when Total < NTH.
    if (Lo < Hi) {
      uint32_t* P = &MCD[TIB * (LSS | 1U)];
      uint32_t TMC = ~0U;
      uint64_t TRK = Lo;

      qap::unrank(P, LSS, Lo);

      for (uint64_t R = Lo; R < Hi; ++R) {
        uint32_t CC = ComputeCost(P, LSS);

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

      uint64_t Key = (static_cast<uint64_t>(TMC) << 32) |
                     static_cast<uint64_t>(static_cast<uint32_t>(GTID));

      (void) atomicAdd((unsigned long long*) &IT,
                       static_cast<unsigned long long>(Hi - Lo));
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

uint64_t Iterations() {
  uint64_t LIT;
  checkCudaError(cudaMemcpyFromSymbol(&LIT, IT, sizeof(uint64_t), 0,
                                      cudaMemcpyDeviceToHost));
  return LIT;
}

uint64_t MinimumCostKey() {
  uint64_t LMC;
  checkCudaError(cudaMemcpyFromSymbol(&LMC, MCK, sizeof(uint64_t), 0,
                                      cudaMemcpyDeviceToHost));
  return LMC;
}

// Host mirror of qap::unrank, used once to turn the winning thread's rank
// back into the assignment vector.
void HostUnrank(std::vector<uint32_t>& P, uint32_t N, uint64_t Rank,
                const uint64_t* Fact) {
  P.resize(N);

  for (uint32_t I = 0; I < N; ++I)
    P[I] = I;

  for (uint32_t I = 0; I < N; ++I) {
    uint64_t F = Fact[N - 1U - I];
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
  checkCudaError(cudaMemcpyToSymbol(IT, &IP, sizeof(uint64_t), 0,
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
  std::cerr << "Usage: cudaqap [-h | --help]" << std::endl;
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

  // Ranking the permutation space needs LSS! to fit in a uint64_t.
  if (LSS > 20U) {
    std::cerr << "Problem size " << LSS << " is too large: the permutation "
      << "count does not fit in 64 bits (the limit is 20)." << std::endl;
    return 1;
  }

  uint64_t HFACT[21];
  HFACT[0] = 1UL;

  for (uint32_t I = 1U; I <= 20U; ++I)
    HFACT[I] = HFACT[I - 1U] * static_cast<uint64_t>(I);

  checkCudaError(cudaMemcpyToSymbol(FACT, HFACT, sizeof(HFACT), 0,
                                    cudaMemcpyHostToDevice));

  uint64_t Total = HFACT[LSS];

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

  if (static_cast<uint64_t>(MaxBlock) > Total)
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
  size_t NeedBlocks = static_cast<size_t>((Total + BlockThreads - 1UL) /
                                          BlockThreads);
  size_t GridBlocks = NeedBlocks < WaveBlocks ? NeedBlocks : WaveBlocks;

  if (GridBlocks == 0UL)
    GridBlocks = 1UL;

  size_t NTH = GridBlocks * BlockThreads;

  // The argmin key packs the thread id into its low 32 bits.
  if (NTH > 0xffffffffUL) {
    std::cerr << "Launch geometry of " << NTH << " threads exceeds the "
      << "32-bit thread id packed into the argmin key." << std::endl;
    return 1;
  }

  dim3 ThreadsPerBlock(static_cast<unsigned int>(BlockThreads));
  dim3 NumBlocks(static_cast<unsigned int>(GridBlocks));

  checkCudaError(cudaMallocManaged((void**) &GRNK, NTH * sizeof(uint64_t)));

  // Order matters: call Timestamp first so it is never short-circuited away.
  bool Timed = Timestamp(&tp_start) == 0;
  QuadraticAssignment<<<NumBlocks, ThreadsPerBlock, ShmemBytes>>>(LSS, Total,
                                                                 GRNK);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
  Timed = (Timestamp(&tp_end) == 0) && Timed;

  uint64_t Key = MinimumCostKey();

  std::cout << "Minimum cost: " << (Key >> 32) << std::endl;
  std::cout << "Iterations:   " << Iterations() << std::endl;
  std::cout << "NumThreads:   " << NumThreads() << std::endl;

  // Recover the minimizing assignment from the winning thread's best rank,
  // before cudaDeviceReset() takes the managed buffer away.
  if (Key != ~0ULL) {
    uint32_t Win = static_cast<uint32_t>(Key & 0xffffffffULL);

    if (Win < NTH && GRNK[Win] != ~0ULL)
      HostUnrank(HAS, LSS, GRNK[Win], HFACT);
  }

  checkCudaError(cudaFree(GRNK));
  checkCudaError(cudaDeviceReset());

  if (DoPrint)
    Print();

  // A clock failure must not discard the result of the search itself;
  // Timestamp has already said what went wrong on stderr.
  if (Timed)
    PrintTimediff(&tp_start, &tp_end);

  return 0;
}

