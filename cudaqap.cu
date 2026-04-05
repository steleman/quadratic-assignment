// Copyright (c) 2025-2026 Stefan Teleman.
//
// Licensed under the MIT License.
// See https://opensource.org/license/mit
// SPDX-License-Identifier: MIT
//

#include <iostream>
#include <iomanip>
#include <vector>
#include <set>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
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
#include <cuda/atomic>

__managed__ uint32_t* GFLG;
__managed__ uint32_t* GDST;
__managed__ uint32_t* GAS;
__device__  uint32_t DSS = 0U;
__device__  uint64_t IT = 0UL;
__device__  uint64_t MC = 0UL;
__device__  uint32_t H = 0U;
__device__  uint32_t W = 0U;
__device__ uint64_t HIT = 0UL;
__device__ uint64_t HMC = 0UL;
__device__ uint64_t TOT = 0UL;
__shared__ uint32_t MCDI;
extern __shared__ uint32_t MCD[];
__shared__ uint64_t __align__(8) XMC;

uint32_t* HGFLG;
uint32_t* HGDST;
uint32_t* HGAS;
uint32_t HSS;
uint32_t HH;
uint32_t HW;
uint64_t GMC;
uint64_t GIT;

std::vector<std::vector<uint32_t>> HFLG;
std::vector<std::vector<uint32_t>> HDST;
std::vector<uint32_t> HAS;

uint32_t AH = std::numeric_limits<uint32_t>::max();
uint32_t AW = std::numeric_limits<uint32_t>::max();

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
  (void) __nv_atomic_exchange(A, B, B, __ATOMIC_SEQ_CST);
}

template<typename _Ty>
__device__
_Ty min(_Ty A, _Ty B) {
  return A < B ? A : B;
}

template<typename _Ty>
__device__
void reverse(_Ty* AX, int32_t B, int32_t E) {
  while (B < E) {
    swap(&AX[B], &AX[E]);
    ++B;
    --E;
  }

  __syncthreads();
}

template<typename _Ty>
__device__ __forceinline__
_Ty array_index(const _Ty* AX, uint32_t IXW, uint32_t IXH, uint32_t W) {
  const _Ty* AXI = static_cast<const _Ty*>(AX + IXW * W);
  const _Ty* AXP = AXI + IXH;
  return *AXP;
}

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

} // namespace qap

static struct timespec tp_start;
static struct timespec tp_end;

static int Timestamp(struct timespec* ts) {
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

  int64_t Sec = ((int64_t) E->tv_sec - (int64_t) S->tv_sec) * 1000000000;
  int64_t Nns = ((int64_t) E->tv_nsec - (int64_t) S->tv_nsec) / 1000000;
  Nns = labs(Nns);

  std::cout << "CPU Clock Resolution: " << (int64_t) R.tv_sec << '.'
    << std::setfill('0') << std::setw(9) << (int64_t) R.tv_nsec
    << '.' << std::endl;
  std::cout << "GPU time: " << (int64_t) Sec / 1000000000L << '.'
    << std::setfill('0') << std::setw(12) << (int64_t) Nns << " second(s)."
    << std::endl;
}

void ReadFromFile(const std::string& FileName,
                  std::vector<std::vector<uint32_t>>& IV) {
  std::ifstream IFS(FileName);
  if (IFS.good()) {
    std::vector<std::vector<uint32_t>> LV;
    std::string Line;
    const char* Sep = ", \t\n";
    char* End;
    uint32_t MX = 0U;
    uint32_t LX;

    while (std::getline(IFS, Line)) {
      if (!Line.empty()) {
        LX = 0U;
        std::vector<uint32_t> V;
        if (char* Tok = strtok_r(const_cast<char*>(Line.data()), Sep, &End)) {
          V.push_back(static_cast<uint32_t>(std::stoul(Tok)));
          ++LX;
        }

        while (char* Tok = strtok_r(NULL, Sep, &End)) {
          V.push_back(static_cast<uint32_t>(std::stoul(Tok)));
          ++LX;
        }

        IV.push_back(V);
      }
    }

    IFS.close();

    uint32_t IVS = static_cast<uint32_t>(IV.size());
    uint32_t* LH;
    uint32_t* LW;

    checkCudaError(cudaGetSymbolAddress((void**) &LH, H));
    checkCudaError(cudaGetSymbolAddress((void**) &LW, W));
    checkCudaError(cudaMemcpyToSymbol(H, &MX, sizeof(MX), 0, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpyToSymbol(W, &IVS, sizeof(IVS), 0, cudaMemcpyHostToDevice));
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
    uint32_t MX = 0U;
    uint32_t LX;

    while (std::getline(IFS, Line)) {
      if (!Line.empty()) {
        if (char* Tok = strtok_r(const_cast<char*>(Line.data()), Sep, &End))
          N = static_cast<uint32_t>(std::stoul(Tok));
        while (strtok_r(NULL, Sep, &End));
        break;
      }
    }

    uint32_t LW;
    uint32_t LH;

    checkCudaError(cudaGetSymbolAddress((void**) &LW, W));
    checkCudaError(cudaGetSymbolAddress((void**) &LH, H));
    checkCudaError(cudaMemcpyToSymbol(W, &N, sizeof(N), 0, cudaMemcpyHostToDevice));

    std::vector<uint32_t> V;
    C = 0U;

    while (std::getline(IFS, Line) && C++ <= N) {
      if (!Line.empty()) {
        V.clear();
        LX = 0U;
        if (char* Tok = strtok_r(const_cast<char*>(Line.data()), Sep, &End)) {
          V.push_back(static_cast<uint32_t>(std::stoul(Tok)));
          ++LX;
        }

        while (char* Tok = strtok_r(NULL, Sep, &End)) {
          V.push_back(static_cast<uint32_t>(std::stoul(Tok)));
          ++LX;
        }

        HFLG.push_back(V);
      }
    }


    V.clear();
    C = 0U;
    MX = 0U;

    while (std::getline(IFS, Line) && C++ <= N) {
      if (!Line.empty()) {
        V.clear();
        LX = 0U;
        if (char* Tok = strtok_r(const_cast<char*>(Line.data()), Sep, &End)) {
          V.push_back(static_cast<uint32_t>(std::stoul(Tok)));
          ++LX;
        }

        while (char* Tok = strtok_r(NULL, Sep, &End)) {
          V.push_back(static_cast<uint32_t>(std::stoul(Tok)));
          ++LX;
        }

        HDST.push_back(V);
      }
    }

    checkCudaError(cudaMemcpyToSymbol(H, &MX, sizeof(MX), 0, cudaMemcpyHostToDevice));
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

void PrintVector(const std::vector<uint32_t>& V) {
  std::cout << "{ ";
  if (!V.empty()) {
    std::vector<uint32_t>::const_iterator I = V.begin();
    std::vector<uint32_t>::const_iterator E = V.end();

    std::cout << *I;
    while (++I != E) {
      std::cout << ", " << *I;
    }
  }

  std::cout << " }";
}

void SetupDeviceVectors(uint32_t W, uint32_t H,
                        const std::vector<std::vector<uint32_t>>& HFLG,
                        const std::vector<std::vector<uint32_t>>& HDST,
                        uint32_t** OFLG, uint32_t** ODST,
                        uint32_t** DAS) {
  checkCudaError(cudaMallocManaged((void**) &GFLG, H * sizeof(uint32_t) *
                                   W * sizeof(uint32_t)));
  checkCudaError(cudaMallocManaged((void**) &GDST, H * sizeof(uint32_t) *
                                   W * sizeof(uint32_t)));

  checkCudaError(cudaGetSymbolAddress((void**) &HGFLG, GFLG));
  checkCudaError(cudaGetSymbolAddress((void**) &HGDST, GDST));

  uint32_t* XA = (uint32_t*) malloc(H * W * sizeof(uint32_t));
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
  (void) memset(XA, 0, H * W * sizeof(uint32_t));
  XAP = XA;

  for (std::vector<std::vector<uint32_t>>::const_iterator I = HDST.begin();
       I != HDST.end(); ++I) {
    const std::vector<uint32_t>& HV = *I;
    (void) memcpy(XAP, HV.data(), HV.size() * sizeof(uint32_t));
    XAP += HV.size();
    XAS += HV.size() * sizeof(uint32_t);
  }

  checkCudaError(cudaMemcpy(GDST, XA, XAS, cudaMemcpyHostToDevice));

  HAS.resize(HFLG.size());
  std::iota(HAS.begin(), HAS.end(), 0U);

  XAS = HAS.size() * sizeof(uint32_t);
  checkCudaError(cudaMallocManaged((void**) &GAS, XAS));
  checkCudaError(cudaGetSymbolAddress((void**) &HGAS, GAS));
  checkCudaError(cudaMemcpy(GAS, HAS.data(), XAS, cudaMemcpyHostToDevice));

  free(XA);

  *OFLG = HGFLG;
  *ODST = HGDST;
  *DAS = HGAS;
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

void SetupWeightAndHeight() {
  checkCudaError(cudaMemcpyFromSymbol((void*) &HW, W, sizeof(uint32_t), 0,
                                      cudaMemcpyDeviceToHost));
  checkCudaError(cudaMemcpyFromSymbol((void*) &HH, H, sizeof(uint32_t), 0,
                                      cudaMemcpyDeviceToHost));
}

void SetupSequenceLength() {
  checkCudaError(cudaMemcpyToSymbol(DSS, &HSS, sizeof(HSS), 0,
                                    cudaMemcpyHostToDevice));
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

  uint32_t* LH;
  uint32_t* LW;
  uint32_t HSZ = static_cast<uint32_t>(HFLG.size());

  checkCudaError(cudaGetSymbolAddress((void**) &LH, H));
  checkCudaError(cudaGetSymbolAddress((void**) &LW, W));

  checkCudaError(cudaMemcpyToSymbol(H, &HSZ, sizeof(HSZ), 0,
                                    cudaMemcpyHostToDevice));

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

    HSZ = static_cast<uint32_t>((*FI).size());
    checkCudaError(cudaMemcpyToSymbol(W, &HSZ, sizeof(HSZ), 0,
                                      cudaMemcpyHostToDevice));
  }

  return true;
}

__device__
uint32_t ComputeCost(uint32_t WID) {
  uint32_t DI;
  uint32_t DJ;
  uint32_t Cost = 0U;

  for (uint32_t I = 0; I < DSS; ++I) {
    for (uint32_t J = 0; J < DSS; ++J) {
      DI = GAS[I];
      DJ = GAS[J];
      Cost += qap::array_index(GFLG, I, J, WID) *
              qap::array_index(GDST, DI, DJ, WID);
    }
  }

  return Cost;
}

__device__
void PrintDeviceVector(const uint32_t* AS, uint32_t S, uint32_t TID) {
  (void) printf("\nTID: %u { ", TID);
  const uint32_t* B = AS;
  const uint32_t* E = AS + (S - 1);

  (void) printf("%u", *B);

  while (++B != E)
    (void) printf(", %u", *B);

    (void) printf(" } ");
}

__global__ void QuadraticUniverse(size_t LSS, uint32_t* XAS) {
  if (threadIdx.x == 0 && LSS > 2) {
    XMC = std::numeric_limits<uint64_t>::max();

    for (uint32_t I = 0; I < LSS; ++I) {
      MCD[I] = XAS[I];
    }
  }

  __syncthreads();
}

template<uint32_t _ShmemSize = 4096, uint32_t _MCSize>
__global__ void QuadraticAssignment(size_t LSS, uint32_t MCSize, uint32_t* XAS) {
  if (LSS < 2)
    return;

  (void) atomicAdd((unsigned long long*) &TOT, 1UL);
  uint32_t TID = blockIdx.x * blockDim.x + threadIdx.x;

  do {
    uint32_t CC = ComputeCost(W);
    (void) atomicMin((unsigned long long*) &XMC, CC);
    (void) atomicAdd((unsigned long long*) &IT, 1UL);
  } while (qap::next_permutation(MCD, LSS));

  __syncthreads();

  MC = XMC;
  __syncthreads();
}

uint64_t Iterations() {
  uint64_t ITP[2];
  uint64_t LIT;
  checkCudaError(cudaGetSymbolAddress((void**) &ITP, IT));
  checkCudaError(cudaMemcpyFromSymbol(&LIT, IT, sizeof(uint64_t), 0,
                                      cudaMemcpyDeviceToHost));
  GIT = LIT;
  return LIT;
}

uint64_t MinimumCost() {
  uint64_t MCP[2];
  uint64_t LMC;
  checkCudaError(cudaGetSymbolAddress((void**) &MCP, MC));
  checkCudaError(cudaMemcpyFromSymbol(&LMC, MC, sizeof(uint64_t), 0,
                                      cudaMemcpyDeviceToHost));
  GMC = LMC;
  return LMC;
}

uint32_t NumThreads() {
  uint64_t TOTP[2];
  uint64_t LTOT;
  checkCudaError(cudaGetSymbolAddress((void**) &TOTP, TOT));
  checkCudaError(cudaMemcpyFromSymbol(&LTOT, TOT, sizeof(uint64_t), 0,
                                      cudaMemcpyDeviceToHost));
  return LTOT;
}

void SetupInitialCounters() {
  uint64_t IP = 0UL;
  uint32_t CP = std::numeric_limits<uint32_t>::max();;

  checkCudaError(cudaMemcpyToSymbol(MC, &CP, sizeof(uint32_t), 0,
                                    cudaMemcpyHostToDevice));
  checkCudaError(cudaMemcpyToSymbol(IT, &IP, sizeof(uint64_t), 0,
                                    cudaMemcpyHostToDevice));
}

void ReadCounters() {
  uint64_t* ITP;
  uint32_t* MCP;

  checkCudaError(cudaGetSymbolAddress((void**) &ITP, (const void*) &IT));
  checkCudaError(cudaGetSymbolAddress((void**) &MCP, (const void*) &MC));

  checkCudaError(cudaMemcpy(&HIT, ITP, sizeof(uint64_t), cudaMemcpyDeviceToHost));
  checkCudaError(cudaMemcpy(&HMC, MCP, sizeof(uint32_t), cudaMemcpyDeviceToHost));
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
  std::cerr << "Usage: qap [-h | --help]" << std::endl;
  std::cerr << "           [-s <input-file> | --single-file <input-file>]"
    << std::endl;
  std::cerr << "           (use single-file matrix input format)." << std::endl;
  std::cerr << "           [ -f <flow-input-file> | --flinput <flow-input-file>]"
    << std::endl;
  std::cerr << "           [ -d <distance-input-file> | --dstinput <distance-input-file>]"
    << std::endl;
  std::cerr << "           [ -m <gpu-shared-memory-size> | --shmem-size <gpu-shared-memory-size>]" << std::endl;
  std::cerr << "           (GPU shared memory size in KB - default is 1024)."
    << std::endl;
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
  uint32_t ShmemSize = 1024U;
  bool DoPrint = false;
  uint32_t* DFLG;
  uint32_t* DDST;
  uint32_t* DAS;

  while (1) {
    C = getopt_long(argc, argv, "hps:f:d:m:", long_options, &OIx);
    if (C == -1)
      break;

    switch (C) {
    case 'h':
      PrintHelp();
      return 0;
      break;
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
      break;
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

  SetupWeightAndHeight();
  SetupDeviceVectors(HW, HH, HFLG, HDST, &DFLG, &DDST, &DAS);
  HSS = static_cast<uint32_t>(HAS.size());
  SetupSequenceLength();
  SetupInitialCounters();

  uint32_t NBlocks = 16U;
  dim3 ThreadsPerBlock(16U, 16U);
  dim3 NumBlocks(16U, 16U);

  Timestamp(&tp_start);
  QuadraticUniverse<<<1, 1>>>(HAS.size(), GAS);
  QuadraticAssignment<4096, 1024><<<ThreadsPerBlock, NumBlocks, ShmemSize + HAS.size()>>>(HAS.size(), NBlocks, DAS);
  checkCudaError(cudaDeviceSynchronize());
  Timestamp(&tp_end);

  std::cout << "Minimum cost: " << MinimumCost() << std::endl;
  std::cout << "Iterations:   " << Iterations() << std::endl;
  std::cout << "NumThreads:   " << NumThreads() << std::endl;

  checkCudaError(cudaDeviceReset());

  if (DoPrint)
    Print();

  PrintTimediff(&tp_start, &tp_end);

  return 0;
}

