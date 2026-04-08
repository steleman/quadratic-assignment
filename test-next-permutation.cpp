// Copyright (c) 2025-2026 Stefan Teleman.
//
// Licensed under the MIT License.
// See https://opensource.org/license/mit
// SPDX-License-Identifier: MIT
//
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <atomic>
#include <cstdint>

namespace qap {

template<typename _Ty>
inline
void swap(_Ty* A, _Ty* B) {
  _Ty T = *B;
  *B = *A;
  *A = T;
}

template<typename _Ty>
inline
_Ty min(_Ty A, _Ty B) {
  return A < B ? A : B;
}

template<typename _Ty>

void reverse(_Ty* AX, int32_t B, int32_t E) {
  while (B < E) {
    swap(&AX[B], &AX[E]);
    ++B;
    --E;
  }
}

template<typename _Ty>
inline
_Ty array_index(const _Ty* AX, uint32_t IXW, uint32_t IXH, uint32_t W) {
  const _Ty* AXI = static_cast<const _Ty*>(AX + IXW * W);
  const _Ty* AXP = AXI + IXH;
  return *AXP;
}

template<typename _Ty>
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

template<typename _Ty>
void print_vector(const std::vector<_Ty>& V, std::ostream& ofs = std::cerr) {
  ofs << "{ ";
  if (!V.empty()) {
    typename std::vector<_Ty>::const_iterator I = V.begin();
    typename std::vector<_Ty>::const_iterator E = V.end();

    ofs << *I;
    while (++I != E)
      ofs << ", " << *I;
  }

  ofs << " }" << std::endl;
}

template<typename _Ty>
void print_array(const _Ty* A, size_t N, std::ostream& ofs = std::cerr) {
  ofs << "{ ";
  if (N > 0) {
    const _Ty* I = A;
    const _Ty* E = A + N;

    ofs << *I;
    while (++I != E)
      ofs << ", " << *I;
  }

  ofs << " }" << std::endl;
}

int main()
{
  std::ofstream VFS;
  std::ofstream AFS;
  const char* VO = "vector-tnp-output.txt";
  const char* AO = "array-tnp-output.txt";

  // The bigger the vectors the bigger the permutation set.
  // You have been warned. :-)
  std::vector<uint32_t> V = { 0, 1, 2, 3, 4, 5 };
  uint32_t A[] = { 0, 1, 2, 3, 4, 5 };
  uint32_t IX = 0U;

  VFS.open(VO);
  if (!VFS.good() || VFS.eof()) {
    std::cerr << "Could not open vector output file " << VO
      << " for writing." << std::endl;
    return 1;
  }

  AFS.open(AO);
  if (!AFS.good() || AFS.eof()) {
    std::cerr << "Could not open array output file " << AO
      << " for writing." << std::endl;
    return 1;
  }

  do {
    VFS << ++IX << ": ";
    print_vector(V, VFS);
  } while (std::next_permutation(V.begin(), V.end()));

  IX = 0U;
  uint32_t AX = sizeof(A) / sizeof(A[0]);

  do {
    AFS << ++IX << ": ";
    print_array(A, AX, AFS);
  } while (qap::next_permutation(A, AX));

  AFS.close();
  VFS.close();

  std::cerr << "std::vector output is in " << VO << std::endl;
  std::cerr << "array output is in " << AO << std::endl;

  return 0;
}

