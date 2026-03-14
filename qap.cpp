// Copyright (c) 2024-2026 Stefan Teleman.
//
// Licensed under the MIT License.
// See https://opensource.org/license/mit
// SPDX-License-Identifier: MIT
//

#include <iostream>
#include <iomanip>
#include <vector>
#include <set>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <cerrno>
#include <getopt.h>

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

  std::cout << "Clock Resolution: " << (int64_t) R.tv_sec << '.'
    << std::setfill('0') << std::setw(9) << (int64_t) R.tv_nsec
    << '.' << std::endl;
  std::cout << "CPU time: " << (int64_t) Sec / 1000000000L << '.'
    << std::setfill('0') << std::setw(12) << (int64_t) Nns << " second(s)."
    << std::endl;
}

class QAP {
private:
  std::vector<std::vector<uint32_t>> FLG;
  std::vector<std::vector<uint32_t>> DST;
  std::vector<uint32_t> AS;
  uint64_t IT;
  uint32_t MC;

private:
  void ReadFromFile(const std::string& FileName,
                    std::vector<std::vector<uint32_t>>& IV) {
    IV.clear();

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

  void PrintVector(const std::vector<uint32_t>& V) const {
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

  void PrintGraphVector(const std::vector<std::vector<uint32_t>>& V) const {
    std::cout << "{ ";
    if (!V.empty()) {
      std::vector<std::vector<uint32_t>>::const_iterator I = V.begin();
      std::vector<std::vector<uint32_t>>::const_iterator E = V.end();

      PrintVector(*I);

      while (++I != E) {
        std::cout << ',' << std::endl << "  ";
        PrintVector(*I);
      }
    }

    std::cout << " }" << std::endl;
  }

public:
  QAP() : FLG(), DST(), AS(), IT(0UL), MC(std::numeric_limits<uint32_t>::max())
  { }

  void ReadFromFile(const std::string& FLGName,
                    const std::string& DSTName) {
    ReadFromFile(FLGName, FLG);
    ReadFromFile(DSTName, DST);
  }

  void ReadFromSingleFile(const std::string& FileName) {
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

          FLG.push_back(V);
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

          DST.push_back(V);
        }
      }
    } else {
      std::cerr << "Invalid input file " << FileName << " was specified."
        << std::endl;
      std::exit(1);
    }

  }

  bool Verify() const {
    if (FLG.size() != DST.size()) {
      std::cerr << "Outer Vectors are not the same size." << std::endl;
      return false;
    }

    std::vector<std::vector<uint32_t>>::const_iterator FI;
    std::vector<std::vector<uint32_t>>::const_iterator FE = FLG.end();
    std::vector<std::vector<uint32_t>>::const_iterator DI;
    std::vector<std::vector<uint32_t>>::const_iterator DE = DST.end();

    for (FI = FLG.begin(), DI = DST.begin(); FI != FE && DI != DE;
         ++FI, ++DI) {
      if ((*FI).size() != (*DI).size()) {
        std::cerr << "Inner Vectors are not the same size." << std::endl;
        return false;
      }
    }

    return true;
  }

  uint32_t ComputeCost() const {
    uint32_t Cost = 0U;

    for (uint32_t I = 0; I < AS.size(); ++I) {
      for (uint32_t J = 0; J < AS.size(); ++J) {
        Cost += FLG[I][J] * DST[AS[I]][AS[J]];
      }
    }

    return Cost;
  }

  uint32_t QuadraticAssignment() {
    AS.clear();
    AS.resize(FLG.size());

    for (uint32_t I = 0; I < FLG.size(); ++I)
      AS[I] = I;

    MC = std::numeric_limits<uint32_t>::max();

    do {
      uint32_t Cost = ComputeCost();
      if (Cost < MC) {
        MC = Cost;
      }
      ++IT;
    } while (std::next_permutation(AS.begin(), AS.end()));

    return MC;
  }

  uint64_t Iterations() const {
    return IT;
  }

  void PrintFlowGraphVector() const {
    PrintGraphVector(FLG);
  }

  void PrintDistanceVector() const {
    PrintGraphVector(DST);
  }

  void PrintAssignmentVector() const {
    PrintVector(AS);
  }

  void Print() const {
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
};

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
  std::cerr << "           [-p | --print]" << std::endl;
  std::cerr << "           (print vector contents)." << std::endl;
}

static struct option long_options[] = {
  { "help",             no_argument,        0,  'h' },
  { "print",            no_argument,        0,  'p' },
  { "single-file",      required_argument,  0,  's' },
  { "flinput",          required_argument,  0,  'f' },
  { "dstinput",         required_argument,  0,  'd' },
  { 0,                  0,                  0,   0  }
};

int main(int argc, char* const argv[]) {
  int C;
  int OIx = 0;
  std::string FLGFile;
  std::string DSTFile;
  std::string SFile;
  bool Print = false;

  while (1) {
    C = getopt_long(argc, argv, "hps:f:d:", long_options, &OIx);
    if (C == -1)
      break;

    switch (C) {
    case 'h':
      PrintHelp();
      return 0;
      break;
    case 'p':
      Print = true;
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

  QAP Q;

  if (!SFile.empty())
    Q.ReadFromSingleFile(SFile);
  else
    Q.ReadFromFile(FLGFile, DSTFile);

  if (!Q.Verify())
    return 1;

  Timestamp(&tp_start);
  uint32_t MC = Q.QuadraticAssignment();
  Timestamp(&tp_end);

  std::cout << "Minimum cost: " << MC << std::endl;
  std::cout << "Iterations: " << Q.Iterations() << std::endl;

  if (Print)
    Q.Print();

  PrintTimediff(&tp_start, &tp_end);

  return 0;
}

