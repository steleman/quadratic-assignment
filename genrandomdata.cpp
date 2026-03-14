// Copyright (c) 2024-2026 Stefan Teleman.
//
// Licensed under the MIT License.
// See https://opensource.org/license/mit
// SPDX-License-Identifier: MIT
//

#include <iostream>
#include <vector>
#include <set>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <getopt.h>

static std::vector<std::set<uint32_t>> SV;
static std::set<uint32_t> FUS;

static inline int32_t GenSetElement(uint32_t ub) {
  uint32_t e = 0;

  do {
    e = (int32_t) lrand48();
  } while (e > ub || e < 1U);

  return (int32_t) e;
}

static inline uint32_t GenLowerBound(uint32_t ub) {
  uint32_t lb = 0;

  do {
    lb = (int32_t) lrand48();
  } while (lb > ub);

  return lb;
}

static inline uint32_t GenUpperBound(uint32_t lim) {
  uint32_t ub = 0;

  do {
    ub = (int32_t) lrand48();
  } while (ub > lim);

  return ub;
}

static inline uint32_t GenSetSize(uint32_t usize, uint32_t mSize, uint32_t MSize) {
  uint32_t ssize = 0;

  do {
    ssize = (uint32_t) lrand48();
  } while (ssize > MSize || ssize > usize || ssize < mSize);

  return ssize;
}

bool WriteFile(const std::string& Filename) {
  std::ofstream OF(Filename);
  if (!OF.good()) {
    std::cerr << "Could not open file " << Filename << " for writing."
      << std::endl;
    return false;
  }

  std::vector<std::set<uint32_t>>::const_iterator VI = SV.begin();
  std::vector<std::set<uint32_t>>::const_iterator VE = SV.end();

  while (VI != VE) {
    const std::set<uint32_t>& S = *VI;
    std::set<uint32_t>::const_iterator SI = S.begin();
    std::set<uint32_t>::const_iterator SE = S.end();

    OF << *SI;
    while (++SI != SE) {
      OF << ", " << *SI;
    }

    OF << '\n';
    ++VI;
  }

  OF.close();
  return true;
}

void PrintSet(const std::set<uint32_t>& S) {
  std::cerr << "{ ";
  if (!S.empty()) {
    std::set<uint32_t>::const_iterator I = S.begin();
    std::set<uint32_t>::const_iterator E = S.end();

    std::cerr << *I;
    while (++I != E) {
      std::cerr << ", " << *I;
    }
  }

  std::cerr << " }" << std::endl;
}

void PrintSetVector(const std::vector<std::set<uint32_t>> SV) {
  if (!SV.empty()) {
    std::cerr << "======================================" << std::endl;
    std::vector<std::set<uint32_t>>::const_iterator I = SV.begin();
    std::vector<std::set<uint32_t>>::const_iterator E = SV.end();

    while (I != E) {
      std::cerr << "--------------------------------------" << std::endl;
      PrintSet(*I);
      std::cerr << "--------------------------------------" << std::endl;
      ++I;
    }
    std::cerr << "======================================" << std::endl;
  }
}

static void PrintHelp()
{
  std::cerr << "Usage: genrandomdata  [--help | -h]" << std::endl;
  std::cerr << "                      (print this message)" << std::endl;
  std::cerr << "                      [--quiet | -q]" << std::endl;
  std::cerr << "                      (nothing printed to stdout)" << std::endl;
  std::cerr << "                      [--auto | -a]" << std::endl;
  std::cerr << "                      (auto-fill the generated sets to cover the full universe)" << std::endl;
  std::cerr << "                      [--fixed | -f]" << std::endl;
  std::cerr << "                      (constant (fixed) set size. maxsize == minsize)." << std::endl;
  std::cerr << "                      [--output <filename> | -o <filename>]"
    << std::endl;
  std::cerr << "                      (output filename)" << std::endl;
  std::cerr << "                      [--usize <universe-size> | -u <universe-size>]"
    << std::endl;
  std::cerr << "                      (number of elements in the universe)" << std::endl;
  std::cerr << "                      [--minsize <minimum-subset-size> | -m <minimum-subset-size>]" << std::endl;
  std::cerr << "                      (minimum size of a generated subset)" << std::endl;
  std::cerr << "                      [--maxsize <maximum-subset-size> | -M <maximum-subset-size>]" << std::endl;
  std::cerr << "                      (maximum size of a generated subset)" << std::endl;
  std::cerr << "                      [--nsets <number-of-sets> | -n <number-of-sets>]"
    << std::endl;
}

static bool CheckArguments(uint32_t U, uint32_t S, bool A) {
  return A ? U != 0U : U != 0U && S != 0U;
}

static bool ValidateUniverseSets(uint32_t USize) {
  return FUS.size() == USize;
}

static struct option long_options[] = {
  { "help",     no_argument,        0,  'h' },
  { "quiet",    no_argument,        0,  'q' },
  { "auto",     no_argument,        0,  'a' },
  { "fixed",    no_argument,        0,  'f' },
  { "output",   required_argument,  0,  'o' },
  { "usize",    required_argument,  0,  'u' },
  { "nsets",    required_argument,  0,  'n' },
  { "minsize",  required_argument,  0,  'm' },
  { "maxsize",  required_argument,  0,  'M' },
  { 0,          0,                  0,   0  }
};

int main(int argc, char* const argv[])
{
  int c;
  int OIx = 0;
  uint32_t USize = 0U;
  uint32_t NSets = 0U;
  uint32_t mSize = 4U;
  uint32_t MSize = 8U;
  uint32_t GSets = 0;
  std::string Outfile;
  bool Quiet = false;
  bool Autofill = false;
  bool Fixed = false;

  while (1) {
    c = getopt_long(argc, argv, "afhqo:u:m:M:n:",
                    long_options, &OIx);
    if (c == -1)
      break;

    switch (c) {
    case 'o':
      Outfile = optarg;
      break;
    case 'u':
      USize = (uint32_t) std::stoul(optarg);
      break;
    case 'n':
      NSets = (uint32_t) std::stoul(optarg);
      break;
    case 'm':
      mSize = (uint32_t) std::stoul(optarg);
      break;
    case 'M':
      MSize = (uint32_t) std::stoul(optarg);
      break;
    case 'q':
      Quiet = true;
      break;
    case 'a':
      Autofill = true;
      break;
    case 'f':
      Fixed = true;
      break;
    default:
      PrintHelp();
      return 1;
      break;
    }
  }

  if (!CheckArguments(USize, NSets, Autofill)) {
    PrintHelp();
    return 1;
  }

  if (Fixed)
    MSize = mSize;

  srand48(time(0));

  if (Autofill) {
    while (FUS.size() < USize) {
      uint32_t SSize = GenSetSize(USize, mSize, MSize);
      std::set<uint32_t> TS;

      while (TS.size() < SSize) {
        uint32_t E = GenSetElement(USize);
        if (TS.insert(E).second) {
          FUS.insert(E);
          if (!Quiet)
            std::cerr << E << ' ';
        }
      }

      SV.push_back(TS);
      ++GSets;
      if (!Quiet)
        std::cerr << std::endl;
    }

    if (GSets < NSets) {
      for (uint32_t j = 0; j < NSets - GSets; ++j) {
        uint32_t SSize = GenSetSize(USize, mSize, MSize);
        std::set<uint32_t> TS;

        while (TS.size() < SSize) {
          uint32_t E = GenSetElement(USize);
          if (TS.insert(E).second) {
            FUS.insert(E);
            if (!Quiet)
              std::cerr << E << ' ';
          }
        }

        SV.push_back(TS);
        if (!Quiet)
          std::cerr << std::endl;
      }
    }
  } else {
    for (uint32_t j = 0; j < NSets; ++j) {
      uint32_t SSize = GenSetSize(USize, mSize, MSize);
      std::set<uint32_t> TS;

      while (TS.size() < SSize) {
        uint32_t E = GenSetElement(USize);
        if (TS.insert(E).second) {
          FUS.insert(E);
          if (!Quiet)
            std::cerr << E << ' ';
        }
      }

      SV.push_back(TS);
      if (!Quiet)
        std::cerr << std::endl;
    }
  }

  if (!Quiet)
    PrintSetVector(SV);

  if (!ValidateUniverseSets(USize)) {
    std::cerr << "Generated Universe Sets do not cover the entire Universe."
      << std::endl;
    return 1;
  }

  if (!Outfile.empty()) {
    if (!WriteFile(Outfile))
      return 1;
  }

  return 0;
}

