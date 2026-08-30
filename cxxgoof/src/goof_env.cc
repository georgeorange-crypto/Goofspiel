// Author: 陈子聪 (Chen Zicong)
// Date:   2026-08-30
// Purpose: Vectorised (batched) Goofspiel environment, plus single-env
//          convenience wrapper used for debugging / exact-Nash.
//
// Designed for training throughput.  Key optimisations:
//   * struct-of-arrays (SoA) state layout.
//   * no heap allocation in reset_batch / step_batch.
//   * `step_batch` is a tight loop over M independent envs that compiles to
//     auto-vectorised code on MSVC/GCC/clang at -O2+.
//
// The Python-level Gymnasium wrapper builds (obs, rew, term, info) tensors
// from the raw outputs (see bindings.cc and goofspiel/_cxx.py).
#include "goofspiel/goof_env.h"

// NOTE: SingleEnv / VectorizedEnv class definitions now live in the header
// (goof_env.h) so the pybind11 translation unit (bindings.cc) can see their
// complete types and instantiate std::make_unique<VectorizedEnv>.  This TU
// previously duplicated them — the duplication caused ODR violations once
// the header carried the in-class definitions, so the body was removed in
// favour of the single-source-of-truth header.
//
// Historical copies are retained only for blame-tracking: see git log.
#include <cassert>
#include <cstdint>

namespace cxxgoof {

// (intentionally empty — classes defined in goof_env.h)

}  // namespace cxxgoof
