// Author: 陈子聪 (Chen Zicong)
// Date:   2026-08-30
// Purpose: Level-A closed-form complexity estimator (matches Python's
// goofspiel.solver.estimate_complexity exactly).  Needed by the Nash solver
// so that N=13 is correctly classified BLACK and never even enters the
// recursion.
//
// Reference: OEIS A000172 = sum_{k=0..N} C(N,k)^3.
//
// Also implements Level-B preflight (RAM estimate + LP-bench approximation).
// The exact runtime calibration is handled from Python because the LP
// fallback currently lives in scipy; the C++ side only gives us a
// conservative closed-form bound.
#pragma once
#include <cstdint>
#include <utility>

namespace cxxgoof {

struct ComplexityReportCpp {
    int         N;
    uint64_t    C_N;                // sum C(N,k)^3
    uint64_t    L_N;                // sum C(N,k)^2 (states-per-layer, swap-symm cut)
    uint64_t    E_N;                // expected sub-LP calls = sum_k C(N,k)^4 / k? No —
                                    //   E_N = sum_k C(N,k)^3 · floor(k/2 + 1)^2 (heuristic)
    double      mem_bytes_expected; // N * C_N * (4 double per cache slot)
    double      mem_bytes_conservative;
    const char* risk_level;         // "GREEN","YELLOW","ORANGE","RED","BLACK" (C string)
    bool        can_run_without_force;
    // ---- Per-layer C(N,k) counts for k = 0..N ----
    int         layer_N;
    uint64_t    layers[14]; // N ≤ 13
};

// Binomial coefficient C(n, k) — exact uint64, n ≤ 26 (only called for N<=13).
inline uint64_t binom_small(int n, int k) noexcept {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n - k) k = n - k;
    uint64_t r = 1;
    for (int i = 1; i <= k; ++i) {
        r = r * static_cast<uint64_t>(n - k + i);
        r /= static_cast<uint64_t>(i);
    }
    return r;
}

inline ComplexityReportCpp estimate_complexity_level_a(int N) noexcept {
    ComplexityReportCpp r{};
    r.N = N;
    r.layer_N = N + 1;
    uint64_t C = 0, L = 0, E = 0;
    for (int k = 0; k <= N; ++k) {
        const uint64_t cnk = binom_small(N, k);
        const uint64_t c2 = cnk * cnk;
        const uint64_t c3 = c2 * cnk;
        r.layers[k] = c3;
        C += c3;
        L += c2;
        // Each (A,B,R) at layer k triggers a k×k LP — work ∝ k^3 (Loewner).
        E += c3 * static_cast<uint64_t>((k + 1) * (k + 1) * (k + 1));
    }
    r.C_N = C;
    r.L_N = L;
    r.E_N = E;
    // memory: each cached chance-state costs ~4 doubles + 64-bit key = 40 byte.
    // Canonicalization halves cache roughly:
    const double states = static_cast<double>(C) * 0.5;
    const double per = 40.0;
    r.mem_bytes_expected = states * per;
    r.mem_bytes_conservative = r.mem_bytes_expected * 1.8 + 64.0 * 1024.0 * 1024.0;

    // Risk thresholds (Python's Level-B tuned numbers, approximated here
    // because from C++ we don't know installed RAM exactly; caller can
    // override via force flag.)
    if (N <= 5)                              r.risk_level = "GREEN";
    else if (N == 6 || N == 7)                r.risk_level = "YELLOW";
    else if (N == 8)                          r.risk_level = "ORANGE";
    else if (N == 9)                          r.risk_level = "RED";
    else                                      r.risk_level = "BLACK";
    r.can_run_without_force = (N <= 9);
    return r;
}

}  // namespace cxxgoof
