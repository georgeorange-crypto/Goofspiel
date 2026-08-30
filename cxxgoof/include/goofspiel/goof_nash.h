// Author: 陈子聪 (Chen Zicong)
// Date:   2026-08-30
// Purpose: Exact Nash Goofspiel solver, C++ implementation.
//          Mirrors goofspiel/solver.py::GoofspielExactSolver semantics so
//          that the Python `SolveResult` wrapper can consume the output map
//          directly (bitmask keys, per-(A,B,R,prize) V,x,y entries).
//
// Recap of the exact recursion (repeated here so this file is self-contained):
//
//   F(A,B,R) = 1/|R| * Σ_{p∈R}  V_p(A,B,R \ {p})
//   where V_p = value of the k×k zero-sum matrix game with
//     M_{a,b} = sgn(a-b) * p  +  F(A\{a}, B\{b}, R\{p})
//   and the Nash policies (x*, y*) for that V_p are stored alongside V_p
//   in policy_map[(A,B,R,p)] = (V_p, x*, y*).
//
// Symmetry shortcut: F(A,A,R) = 0 (game symmetric when players have equal
// hands).  Swap canonicalization: F(A,B,R) = -F(B,A,R) so the cache only
// needs the smaller half of all (A,B,R) triples — halves memory + time.
//
// Two-phase recursion (fixes the sign-leakage bug):
//   Phase 1. Before constructing matrix M for the *current* chance-node,
//            EAGERLY recurse into every *child* chance-node (for every
//            (a,b,p) combination) and store them in the cache.  The current
//            node only starts once every reachable child has a value.
//   Phase 2. Build matrix M by *pure cache reads* via lookup_chance() which
//            reads cache[canonical(child)] * canonical_sign.  This decouples
//            the read semantics from the parent's sign context so the stored
//            value is always the canonical (row=A <= B) signed value.
//
// LP backend: because shipping a HiGHS binary in every user environment is
//          hard, the *default* path delegates k×k matrix solves to the
//          Python scipy.optimize.linprog via a pybind11 callback installed
//          at module-import time (see bindings.cc).  A native HiGHS path is
//          also available (CXXGOOF_HAVE_HIGHS) for HPC clusters; when used
//          the solver is fully compiled and Python-free.
#pragma once
#include "goofspiel/goof_env.h"
#include "goofspiel/goof_estimate.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(CXXGOOF_HAVE_HIGHS) && CXXGOOF_HAVE_HIGHS
#  include <Highs.h>
#endif

namespace cxxgoof {

// LP result returned by the backend (native or callback).
struct LpSolution {
    double v;          // Nash value of k×k matrix (row player, maximiser)
    std::vector<double> x; // row policy, length k (sorted-card indices)
    std::vector<double> y; // col policy, length k
    bool   converged;
};

// Callback signature for the Python-delegated LP backend.
// args: k, (k*k) doubles row-major matrix M[0..k²-1].
// returns a filled LpSolution.
using LpSolverFn = std::function<LpSolution(int k, const double* M_row_major)>;

// Policy entry for every (A,B,R,prize).  Python wrapper converts this into
// goofspiel.solver.SolveResult::policy_map = dict[tuple(A,B,R,p)] -> tuple(V,x,y).
struct PolicyEntry {
    uint16_t A, B, R;
    int      prize;
    double   V;
    std::vector<double> x;
    std::vector<double> y;
};

struct SolveResultCpp {
    int                             N;
    double                          root_value;   // F(full, full, full)
    ComplexityReportCpp             complexity;
    std::vector<PolicyEntry>        policy_map;   // 1 entry per (A,B,R,p)
    std::string                     level_b_risk;
    bool                            used_force;
    std::string                     message;
};

// ==========================================================================
// The Solver Object
// ==========================================================================
class GoofspielExactSolverCpp {
public:
    explicit GoofspielExactSolverCpp(LpSolverFn lp) : lp_(std::move(lp)) {}

    ComplexityReportCpp estimate(int N) const { return estimate_complexity_level_a(N); }

    SolveResultCpp solve_with_policy(int N, bool force) const {
        SolveResultCpp out{};
        out.N = N;
        out.complexity = estimate(N);
        out.level_b_risk = out.complexity.risk_level;
        out.used_force = force;
        if (!out.complexity.can_run_without_force && !force) {
            out.message = std::string("BLACK risk (N=") + std::to_string(N)
                        + "). Pass force=True to run.";
            return out;
        }
        if (!lp_) {
            throw std::logic_error("GoofspielExactSolverCpp: no LP solver callback installed. "
                                   "Call install_lp_callback(py_scipy_linprog) from Python first.");
        }
        cache_.clear();
        policy_out_.clear();
        // Shortcut: symmetric root state has F=0 by definition.
        const uint16_t full = (N == 13) ? uint16_t(0x1FFF)
                                        : static_cast<uint16_t>((1u << N) - 1);
        // But we still descend to fill the full policy_map.
        out.root_value = solve_chance_with_policies(full, full, full);
        // Steal policy entries into the return value
        out.policy_map = std::move(policy_out_);
        policy_out_.clear();
        cache_.clear();
        out.message = "solved";
        return out;
    }

private:
    // mutable because solve_with_policy is logically const (caches are tmp scratchpads).
    LpSolverFn lp_;
    mutable std::unordered_map<uint64_t, double> cache_;
    mutable std::vector<PolicyEntry> policy_out_;

    // ---------------- sign-aware cache read helper (Phase 2) ----------------
    // Returns F(A,B,R); on cache-miss (shouldn't happen in 2-phase design)
    // we compute fresh to be safe rather than returning garbage.
    double lookup_chance(uint16_t A, uint16_t B, uint16_t R) const {
        if (A == B) return 0.0;  // short-cut symmetry for all equal-hand nodes
        const auto c = canonicalize(A, B, R);
        auto it = cache_.find(c.key);
        if (it != cache_.end()) return c.sign * it->second;
        // Fallback compute (should not happen after Phase 1 done, but kept
        // for robustness when, e.g., user asks for policy_at a custom key).
        const double v = solve_chance_only(A, B, R);
        // Store canonical signed value.
        cache_.emplace(c.key, v / c.sign);
        return v;
    }

    // ------ Solve F(A,B,R) WITHOUT storing policy entries (leaf Phase 1). ----
    // Uses recursion with identical semantics; identical to
    // solve_chance_with_policies but without the policy map writes.
    double solve_chance_only(uint16_t A, uint16_t B, uint16_t R) const {
        if (A == 0 || B == 0 || R == 0) return 0.0;
        if (A == B) return 0.0;
        const auto c = canonicalize(A, B, R);
        auto it = cache_.find(c.key);
        if (it != cache_.end()) return c.sign * it->second;
        // Expand chance-node: enumerate remaining prizes.
        std::array<int, kMaxCards> prizes_arr; int kR = 0;
        for (uint16_t mm = R; mm; ) {
            const int v = lsb_value(mm);
            prizes_arr[kR++] = v;
            mm = static_cast<uint16_t>(mm & ~card_mask(v));
        }
        // Phase 1: pre-solve all children reachable from ANY (a,b,p) triple.
        // Collect unique children here.  (Solving duplicates is O(1) lookup so
        // we use a simple nested loop + recursive calls; the cache takes care
        // of dedup.)
        std::array<int, kMaxCards> As, Bs; int kA = 0, kB = 0;
        for (uint16_t mm = A; mm; ) { int v = lsb_value(mm); As[kA++]=v; mm &= ~card_mask(v); }
        for (uint16_t mm = B; mm; ) { int v = lsb_value(mm); Bs[kB++]=v; mm &= ~card_mask(v); }
        for (int p = 0; p < kR; ++p) {
            const int pv = prizes_arr[p];
            const uint16_t R1 = static_cast<uint16_t>(R & ~card_mask(pv));
            for (int a = 0; a < kA; ++a) {
                const uint16_t A1 = static_cast<uint16_t>(A & ~card_mask(As[a]));
                for (int b = 0; b < kB; ++b) {
                    const uint16_t B1 = static_cast<uint16_t>(B & ~card_mask(Bs[b]));
                    // Note: we IGNORE the return value here; we only need the
                    // side effect that cache_ is populated for key (A1,B1,R1)
                    // so Phase-2 lookup below is guaranteed a miss-free read.
                    (void)solve_chance_only(A1, B1, R1);
                }
            }
        }
        // Phase 2: build M per p, solve LP, sum V_p.
        const double inv_kR = 1.0 / static_cast<double>(kR);
        double F = 0.0;
        for (int p = 0; p < kR; ++p) {
            const int pv = prizes_arr[p];
            const uint16_t R1 = static_cast<uint16_t>(R & ~card_mask(pv));
            std::vector<double> M(static_cast<size_t>(kA) * static_cast<size_t>(kB), 0.0);
            for (int a = 0; a < kA; ++a) {
                const int av = As[a];
                const uint16_t A1 = static_cast<uint16_t>(A & ~card_mask(av));
                const double sgn =
                    (av > pv) ? 1.0 : (av < pv) ? -1.0 : 0.0;
                for (int b = 0; b < kB; ++b) {
                    const int bv = Bs[b];
                    const uint16_t B1 = static_cast<uint16_t>(B & ~card_mask(bv));
                    const double sgn2 =
                        (av > bv) ? 1.0 : (av < bv) ? -1.0 : 0.0;
                    const double future = lookup_chance(A1, B1, R1);
                    // M is zero-sum: payoff to row player = p·sgn(a-b) + F(…)
                    M[static_cast<size_t>(a) * kB + b] =
                        static_cast<double>(pv) * sgn2 + future;
                    (void)sgn;  // (keep the explicit form; sgn2 = sgn(a-b) which is what the
                                //  game actually needs — previous line used `sgn` buggy per spec)
                }
            }
            const auto sol = lp_(kA, M.data());
            F += inv_kR * sol.v;
        }
        // Store in canonical form.
        cache_.emplace(c.key, F / c.sign);
        return F;
    }

    // ---- Solve F(A,B,R) AND fill policy_out_ with V,x,y for every prize. ---
    double solve_chance_with_policies(uint16_t A, uint16_t B, uint16_t R) const {
        if (A == 0 || B == 0 || R == 0) return 0.0;
        if (A == B) {
            // Even in symmetric cases, caller needs policy entries for every
            // (A,B,R,p).  Solve the matrix for each p to get x*, y*.
            fill_policy_for_equal_hands(A, R);
            return 0.0;
        }
        const auto c = canonicalize(A, B, R);
        auto it = cache_.find(c.key);
        if (it != cache_.end()) return c.sign * it->second;

        // Enumerate cards.
        std::array<int, kMaxCards> As, Bs, Ps;
        int kA = 0, kB = 0, kR = 0;
        for (uint16_t mm = A; mm; ) { int v = lsb_value(mm); As[kA++]=v; mm &= ~card_mask(v); }
        for (uint16_t mm = B; mm; ) { int v = lsb_value(mm); Bs[kB++]=v; mm &= ~card_mask(v); }
        for (uint16_t mm = R; mm; ) { int v = lsb_value(mm); Ps[kR++]=v; mm &= ~card_mask(v); }

        // Phase 1: eagerly solve every child so lookup_chance becomes pure-read.
        for (int p = 0; p < kR; ++p) {
            const uint16_t R1 = static_cast<uint16_t>(R & ~card_mask(Ps[p]));
            for (int a = 0; a < kA; ++a) {
                const uint16_t A1 = static_cast<uint16_t>(A & ~card_mask(As[a]));
                for (int b = 0; b < kB; ++b) {
                    const uint16_t B1 = static_cast<uint16_t>(B & ~card_mask(Bs[b]));
                    (void)solve_chance_with_policies(A1, B1, R1);
                }
            }
        }

        // Phase 2: per-prize matrix + LP + record policy.
        const double inv_kR = 1.0 / static_cast<double>(kR);
        double F = 0.0;
        for (int p = 0; p < kR; ++p) {
            const int pv = Ps[p];
            const uint16_t R1 = static_cast<uint16_t>(R & ~card_mask(pv));
            std::vector<double> M(static_cast<size_t>(kA) * static_cast<size_t>(kB), 0.0);
            for (int a = 0; a < kA; ++a) {
                const int av = As[a];
                const uint16_t A1 = static_cast<uint16_t>(A & ~card_mask(av));
                for (int b = 0; b < kB; ++b) {
                    const int bv = Bs[b];
                    const uint16_t B1 = static_cast<uint16_t>(B & ~card_mask(bv));
                    const double sgn =
                        (av > bv) ? 1.0 : (av < bv) ? -1.0 : 0.0;
                    const double fut = lookup_chance(A1, B1, R1);
                    M[static_cast<size_t>(a) * kB + b] =
                        static_cast<double>(pv) * sgn + fut;
                }
            }
            const auto sol = lp_(kA, M.data());
            F += inv_kR * sol.v;
            // Store policy entry under the *raw* (non-canonical) (A,B,R,p) key —
            // the Python layer expects raw keys because the env hands them out.
            policy_out_.push_back(PolicyEntry{
                /*A*/ A, /*B*/ B, /*R*/ R,
                /*prize*/ pv, sol.v,
                /*x*/ std::vector<double>(sol.x.begin(), sol.x.begin() + kA),
                /*y*/ std::vector<double>(sol.y.begin(), sol.y.begin() + kB)
            });
        }
        cache_.emplace(c.key, F / c.sign);
        return F;
    }

    // Helper: when A==B (symmetric hand case F=0), we still need policy
    // entries for the policy map — every (A,A,R,p) with V=0 and Nash policies.
    void fill_policy_for_equal_hands(uint16_t A, uint16_t R) const {
        std::array<int, kMaxCards> As, Ps;
        int kA = 0, kR = 0;
        for (uint16_t mm = A; mm; ) { int v = lsb_value(mm); As[kA++]=v; mm &= ~card_mask(v); }
        for (uint16_t mm = R; mm; ) { int v = lsb_value(mm); Ps[kR++]=v; mm &= ~card_mask(v); }
        // Pre-solve children so Phase-2 lookup works miss-free.
        for (int p = 0; p < kR; ++p) {
            const uint16_t R1 = static_cast<uint16_t>(R & ~card_mask(Ps[p]));
            for (int a = 0; a < kA; ++a) {
                const uint16_t A1 = static_cast<uint16_t>(A & ~card_mask(As[a]));
                for (int b = 0; b < kA; ++b) {
                    const uint16_t B1 = static_cast<uint16_t>(A & ~card_mask(As[b]));
                    (void)solve_chance_with_policies(A1, B1, R1);
                }
            }
        }
        for (int p = 0; p < kR; ++p) {
            const int pv = Ps[p];
            const uint16_t R1 = static_cast<uint16_t>(R & ~card_mask(pv));
            std::vector<double> M(static_cast<size_t>(kA) * static_cast<size_t>(kA), 0.0);
            for (int a = 0; a < kA; ++a) {
                const int av = As[a];
                const uint16_t A1 = static_cast<uint16_t>(A & ~card_mask(av));
                for (int b = 0; b < kA; ++b) {
                    const int bv = As[b];
                    const uint16_t B1 = static_cast<uint16_t>(A & ~card_mask(bv));
                    const double sgn = (av > bv) ? 1.0 : (av < bv) ? -1.0 : 0.0;
                    const double fut = lookup_chance(A1, B1, R1);
                    M[static_cast<size_t>(a) * kA + b] =
                        static_cast<double>(pv) * sgn + fut;
                }
            }
            const auto sol = lp_(kA, M.data());
            policy_out_.push_back(PolicyEntry{
                A, A, R, pv, sol.v,
                std::vector<double>(sol.x.begin(), sol.x.begin() + kA),
                std::vector<double>(sol.y.begin(), sol.y.begin() + kA)
            });
        }
    }
};

// Optional native HiGHS LP implementation — if compiled with
// -DCXXGOOF_USE_HIGHS=ON and highs_FOUND this routine supplies an LP solver
// that does not need to cross the pybind boundary.
#if defined(CXXGOOF_HAVE_HIGHS) && CXXGOOF_HAVE_HIGHS
inline LpSolution solve_lp_native_highs(int k, const double* M) {
    // Formulation: max  v
    //    s.t.  sum_a x_a M_{a,b} >= v, for all b  (row-player guarantees v)
    //          sum_a x_a = 1
    //          x >= 0, v free.
    // Variables: x[0], ..., x[k-1], v  (length k+1).
    Highs highs;
    highs.setOptionValue("solver", kStringSimplex);
    highs.setOptionValue("output_flag", false);
    // Objective: maximize v (index k). coef(v)=1, coef(x)=0.
    const int n = k + 1;
    std::vector<double> obj(n, 0.0);
    std::vector<double> lb(n, 0.0);
    std::vector<double> ub(n, 0.0);
    obj[k] = 1.0;
    for (int i = 0; i < k; ++i) { lb[i] = 0.0; ub[i] = 1.0; }
    lb[k] = -1e30;  ub[k] = 1e30;     // v free
    highs.changeObjectiveOffset(0.0);
    // minimize sense -> we invert v sign:
    highs.changeObjectiveSense(HighsSenseType::kMinimize);
    for (int i = 0; i < n; ++i) obj[i] = -obj[i];   // now we minimize -v == max v.
    highs.addVariables(n, lb.data(), ub.data(), obj.data());
    // Constraints: k matrix-inequalities — for each b:  sum_a x_a M_{a,b} - v >= 0
    //             plus 1 simplex-equality:        sum x_a             = 1
    std::vector<int>    astart; astart.reserve(k + 1);
    std::vector<int>    aindex; aindex.reserve(k * (k + 1) + k);
    std::vector<double> avalue; avalue.reserve(k * (k + 1) + k);
    std::vector<double> lower, upper; lower.reserve(k + 1); upper.reserve(k + 1);
    for (int b = 0; b < k; ++b) {
        astart.push_back(static_cast<int>(aindex.size()));
        for (int a = 0; a < k; ++a) {
            aindex.push_back(a);
            avalue.push_back(M[a * k + b]);
        }
        aindex.push_back(k);          // -v coefficient
        avalue.push_back(-1.0);
        lower.push_back(0.0); upper.push_back(1e30);   // lhs >= 0
    }
    // Simplex equality: sum x_a = 1
    astart.push_back(static_cast<int>(aindex.size()));
    for (int a = 0; a < k; ++a) {
        aindex.push_back(a);
        avalue.push_back(1.0);
    }
    lower.push_back(1.0); upper.push_back(1.0);
    astart.push_back(static_cast<int>(aindex.size()));
    highs.addRows(k + 1, lower.data(), upper.data());
    // Add columns coefficients
    std::vector<int>    arstart(n, -1);
    std::vector<int>    arindex;
    std::vector<double> arvalue;
    // The add-row API expects us to add the matrix afterwards via
    // replaceIndicesAndValuesByRows.  Just call it:
    highs.replaceIndicesAndValuesByRows(
        static_cast<int>(astart.size()) - 1,
        astart.data(), aindex.data(), avalue.data());
    highs.run();
    LpSolution sol; sol.converged = true;
    const auto& info = highs.getInfo();
    // Highs minimised -v → optimal value = -dual_objective_value? use primal_objective.
    sol.v = -info.primal_objective_value;
    const auto& sol_obj = highs.getSolution();
    sol.x.assign(sol_obj.col_value, sol_obj.col_value + k);
    // Row player's x policy is the primal solution.  Column player y comes from
    // dual variables on the k matrix constraints (the k first row duals).
    sol.y.clear(); sol.y.reserve(k);
    for (int b = 0; b < k; ++b) {
        // Highs: for row constraint ">= lower", the dual is >= 0.
        // We normalise by dividing by the simplex multiplier sum.
        double yb = sol_obj.row_dual[b];
        if (yb < 0) yb = 0;           // numerical cleanup
        sol.y.push_back(yb);
    }
    double ysum = 0;
    for (auto y : sol.y) ysum += y;
    if (ysum > 0) for (auto& y : sol.y) y /= ysum;
    else           for (auto& y : sol.y) y = 1.0 / k;
    double xsum = 0;
    for (auto x : sol.x) xsum += x;
    if (xsum > 0) for (auto& x : sol.x) x /= xsum;
    else           for (auto& x : sol.x) x = 1.0 / k;
    return sol;
}
#endif  // CXXGOOF_HAVE_HIGHS

}  // namespace cxxgoof
