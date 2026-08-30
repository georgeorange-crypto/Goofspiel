// Author: 陈子聪 (Chen Zicong)
// Date:   2026-08-30
// Purpose: pybind11 bindings for cxxgoof (vectorised env + Nash solver).
//
// Exposed symbols (under `import goofspiel._core as _core`):
//   * estimate_complexity(N)         -> dict  (goofspiel.ComplexityReport-compatible)
//   * install_lp_solver(scipy_lp_callback)
//         The callback is a pure-Python function with signature:
//             def solve_lp(k: int, M_flat: numpy.ndarray[float64] shape=(k*k,))
//                 -> tuple[V:float, x:ndarray(k), y:ndarray(k)]
//         Once installed, all C++ Nash-solver sub-LPs are dispatched via
//         this callable (=> scipy.optimize.linprog('highs')).
//   * solve_exact_nash_policy(N, force=False) -> dict with:
//       { "N": int, "value": float, "complexity": {...},
//         "policy_map": dict[tuple(int,int,int,int)] -> tuple(float, list[float], list[float])
//         "message": str }
//   * class VectorizedEnv(num_cards: int, num_envs: int):
//         .num_cards / .num_envs
//         .reset(base_seed: int | list[int]) -> None
//         .step(actions_h: ndarray(int, (M,)), actions_b: ndarray(int, (M,)))
//             -> dict(rew_h, rew_b, winner_id, game_winner_id,
//                     human_masks, bot_masks, prize_masks, score_h, score_b,
//                     rounds, dones)  all numpy arrays shape (M,).
//         .reset_done_envs(seed_offset: int) -> None
//
// Notes:
//   * numpy-array outputs for step() are COPY-FREE: we allocate numpy arrays
//     with py::array_t<T>::allocate() and write into their mutable C++ buffers
//     before returning.  This avoids malloc/free per sub-array and the final
//     dict is ready for direct tensor conversion with torch.from_numpy(..., copy=False).
//   * policy_map for Nash uses a py::dict of py::tuple keys; for N=5 the total
//     size is ~3130 entries — tiny.  The Python wrapper SolveResult transparently
//     accepts either native solver or C++ solver policy_maps.
#include "goofspiel/goof_env.h"
#include "goofspiel/goof_nash.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace py = pybind11;
using namespace cxxgoof;

namespace {

// ---------------------------------------------------------------------------
// Installed-LP-solver global (the C++ Nash solver gets its LpSolverFn here).
// ---------------------------------------------------------------------------
std::shared_ptr<LpSolverFn> g_installed_lp;

py::object install_lp_solver(py::object fn) {
    if (!py::hasattr(fn, "__call__")) {
        throw std::invalid_argument("install_lp_solver(arg): arg must be callable.");
    }
    // Wrap into LpSolverFn.  Each call:
    //   v, x_arr, y_arr = fn(k, np.asarray(M))
    auto callback = [fn = std::move(fn)](int k, const double* M) -> LpSolution {
        py::gil_scoped_acquire gil;  // required because Nash solver main thread
                                     // may have released GIL in tight C++ loops
        // Wrap M into a 1D numpy array (no copy).
        py::array_t<double, py::array::c_style | py::array::forcecast>
            m_arr({k * k}, {sizeof(double)}, M);
        // We need to handle the case where `M` is a pointer into the solver's
        // stack-allocated scratch vector — so we create a numpy view *and then*
        // make a defensive copy inside Python before returning, because the
        // solver might reuse M while the python caller still reads it.  This is
        // done here by calling .copy() inside the pybind lambda arguments.
        py::object ret = fn(k, m_arr.attr("copy")());
        // Unpack (v, x, y)
        if (!py::isinstance<py::tuple>(ret)) {
            throw std::runtime_error(
                "LP-solver callback must return (v, x, y) tuple.");
        }
        py::tuple t = ret;
        if (t.size() != 3) {
            throw std::runtime_error(
                "LP-solver callback must return exactly 3 elements (v, x, y).");
        }
        LpSolution sol;
        sol.converged = true;
        sol.v = t[0].cast<double>();
        py::array_t<double> xnp = t[1].cast<py::array_t<double>>();
        py::array_t<double> ynp = t[2].cast<py::array_t<double>>();
        if (py::len(xnp) != k || py::len(ynp) != k) {
            throw std::runtime_error(
                "LP-solver callback returned x/y with wrong length (expected k).");
        }
        sol.x.assign(xnp.data(), xnp.data() + k);
        sol.y.assign(ynp.data(), ynp.data() + k);
        // Cleanup: clamp tiny negatives from HiGHS/scipy numerical noise.
        for (auto& v : sol.x) if (v < 0 && v > -1e-12) v = 0;
        for (auto& v : sol.y) if (v < 0 && v > -1e-12) v = 0;
        double xs = 0, ys = 0;
        for (auto v : sol.x) xs += v;
        for (auto v : sol.y) ys += v;
        if (xs > 0) for (auto& v : sol.x) v /= xs;
        else         for (auto& v : sol.x) v = 1.0 / k;
        if (ys > 0) for (auto& v : sol.y) v /= ys;
        else         for (auto& v : sol.y) v = 1.0 / k;
        return sol;
    };
    g_installed_lp = std::make_shared<LpSolverFn>(std::move(callback));
    return py::none();
}

py::dict estimate_complexity_dict(int N) {
    const auto r = estimate_complexity_level_a(N);
    py::dict d;
    d["N"] = r.N;
    d["C_N"] = r.C_N;
    d["L_N"] = r.L_N;
    d["E_N"] = r.E_N;
    d["mem_bytes_expected"]      = r.mem_bytes_expected;
    d["mem_bytes_conservative"]  = r.mem_bytes_conservative;
    d["risk_level"]              = r.risk_level;
    d["can_run_without_force"]   = r.can_run_without_force;
    auto layers = py::list();
    for (int k = 0; k <= r.N; ++k) layers.append(r.layers[k]);
    d["layers"] = layers;
    return d;
}

py::dict solve_exact_nash(int N, bool force) {
    if (!g_installed_lp) {
#if defined(CXXGOOF_HAVE_HIGHS)
        // If user compiled with native HiGHS, provide default native solver fn
        auto native_lp = [](int k, const double* M) -> LpSolution {
            return solve_lp_native_highs(k, M);
        };
        GoofspielExactSolverCpp slv(native_lp);
#else
        throw std::runtime_error(
            "cxxgoof: No LP solver installed.  "
            "Either call goofspiel._core.install_lp_solver(cb) first (cb wraps "
            "scipy.optimize.linprog), or build cxxgoof with -DCXXGOOF_USE_HIGHS=ON.");
#endif
    }
#if !defined(CXXGOOF_HAVE_HIGHS)
    GoofspielExactSolverCpp slv(*g_installed_lp);
#endif
    auto r = slv.solve_with_policy(N, force);
    py::dict out;
    out["N"] = r.N;
    out["value"] = r.root_value;
    out["message"] = r.message;
    out["complexity"] = estimate_complexity_dict(r.N);
    out["risk_level"] = r.level_b_risk;
    out["used_force"] = r.used_force;
    // policy_map: dict key = (A,B,R,prize) each int, value = (V, list x, list y)
    py::dict pm;
    for (const auto& e : r.policy_map) {
        py::tuple key = py::make_tuple(
            static_cast<int>(e.A), static_cast<int>(e.B),
            static_cast<int>(e.R), static_cast<int>(e.prize));
        py::list x, y;
        for (double v : e.x) x.append(v);
        for (double v : e.y) y.append(v);
        py::tuple val = py::make_tuple(e.V, x, y);
        pm[key] = val;
    }
    out["policy_map"] = pm;
    return out;
}

// ------- VectorizedEnv pybind class ---------------------------------------
template <typename T>
py::array_t<T> alloc_array1d(py::ssize_t n) {
    return py::array_t<T>(py::array::ShapeContainer{static_cast<py::ssize_t>(n)});
}
template <typename T>
inline T* writable(py::array_t<T>& arr) {
    return static_cast<T*>(arr.request().ptr);
}

struct VecEnvPy {
    int N;
    int M;
    std::unique_ptr<VectorizedEnv> env;
    VecEnvPy(int num_cards, int num_envs)
        : N(num_cards), M(num_envs),
          env(std::make_unique<VectorizedEnv>(num_cards, num_envs)) {}
    int num_cards() const { return N; }
    int num_envs()  const { return M; }

    void reset_seed(py::object arg) {
        if (py::isinstance<py::int_>(arg)) {
            env->reset_batch(static_cast<uint64_t>(arg.cast<int64_t>()));
            return;
        }
        if (py::isinstance<py::array_t<uint64_t>>(arg)) {
            py::array_t<uint64_t> arr = arg.cast<py::array_t<uint64_t>>();
            if (py::len(arr) != M) {
                throw std::invalid_argument(
                    "reset: seed array length must equal num_envs.");
            }
            env->reset_batch(static_cast<const uint64_t*>(arr.data()));
            return;
        }
        // Fallback: try python iterable of ints
        std::vector<uint64_t> seeds;
        seeds.reserve(M);
        for (auto item : arg) seeds.push_back(item.cast<uint64_t>());
        if (static_cast<int>(seeds.size()) != M) {
            throw std::invalid_argument(
                "reset: seed iterable length must equal num_envs.");
        }
        env->reset_batch(seeds.data());
    }

    // Reset terminal envs only (auto-respawn after .step() fills dones[i]=1).
    void reset_done_envs(py::int_ seed_offset_py) {
        const uint64_t base = static_cast<uint64_t>(seed_offset_py.cast<int64_t>());
        const uint8_t* dn = env->dones();
        for (int i = 0; i < M; ++i) {
            if (dn[i]) env->reset_single(i, base + static_cast<uint64_t>(i));
        }
    }

    py::dict step(py::array_t<int> ah_np, py::array_t<int> ab_np) {
        if (py::len(ah_np) != M || py::len(ab_np) != M) {
            throw std::invalid_argument("step: action arrays must be length M.");
        }
        const int* ah = static_cast<const int*>(ah_np.request().ptr);
        const int* ab = static_cast<const int*>(ab_np.request().ptr);
        auto rew_h = alloc_array1d<int>(M);
        auto rew_b = alloc_array1d<int>(M);
        auto w_id  = alloc_array1d<int>(M);
        auto gw_id = alloc_array1d<int>(M);
        env->step_batch(ah, ab,
                        writable(rew_h), writable(rew_b),
                        writable(w_id),  writable(gw_id));
        // Observation tensors: expose the SoA state slices.
        auto hm = alloc_array1d<uint16_t>(M);
        auto bm = alloc_array1d<uint16_t>(M);
        auto pm = alloc_array1d<uint16_t>(M);
        auto sh = alloc_array1d<uint8_t>(M);
        auto sb = alloc_array1d<uint8_t>(M);
        auto rd = alloc_array1d<uint8_t>(M);
        auto ca = alloc_array1d<uint8_t>(M);
        auto dn = alloc_array1d<uint8_t>(M);
        std::memcpy(writable(hm), env->human_masks(), sizeof(uint16_t) * M);
        std::memcpy(writable(bm), env->bot_masks(),   sizeof(uint16_t) * M);
        std::memcpy(writable(pm), env->prize_masks(), sizeof(uint16_t) * M);
        std::memcpy(writable(sh), env->score_human(), sizeof(uint8_t) * M);
        std::memcpy(writable(sb), env->score_bot(),   sizeof(uint8_t) * M);
        std::memcpy(writable(rd), env->rounds(),      sizeof(uint8_t) * M);
        std::memcpy(writable(ca), env->carry_pool(),  sizeof(uint8_t) * M);
        std::memcpy(writable(dn), env->dones(),       sizeof(uint8_t) * M);
        py::dict out;
        out["rew_h"]        = rew_h;
        out["rew_b"]        = rew_b;
        out["winner_id"]    = w_id;
        out["game_winner"]  = gw_id;
        out["human_masks"]  = hm;
        out["bot_masks"]    = bm;
        out["prize_masks"]  = pm;
        out["score_h"]      = sh;
        out["score_b"]      = sb;
        out["rounds"]       = rd;
        out["carry_pool"]   = ca;   // tie-rollover pool per env (uint8 0..91)
        out["dones"]        = dn;
        return out;
    }

    // ---- Raw single-step accessor for tests (index i). ----
    py::dict get_state(int i) const {
        if (i < 0 || i >= M) throw std::out_of_range("i out of range.");
        py::dict d;
        d["human_mask"] = int(env->human_masks()[i]);
        d["bot_mask"]   = int(env->bot_masks()[i]);
        d["prize_mask"] = int(env->prize_masks()[i]);
        d["score_h"]    = int(env->score_human()[i]);
        d["score_b"]    = int(env->score_bot()[i]);
        d["round"]      = int(env->rounds()[i]);
        d["carry_pool"] = int(env->carry_pool()[i]);
        d["done"]       = bool(env->dones()[i]);
        return d;
    }
};

}  // namespace <anon>

// ==========================================================================
// Module
// ==========================================================================
PYBIND11_MODULE(_core, m) {
    m.doc() = R"pbdoc(
cxxgoof native extension — fast Goofspiel.

  - VectorisedEnv for training rollouts (M parallel envs, SIMD loop).
  - Exact Nash solver (two-phase canonical cache + swap-signed recursion).
  - Optional native HiGHS LP via CXXGOOF_USE_HIGHS=ON; otherwise the user
    must call install_lp_solver(cb) where cb wraps scipy.optimize.linprog.

Typical usage:

    from goofspiel import _core
    import numpy as np
    def my_lp(k, M):
        # scipy-based solver (see goofspiel/_cxx.py for a pre-packaged one).
        from goofspiel.solver import solve_zero_sum_matrix
        return solve_zero_sum_matrix(M.reshape(k, k))
    _core.install_lp_solver(my_lp)

    # Training:
    venv = _core.VectorizedEnv(13, 4096)
    venv.reset(123)
    for t in range(10000):
        a_h, a_b = policy(torch.from_numpy(venv.latest_obs()))
        out = venv.step(a_h.numpy(), a_b.numpy())
        venv.reset_done_envs(seed_offset=t * venv.num_envs)

    # Nash exact:
    r = _core.solve_exact_nash(5, force=False)
    print("root value:", r["value"])  # should be 0.0 (symmetric root)
)pbdoc";

    m.def("estimate_complexity", &estimate_complexity_dict, py::arg("N"),
          "Level-A closed-form complexity estimate (states / memory / risk).");
    m.def("install_lp_solver", &install_lp_solver, py::arg("callback"),
          "Install a (k, M_flat) -> (v, x, y) callable used by the C++ Nash solver.");
    m.def("solve_exact_nash", &solve_exact_nash,
          py::arg("N"), py::arg("force") = false,
          "Run exact Nash recursion. Returns dict with value + policy_map.");

    py::class_<VecEnvPy>(m, "VectorizedEnv",
        "Struct-of-arrays batched Goofspiel environment for training throughput.")
        .def(py::init<int, int>(),
             py::arg("num_cards"), py::arg("num_envs"))
        .def_property_readonly("num_cards", &VecEnvPy::num_cards)
        .def_property_readonly("num_envs",  &VecEnvPy::num_envs)
        .def("reset", &VecEnvPy::reset_seed,
             py::arg("seeds_or_base") = py::int_(0),
             "Reset all envs. arg is either a single int (base seed i -> base+i) "
             "or numpy uint64 array / iterable length M of per-env seeds.")
        .def("step", &VecEnvPy::step,
             py::arg("actions_h"), py::arg("actions_b"),
             "Advance all M envs by one simultaneous step. Returns dict with "
             "M-length numpy arrays for rewards/state SoA fields.")
        .def("reset_done_envs", &VecEnvPy::reset_done_envs,
             py::arg("seed_offset") = py::int_(0),
             "Auto-respawn: for every env i where dones[i]==1, reset_single(i).")
        .def("state_at", &VecEnvPy::get_state, py::arg("i"),
             "Return dict with decoded scalar state for env i (for tests/debug).");
}
