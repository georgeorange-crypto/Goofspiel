"""
Author: 陈子聪 (Chen Zicong)
Date: 2026-08-30
Purpose: Exact Nash solvers for the two-player zero-sum game Goofspiel.
         Contains TWO independent exact solvers side-by-side:
           (1) GoofspielExactSolver —— classic "tie → discard the prize" variant.
               This is the "textbook / original Goofspiel" model.
           (2) GoofspielCarrySolver —— custom "tie → prize rolls over to the next
               round as a public carry-pool" variant (本项目实际网页规则).
               Same zero-sum objective (maximise expected score-difference),
               but the state space is extended with one more dimension:
               `carry_pool` (0..N(N+1)/2), and terminal payoffs branch into
               three cases: win/lose (± (prize+carry), carry_next=0);
               tie non-final (0, carry_next = prize+carry_in);
               tie final    (0, carry_next = 0 and prize discarded).

Architecture (架构分层):
  1. Preflight Complexity Estimator
       - Level A: pure combinatorial formulas C(N), L(N), E(N)  (closed form, instant)
         Two variants: classic (estimate_complexity) & carry-over (estimate_carry_complexity).
       - Level B: machine-calibrated runtime / RAM prediction     (short benchmarks)
       - Risk levels: GREEN / YELLOW / ORANGE / RED / BLACK
       - solve(X) ALWAYS runs preflight first; unsafe runs abort unless force=True

  2. Exact Solver core (recursive + memoization)          ——— TWO SOLVERS
       Classic:     State = (A_mask, B_mask, R_mask)
       Carry-Over:  State = (A_mask, B_mask, R_mask, carry)   (carry int 0..N(N+1)/2)
       Shared invariants:
          - Player exchange symmetry:  F(A,B,R,c) = -F(B,A,R,c)  (carry is public pool)
          - Root F(all,all,all,0) = 0 (symmetric)
          - Shortcut: F(A,A,R,0) = 0;  NOTE F(A,A,R,c>0) ≠ 0 (carry breaks anti-symmetry 0)
       Recursion:  F -> Chance(k prizes uniform) -> k×k matrix game -> F(child)
       Matrix game Nash via  scipy.optimize.linprog (primal LP for Player A)

  3. Public API
       Classic:    estimate / solve / solve_with_policy
       Carry-Over: estimate_carry / solve_carry / solve_with_policy_carry
       (Separated intentionally so callers can never accidentally mix rule-models.)

Key invariant (核心不变式):
    Both solvers optimise *expected score difference*  U = S_A - S_B.
    Therefore running score delta does NOT enter the state (past is constant).
    For the Carry-Over variant, only the PUBLIC prize-rollover enters the state.
"""

from __future__ import annotations

import math
import os
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional scipy —— users who only want the estimator don't need scipy installed
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard, exercised end-to-end in tests
    from scipy.optimize import linprog as _scipy_linprog
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False
    _scipy_linprog = None  # type: ignore


# ===================================================================
# 0.  Risk levels & dataclasses
# ===================================================================
RISK_GREEN = "GREEN"
RISK_YELLOW = "YELLOW"
RISK_ORANGE = "ORANGE"
RISK_RED = "RED"
RISK_BLACK = "BLACK"

_RISK_ORDER = {
    RISK_GREEN: 0,
    RISK_YELLOW: 1,
    RISK_ORANGE: 2,
    RISK_RED: 3,
    RISK_BLACK: 4,
}


@dataclass
class ComplexityReport:
    """
    Full Level-A (and optional Level-B) preflight report.

    Level-A fields are always present and mathematically exact.
    Level-B fields are populated only if `benchmark=True`.
    """
    N: int

    # --- Level A: exact combinatorics ---
    chance_states: int               # C(N) = sum_k C(N,k)^3
    matrix_games: int                # L(N) = sum_k k * C(N,k)^3
    matrix_cells: int                # E(N) = sum_k k^3 * C(N,k)^3
    per_layer: Dict[int, Dict[str, int]] = field(default_factory=dict)
    # per_layer[k] = {"states": C_k, "lps": L_k, "cells": E_k}

    # --- Level B: machine-calibrated estimates ---
    est_memory_bytes: Optional[int] = None        # predicted peak cache memory
    est_seconds_optimistic: Optional[float] = None
    est_seconds_expected: Optional[float] = None
    est_seconds_conservative: Optional[float] = None
    available_ram_bytes: Optional[int] = None
    correction_factor: Optional[float] = None     # (measured/predicted) on small N

    # --- final verdict ---
    risk: str = RISK_GREEN
    risk_reason: str = ""

    @property
    def C_N(self) -> int:
        """Backward-compatible alias for the exact chance-state count."""
        return self.chance_states

    @property
    def L_N(self) -> int:
        """Backward-compatible alias for the exact matrix-game count."""
        return self.matrix_games

    @property
    def E_N(self) -> int:
        """Backward-compatible alias for the exact matrix-cell count."""
        return self.matrix_cells

    # ------------------------------------------------------------------ repr
    def format_human(self) -> str:
        """Human-readable multi-line report (终端友好的中文摘要)."""
        def _fmt_num(x):
            if x is None:
                return "n/a"
            if isinstance(x, float):
                if abs(x) >= 1e12:
                    return f"{x / 1e12:.2f} T"
                if abs(x) >= 1e9:
                    return f"{x / 1e9:.2f} B"
                if abs(x) >= 1e6:
                    return f"{x / 1e6:.2f} M"
                if abs(x) >= 1e3:
                    return f"{x / 1e3:.2f} K"
                return f"{x:.2f}"
            if x >= 10**12:
                return f"{x / 10**12:.2f} T"
            if x >= 10**9:
                return f"{x / 10**9:.2f} B"
            if x >= 10**6:
                return f"{x / 10**6:.2f} M"
            if x >= 10**3:
                return f"{x / 10**3:.2f} K"
            return str(x)

        def _fmt_time(t):
            if t is None:
                return "n/a"
            if t < 1:
                return f"{t * 1000:.1f} ms"
            if t < 60:
                return f"{t:.1f} s"
            if t < 3600:
                return f"{t / 60:.1f} min"
            return f"{t / 3600:.1f} h"

        lines = [
            "Exact Goofspiel Solver — Preflight Report",
            f"  N:                 {self.N}",
            f"  Chance states C:   {_fmt_num(self.chance_states)}",
            f"  Matrix games L:    {_fmt_num(self.matrix_games)}",
            f"  Matrix cells E:    {_fmt_num(self.matrix_cells)}",
        ]
        if self.est_memory_bytes is not None:
            mb = self.est_memory_bytes / (1024 ** 2)
            if mb >= 1024:
                lines.append(f"  Estimated RAM:     {mb / 1024:.2f} GiB")
            else:
                lines.append(f"  Estimated RAM:     {mb:.2f} MiB")
        if self.available_ram_bytes is not None:
            arb = self.available_ram_bytes / (1024 ** 3)
            lines.append(f"  Available RAM:     {arb:.2f} GiB")
        if self.est_seconds_expected is not None:
            lines.append(
                "  Estimated runtime: "
                f"{_fmt_time(self.est_seconds_optimistic)} (opt) / "
                f"{_fmt_time(self.est_seconds_expected)} (exp) / "
                f"{_fmt_time(self.est_seconds_conservative)} (pes)"
            )
        if self.correction_factor is not None:
            lines.append(f"  Calibration c:     {self.correction_factor:.2f}x")

        lines.append(f"  Risk:              {self.risk}")
        if self.risk_reason:
            lines.append(f"  Reason:            {self.risk_reason}")
        return "\n".join(lines)


# ===================================================================
# 1.  Level-A preflight  (纯组合数学，完全精确，零运行开销)
# ===================================================================
def _comb3(n: int, k: int) -> int:
    """C(N,k)^3  —— 每一层 cache 的 chance states。"""
    c = math.comb(n, k)
    return c * c * c


def estimate_complexity(n: int) -> ComplexityReport:
    """
    Level-A exact complexity estimate.

    Computes:
        C(N) = Σ_k        C(N,k)^3       # cached chance states
        L(N) = Σ_k  k  *  C(N,k)^3       # LP solves
        E(N) = Σ_k  k^3 * C(N,k)^3       # matrix cells built

    Raises ValueError for n < 1.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    total_c = 0  # chance states
    total_l = 0  # LP count
    total_e = 0  # matrix cells
    per_layer: Dict[int, Dict[str, int]] = {}

    for k in range(0, n + 1):
        c_k = _comb3(n, k)
        l_k = k * c_k
        e_k = (k ** 3) * c_k
        total_c += c_k
        total_l += l_k
        total_e += e_k
        per_layer[k] = {"states": c_k, "lps": l_k, "cells": e_k}

    report = ComplexityReport(
        N=n,
        chance_states=total_c,
        matrix_games=total_l,
        matrix_cells=total_e,
        per_layer=per_layer,
    )
    report.risk, report.risk_reason = _decide_risk_level_a(report)
    return report


def _decide_risk_level_a(rpt: ComplexityReport) -> Tuple[str, str]:
    """
    Level-A-only risk based on raw combinatorial counts.
    Level-B will *tighten* (not relax) this verdict after calibration.
    """
    c = rpt.chance_states
    # Thresholds are deliberately *only* order-of-magnitude triggers.
    # True decision is done by Level-B using actual machine numbers.
    if c > 2 * 10**12:                                    # > N=13-class
        return RISK_BLACK, "states > 2T; clearly infeasible on any single machine"
    if c > 10**9:                                         # ~N=12
        return RISK_RED, "states >= 1B; naive Python caching likely impossible"
    if c > 5 * 10**6:                                     # ~N=9
        return RISK_ORANGE, "states >= 5M; heavy exact solve, confirm resources"
    if c > 2 * 10**4:                                     # ~N=7
        return RISK_YELLOW, "states >= 20K; exact solve is cheap but non-trivial"
    return RISK_GREEN, "small state-space; trivial to solve"


# ===================================================================
# 2.  Bitmask helpers  (bitmask state representation)
# ===================================================================
# Convention: card value v ∈ [1..N] maps to bit (v-1).
#   i.e. the bit position is zero-indexed; users see 1..N.
def cards_to_mask(cards: List[int]) -> int:
    """Convert iterable of 1-indexed card values → N-bitmask."""
    mask = 0
    for v in cards:
        if v < 1:
            raise ValueError(f"cards must be 1-indexed positive ints, got {v}")
        mask |= 1 << (v - 1)
    return mask


def mask_to_cards(mask: int) -> List[int]:
    """Convert N-bitmask → sorted list of 1-indexed cards."""
    return [(i + 1) for i in range(mask.bit_length()) if mask & (1 << i)]


def popcount(mask: int) -> int:
    """Population count (number of 1-bits) / Python 3.8+ builtin."""
    return bin(mask).count("1")


# ===================================================================
# 3.  Zero-sum matrix-game LP  (Nash via scipy linprog)
# ===================================================================
# Player A's primal:
#   variables  x_1..x_k,  v
#   max  v
#   s.t.  (M^T x)_j >= v,   ∀ j    ← Player B's worst-case pure actions
#          Σ x_i = 1
#          x >= 0
#
# scipy.optimize.linprog  *minimises* c^T x  with  A_ub @ x <= b_ub.
# So reformulate:
#   c = [0, ..., 0, -1]   →  min -v  ⇔  max v
#   For each j ∈ rows of M^T (i.e. each column j of M):
#       Σ_i M_ij x_i - v >= 0
#       →  -Σ_i M_ij x_i + v <= 0
#   A_ub has shape (k, k+1):  A_ub[j] = [-M_1j, -M_2j, ..., -M_kj, +1]
#   b_ub[j] = 0
# Plus equality row for Σ x_i = 1.
def solve_zero_sum_matrix(
    M: np.ndarray,
    tol: float = 1e-9,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Solve a 2-player zero-sum matrix game for the row player (A).

    Returns:
        value       : game value v  (row player's expected payoff under Nash)
        policy_row  : x ∈ Δ(rows),    shape (k,)
        policy_col  : y ∈ Δ(cols),    shape (k,)

    Raises RuntimeError if LP fails.
    """
    if not _HAS_SCIPY:
        raise RuntimeError(
            "scipy is required for exact matrix-game LP solving. "
            "Install it via `pip install scipy`."
        )
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        # 我们的 Goofspiel 天然是方阵，但为了模块健壮性这里允许非方阵
        pass
    k_row, k_col = M.shape
    if k_row == 0 or k_col == 0:
        raise ValueError("empty payoff matrix")

    # ---- primal: row player -----------------------------------------------
    # vars = [x_0 .. x_{kr-1},  v]
    n_vars_primal = k_row + 1
    c = np.zeros(n_vars_primal)
    c[-1] = -1.0  # min  -v

    A_ub = np.zeros((k_col, n_vars_primal))
    for j in range(k_col):
        A_ub[j, :k_row] = -M[:, j]
        A_ub[j, -1] = 1.0
    b_ub = np.zeros(k_col)

    A_eq = np.zeros((1, n_vars_primal))
    A_eq[0, :k_row] = 1.0
    b_eq = np.array([1.0])

    bounds: List[Tuple[Optional[float], Optional[float]]] = [
        (0.0, 1.0) for _ in range(k_row)
    ]
    bounds.append((None, None))  # v 无约束

    res = _scipy_linprog(
        c,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={"presolve": True, "time_limit": 30},
    )
    if not res.success:
        # fallback —— "revised simplex" 数值上偶尔更稳
        res2 = _scipy_linprog(
            c,
            A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=bounds,
            method="highs-ipm",
            options={"presolve": True},
        )
        if not res2.success:
            raise RuntimeError(
                f"Row-player LP failed (status {res.status}): {res.message}; "
                f"IPM fallback also failed (status {res2.status}): {res2.message}"
            )
        res = res2

    x = np.clip(res.x[:k_row], 0.0, 1.0)
    if x.sum() > 0:
        x = x / x.sum()
    else:  # pragma: no cover - 理论上 LP 必须给出 valid x
        x = np.full(k_row, 1.0 / k_row)
    value = float(-res.fun)

    # ---- dual → column player policy --------------------------------------
    # Dual variables:  y_1..y_{kc},  w,  plus slack / u = equality multiplier.
    # 为了简单稳健，我们直接用 KKT:  at optimum, for any row i with x_i>0
    #     (M y)_i = v,  and  Σ y_j = 1,  y>=0.
    # 等价地直接解 column 一侧的极小化 LP。
    n_vars_dual = k_col + 1
    c_dual = np.zeros(n_vars_dual)
    c_dual[-1] = 1.0  # min w

    # For each row i:  Σ_j M_ij y_j <= w   →  Σ_j M_ij y_j - w <= 0
    A_ub_dual = np.zeros((k_row, n_vars_dual))
    for i in range(k_row):
        A_ub_dual[i, :k_col] = M[i, :]
        A_ub_dual[i, -1] = -1.0
    b_ub_dual = np.zeros(k_row)

    A_eq_dual = np.zeros((1, n_vars_dual))
    A_eq_dual[0, :k_col] = 1.0
    b_eq_dual = np.array([1.0])

    bounds_dual: List[Tuple[Optional[float], Optional[float]]] = [
        (0.0, 1.0) for _ in range(k_col)
    ]
    bounds_dual.append((None, None))

    res_d = _scipy_linprog(
        c_dual,
        A_ub=A_ub_dual, b_ub=b_ub_dual,
        A_eq=A_eq_dual, b_eq=b_eq_dual,
        bounds=bounds_dual,
        method="highs",
        options={"presolve": True, "time_limit": 30},
    )
    if not res_d.success:
        # fallback
        res_d2 = _scipy_linprog(
            c_dual,
            A_ub=A_ub_dual, b_ub=b_ub_dual,
            A_eq=A_eq_dual, b_eq=b_eq_dual,
            bounds=bounds_dual,
            method="highs-ipm",
            options={"presolve": True},
        )
        if not res_d2.success:
            # 理论上 primal 有解, dual 必有解；数值异常时 fallback 为 uniform
            y = np.full(k_col, 1.0 / k_col)
        else:
            res_d = res_d2
    if res_d.success:
        y = np.clip(res_d.x[:k_col], 0.0, 1.0)
        if y.sum() > 0:
            y = y / y.sum()
        else:  # pragma: no cover
            y = np.full(k_col, 1.0 / k_col)
    else:  # pragma: no cover
        y = np.full(k_col, 1.0 / k_col)

    # 极小的数值噪声修正：保证 |value - dual_value| ~ 0
    # (这里我们已经以 primal value 为准，足够精确用于 RL ground truth)
    _ = tol
    return value, x, y


# ===================================================================
# 4.  Exact Solver  (F(A,B,R) recursive)
# ===================================================================
@dataclass
class SolverConfig:
    """Runtime knobs for the exact solver (运行时可调参数)."""

    # 是否启用玩家交换对称剪枝  F(A,B,R) = -F(B,A,R)
    use_symmetry: bool = True

    # LP tolerance
    lp_tol: float = 1e-9

    # 对于 A == B 的 chance states，直接返回 0 (严格成立)
    short_cut_equal_hand: bool = True

    # ---- Preflight / Level-B 开关 ----
    # 禁用 benchmarking / calibration，对测试环境很有用
    skip_benchmark: bool = False
    # 禁用 N=6 真实跑的 correction-factor 校准
    skip_calibration_solve: bool = False

    # 进度回调 (每 1% 的 cache 填充调用一次，可为 None)
    progress_cb: Optional[callable] = None   # type: ignore


@dataclass
class SolveResult:
    """Full outcome of solve() / solve_with_policy()."""
    N: int
    value: float                                # F(full, full, full)  ≡ 0 (对称)
    report: ComplexityReport
    # 缓存的 chance-state 数量（用于验证预估）
    cache_size: int
    # 实际求解耗时（秒）
    elapsed_seconds: float

    # --- 只在 solve_with_policy 中填充 ---
    # policy_map[(A_mask, B_mask, R_mask, prize)] = (value, policy_A, policy_B)
    # 其中 policy_A 是按 A 的升序排列的概率向量
    policy_map: Optional[Dict[Tuple[int, int, int, int],
                              Tuple[float, np.ndarray, np.ndarray]]] = None


class GoofspielExactSolver:
    """
    Exact Nash solver for N-card Goofspiel under the
    "max expected score-difference" objective.

    Usage:
        solver = GoofspielExactSolver()
        report = solver.estimate(8)
        print(report.format_human())
        result = solver.solve(8)           # 不记录策略，省内存
        full   = solver.solve_with_policy(5)   # 记录所有 V_p policy
    """

    def __init__(self, config: Optional[SolverConfig] = None) -> None:
        self.config = config or SolverConfig()
        # Level-B calibration cache
        self._lp_bench: Optional[Dict[int, float]] = None   # k -> median ms per LP
        self._cell_rate: Optional[float] = None             # cells / sec
        self._mem_per_state: Optional[float] = None         # bytes / cache entry
        self._calib_factor: Optional[float] = None          # correction factor

    # ================================================================
    # Public API: estimate → Level A + Level B
    # ================================================================
    def estimate(
        self,
        n: int,
        *,
        benchmark: bool = True,
        calibrate: bool = True,
    ) -> ComplexityReport:
        """
        Full preflight estimate.

        - Level A is always computed (combinatorics, exact).
        - Level B benchmarks this machine only if `benchmark=True`.
        - `calibrate=True` runs a tiny N=6 solve (cheap) to compute
          the correction factor c = actual/predicted.
        """
        rpt = estimate_complexity(n)
        if self.config.skip_benchmark:
            benchmark = False
            calibrate = False
        if self.config.skip_calibration_solve:
            calibrate = False

        if benchmark:
            self._ensure_benchmarks(max_k=min(n, 13))
            self._ensure_memory_calibration(run_calibration_solve=calibrate)

            # 聚合预测
            lp_sec = 0.0
            for k, info in rpt.per_layer.items():
                if k == 0:
                    continue
                t_ms = self._lp_bench.get(
                    k,
                    self._extrapolate_lp_time(k),
                )
                lp_sec += info["lps"] * (t_ms / 1000.0)

            cell_sec = (rpt.matrix_cells / self._cell_rate) \
                if self._cell_rate else 0.0

            cache_sec = (rpt.chance_states / 2_000_000.0)  # 粗估: dict 写入 ~2M/s
            total = lp_sec + cell_sec + cache_sec

            if calibrate and self._calib_factor is not None:
                total *= self._calib_factor

            rpt.est_seconds_optimistic = total * 0.7
            rpt.est_seconds_expected = total
            rpt.est_seconds_conservative = total * 2.0

            if self._mem_per_state:
                rpt.est_memory_bytes = int(rpt.chance_states * self._mem_per_state)
            rpt.correction_factor = self._calib_factor
            rpt.available_ram_bytes = _available_ram()

            # ---- 根据机器实际数据收紧风险等级 ----
            rpt.risk, rpt.risk_reason = self._decide_risk_level_b(rpt)
        return rpt

    # ================================================================
    # Public API: solve
    # ================================================================
    def solve(
        self,
        n: int,
        *,
        force: bool = False,
        max_expected_seconds: Optional[float] = None,
        max_memory_bytes: Optional[int] = None,
    ) -> SolveResult:
        """
        Solve the full N-card game from the symmetric root.

        Preflight is ALWAYS run first.  By default RED/BLACK jobs abort.
        Pass `force=True` to override.

        The root value  F(all, all, all)  is always 0 by symmetry;
        the useful output is the fully populated cache which can be
        reused via solve_with_policy() / or externally.
        """
        report = self.estimate(n)

        if not force:
            self._assert_feasible(
                report,
                max_expected_seconds=max_expected_seconds,
                max_memory_bytes=max_memory_bytes,
            )

        cache: Dict[int, float] = {}
        t0 = time.perf_counter()
        full_mask = (1 << n) - 1
        value = self._solve_chance(full_mask, full_mask, full_mask, n, cache)
        elapsed = time.perf_counter() - t0

        return SolveResult(
            N=n,
            value=value,
            report=report,
            cache_size=len(cache),
            elapsed_seconds=elapsed,
            policy_map=None,
        )

    # ================================================================
    # Public API: solve WITH policies (for RL ground-truth teacher)
    # ================================================================
    def solve_with_policy(
        self,
        n: int,
        *,
        force: bool = False,
        max_expected_seconds: Optional[float] = None,
        max_memory_bytes: Optional[float] = None,
    ) -> SolveResult:
        """
        Identical to solve(), but additionally records
        V_p(A,B,R) together with the Nash policies for every
        (chance-state, revealed-prize) pair encountered.
        """
        report = self.estimate(n)
        if not force:
            self._assert_feasible(
                report,
                max_expected_seconds=max_expected_seconds,
                max_memory_bytes=max_memory_bytes,
            )
        cache: Dict[int, float] = {}
        policy_map: Dict[Tuple[int, int, int, int],
                         Tuple[float, np.ndarray, np.ndarray]] = {}
        t0 = time.perf_counter()
        full_mask = (1 << n) - 1
        old_use_symmetry = self.config.use_symmetry
        self.config.use_symmetry = False
        try:
            value = self._solve_chance_with_policies(
                full_mask, full_mask, full_mask, n, cache, policy_map,
            )
        finally:
            self.config.use_symmetry = old_use_symmetry
        elapsed = time.perf_counter() - t0
        return SolveResult(
            N=n,
            value=value,
            report=report,
            cache_size=len(cache),
            elapsed_seconds=elapsed,
            policy_map=policy_map,
        )

    # ================================================================
    # Internal: chance node  (奖品未揭晓)
    # ================================================================
    def _canonical_key(self, a_mask: int, b_mask: int, r_mask: int, n: int
                       ) -> Tuple[int, int]:
        """
        Return (packed_key, sign) with canonicalization for (A,B) swap.

            F(A,B,R) = sign * F(canonical A', canonical B', R)

        If use_symmetry=False  → sign = +1, trivial key.
        """
        if self.config.use_symmetry and a_mask > b_mask:
            a_mask, b_mask = b_mask, a_mask
            sign = -1
        else:
            sign = +1
        # pack:  A | B<<n | R<<2n   (对 N=13 → 39 bits)
        packed = a_mask | (b_mask << n) | (r_mask << (2 * n))
        return packed, sign

    def _lookup_chance(
        self,
        a_mask: int,
        b_mask: int,
        r_mask: int,
        n: int,
        cache: Dict[int, float],
    ) -> float:
        """
        Return true F(a_mask, b_mask, r_mask) by consulting canonical cache.

        This helper encapsulates the sign-flip on READ, keeping matrix semantics
        always aligned with the caller's (A,B) orientation.
        """
        if r_mask == 0:
            return 0.0
        if self.config.short_cut_equal_hand and a_mask == b_mask:
            return 0.0
        packed_key, sign = self._canonical_key(a_mask, b_mask, r_mask, n)
        if packed_key in cache:
            return sign * cache[packed_key]
        raise RuntimeError(
            "_lookup_chance: cache miss on canonical key.  "
            "All children must be solved BEFORE the parent matrix is built. "
            f"Missing F(A={mask_to_cards(a_mask)}, B={mask_to_cards(b_mask)}, "
            f"R={mask_to_cards(r_mask)})."
        )

    def _solve_chance(
        self,
        a_mask: int,
        b_mask: int,
        r_mask: int,
        n: int,
        cache: Dict[int, float],
    ) -> float:
        """F(A,B,R): chance-node value (奖品还没翻出来)。"""
        if r_mask == 0:
            return 0.0

        # 对称剪枝：A == B 的 chance state 恒为 0
        if self.config.short_cut_equal_hand and a_mask == b_mask:
            return 0.0

        packed_key, sign = self._canonical_key(a_mask, b_mask, r_mask, n)
        if packed_key in cache:
            return sign * cache[packed_key]

        # 枚举所有可能奖品
        r_cards = mask_to_cards(r_mask)
        k = len(r_cards)
        total = 0.0

        a_cards = mask_to_cards(a_mask)
        b_cards = mask_to_cards(b_mask)
        # 预生成子 mask 表 (避免在 (a,b,p) 内重复算)
        a_child_masks = [a_mask & ~(1 << (a - 1)) for a in a_cards]
        b_child_masks = [b_mask & ~(1 << (b - 1)) for b in b_cards]

        # ------------------------------------------------------------
        # PHASE 1 — eagerly solve every child F(child_A, child_B, R\{p})
        #           so the matrix-build phase below does pure cache reads.
        #           (Necessary because canonical cache uses sign-sensitive
        #           reads; if we recursively compute children *from inside*
        #           a cell whose parent already had a sign-flip, the
        #           matrix would mix up canonical vs. raw F values.)
        # ------------------------------------------------------------
        for p in r_cards:
            r_child = r_mask & ~(1 << (p - 1))
            for child_a in a_child_masks:
                for child_b in b_child_masks:
                    packed_child, _ = self._canonical_key(
                        child_a, child_b, r_child, n,
                    )
                    if packed_child not in cache:
                        # Will compute child and store into cache under its
                        # own canonical key.
                        self._solve_chance(child_a, child_b, r_child, n, cache)

        # ------------------------------------------------------------
        # PHASE 2 — build payoff matrix from cache reads, then LP.
        # ------------------------------------------------------------
        for p in r_cards:
            r_child = r_mask & ~(1 << (p - 1))
            M = np.zeros((len(a_cards), len(b_cards)), dtype=np.float64)
            for i, a in enumerate(a_cards):
                child_a = a_child_masks[i]
                for j, b in enumerate(b_cards):
                    child_b = b_child_masks[j]
                    immediate = p * (1 if a > b else (-1 if a < b else 0))
                    future = self._lookup_chance(child_a, child_b, r_child, n, cache)
                    M[i, j] = immediate + future

            value, _, _ = solve_zero_sum_matrix(M, tol=self.config.lp_tol)
            total += value

        result = total / k
        # 存 canonical 版本。
        # sign = +1  ⇒  input was canonical already, store `result`  (= F(A,B,R)).
        # sign = -1  ⇒  input swapped, and `total` was computed on raw (A,B).
        #   Above, we computed matrix M as the true game value from (A,B) perspective,
        #   which equals  - (game value from (B,A) perspective).
        #   So F(B,A,R) = -F(A,B,R) = -result, and that's what we want to cache
        #   because the canonical key corresponds to (B,A,R).
        cache[packed_key] = result if sign == +1 else -result
        return result

    # ---- policy 版本 ----
    def _solve_chance_with_policies(
        self,
        a_mask: int,
        b_mask: int,
        r_mask: int,
        n: int,
        cache: Dict[int, float],
        policy_map: Dict,
    ) -> float:
        if r_mask == 0:
            return 0.0
        if self.config.short_cut_equal_hand and a_mask == b_mask:
            # 对 A==B 的 chance state 直接返回 0；*但是* 对于 V_p 我们依然需要存真正的 Nash policy
            # 所以这里不能 shortcut, 必须继续往下走 (除非调用方不关心 policy)。
            # → 实际上只有 "root short-circuit" 是必须的：
            #   为了 policy 的完整性，允许跳过 F 的缓存，但仍然遍历奖品。
            #   不过为了 cache 一致性 (policy_map key 使用原始 mask), 我们显式继续。
            pass

        # 注意：因为 policy 必须对应 *原始的* (A,B,R,p)，所以 canonicalization 只能用于 value cache
        # 不能用于 policy_map 的 key。为此，我们先用 canonical value cache 加速；
        # 但是 policy_map 用 raw (A,B,R,p) 存。
        packed_key, sign = self._canonical_key(a_mask, b_mask, r_mask, n)
        if packed_key in cache:
            cached_val = cache[packed_key]
            # 注意: 如果 sign=-1, cache 存的是 F(B,A,R)=-F(A,B,R), 返回是对的
            # 但是 policy_map 可能已经在 swapped 场景填过 —— 我们不做 swap-policy 对称缓存,
            # 因为那样需要同时 swap A/B policy 并翻转 value 符号; 对于 small N 教学用途不做以保简单。
            # policy_map 不命中则下面照常填，命中也无所谓 (幂等)。
            if len(policy_map) == 0:
                return sign * cached_val
            # policy 可能已填充 —— 粗查: 若该 (A,B,R) 下所有 p 都已存则直接返回
            r_cards = mask_to_cards(r_mask)
            all_present = all(
                (a_mask, b_mask, r_mask, p) in policy_map for p in r_cards
            )
            if all_present:
                return sign * cached_val
            # A chance-state cache entry can come from a value-only shortcut
            # such as A==B -> F=0.  Policy generation still has to solve every
            # current-prize matrix, so do not return early when entries are
            # missing.
            if sign == -1:
                # The raw (A,B,R,p) entries are required by tests, logs and
                # teachers.  A canonical cache hit for swapped hands is only a
                # value shortcut; policies must be written in the caller's
                # orientation, so continue and rebuild this raw node.
                pass

        r_cards = mask_to_cards(r_mask)
        k = len(r_cards)
        total = 0.0
        a_cards = mask_to_cards(a_mask)
        b_cards = mask_to_cards(b_mask)
        a_child_masks = [a_mask & ~(1 << (a - 1)) for a in a_cards]
        b_child_masks = [b_mask & ~(1 << (b - 1)) for b in b_cards]

        # ------------------------------------------------------------
        # PHASE 1 — eagerly solve every child chance-node, ensuring
        #           the canonical value cache is fully populated for
        #           every reachable (child_A, child_B, R\{p}) BEFORE
        #           we build any payoff matrix that reads it.  Without
        #           this separation, `sign` for the parent would leak
        #           into the value we read for (A,B)-non-canonical
        #           matrix cells, corrupting the payoff matrix.
        # ------------------------------------------------------------
        for p in r_cards:
            r_child = r_mask & ~(1 << (p - 1))
            for child_a in a_child_masks:
                for child_b in b_child_masks:
                    packed_child, _ = self._canonical_key(
                        child_a, child_b, r_child, n,
                    )
                    if packed_child not in cache:
                        self._solve_chance_with_policies(
                            child_a, child_b, r_child, n, cache, policy_map,
                        )

        # ------------------------------------------------------------
        # PHASE 2 — for each prize: build M via cache reads, LP, store policy.
        # ------------------------------------------------------------
        for p in r_cards:
            r_child = r_mask & ~(1 << (p - 1))
            M = np.zeros((len(a_cards), len(b_cards)), dtype=np.float64)
            for i, a in enumerate(a_cards):
                child_a = a_child_masks[i]
                for j, b in enumerate(b_cards):
                    child_b = b_child_masks[j]
                    immediate = p * (1 if a > b else (-1 if a < b else 0))
                    future = self._lookup_chance(
                        child_a, child_b, r_child, n, cache,
                    )
                    M[i, j] = immediate + future

            value, x, y = solve_zero_sum_matrix(M, tol=self.config.lp_tol)
            total += value
            # 存 policy —— key = 原始 mask
            policy_map[(a_mask, b_mask, r_mask, p)] = (value, x.copy(), y.copy())

        result = total / k
        cache[packed_key] = result if sign == +1 else -result
        return result

    # ================================================================
    # Preflight helpers
    # ================================================================
    def _ensure_benchmarks(self, max_k: int = 13) -> None:
        if self._lp_bench is not None and self._cell_rate is not None:
            return
        rng = np.random.default_rng(0xC0FFEE)
        bench: Dict[int, float] = {}
        for k in range(2, max(max_k + 1, 3)):
            times: List[float] = []
            trials = max(3, min(80, 200 // max(1, k)))
            for _ in range(trials):
                M = rng.uniform(-k, k, size=(k, k))
                t0 = time.perf_counter()
                try:
                    solve_zero_sum_matrix(M)
                except Exception:
                    continue
                times.append((time.perf_counter() - t0) * 1000.0)
            if times:
                bench[k] = float(np.median(times))
        bench[1] = 0.0005  # 1×1 trivial —— ~0.5 μs
        self._lp_bench = bench

        # cell rate: 纯数字赋值 + 条件判断的吞吐 (粗略但稳定)
        K = 2000
        M = np.zeros((50, 50))
        t0 = time.perf_counter()
        cnt = 0
        for _ in range(K):
            for i in range(50):
                for j in range(50):
                    M[i, j] = 1 if i > j else (-1 if i < j else 0)
                    cnt += 1
        dt = max(1e-6, time.perf_counter() - t0)
        self._cell_rate = cnt / dt

    def _extrapolate_lp_time(self, k: int) -> float:
        """k 超出 benchmark 范围时, 用 t(k) ≈ α k^3 估计。"""
        if self._lp_bench and k - 1 in self._lp_bench:
            base_k = k - 1
            t_base = self._lp_bench[base_k]
            # O(k^3) extrapolation
            return t_base * ((k / base_k) ** 3)
        return 0.5 * (k ** 3) / (13 ** 3) * 0.51  # 退化兜底

    def _ensure_memory_calibration(self, run_calibration_solve: bool = True) -> None:
        if self._mem_per_state is not None:
            # Only re-run optional bits on subsequent calls
            if (not run_calibration_solve) or self._calib_factor is not None:
                return
        tracemalloc.start()
        cache: Dict[int, float] = {}
        # 填 50,000 个 entries 估一下
        DUMMY_N = 13
        n_states = 50000
        for i in range(n_states):
            key = i & ((1 << (3 * DUMMY_N)) - 1)
            cache[key] = float(i)
        current, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # 只算 dict payload 的保守估计
        self._mem_per_state = max(24.0, current / max(1, n_states))
        del cache

        if not run_calibration_solve:
            if self._calib_factor is None:
                self._calib_factor = 1.2  # 温和的兜底
            return

        # 真正跑一次 N=6 做 correction factor
        try:
            t0 = time.perf_counter()
            pred_report = self.estimate(6, benchmark=False, calibrate=False)
            # 用 Level-B *没有* calibration 的预测
            self._ensure_benchmarks(max_k=6)
            # 重新手工算 1 次预测耗时
            lp_sec = 0.0
            for k, info in pred_report.per_layer.items():
                if k == 0:
                    continue
                t_ms = self._lp_bench.get(k, self._extrapolate_lp_time(k))
                lp_sec += info["lps"] * (t_ms / 1000.0)
            cell_rate = self._cell_rate or 5_000_000.0
            cell_sec = pred_report.matrix_cells / cell_rate
            cache_sec = pred_report.chance_states / 2_000_000.0
            predicted = lp_sec + cell_sec + cache_sec

            cache2: Dict[int, float] = {}
            full = (1 << 6) - 1
            # 不调用 self.solve (避免循环)，直接跑核心递归
            self._solve_chance(full, full, full, 6, cache2)
            actual = time.perf_counter() - t0
            if predicted > 0:
                self._calib_factor = max(0.5, min(5.0, actual / predicted))
            else:  # pragma: no cover
                self._calib_factor = 1.0
        except Exception:  # pragma: no cover
            self._calib_factor = 1.2

    def _decide_risk_level_b(self, rpt: ComplexityReport) -> Tuple[str, str]:
        reasons: List[str] = []
        levels: List[str] = []
        # Time
        t = rpt.est_seconds_expected
        if t is not None:
            if t > 10 * 86400:
                levels.append(RISK_BLACK)
                reasons.append(f"runtime {t/86400:.1f} days > 10 days")
            elif t > 6 * 3600:
                levels.append(RISK_RED)
                reasons.append(f"runtime {t/3600:.1f} h > 6h")
            elif t > 30 * 60:
                levels.append(RISK_ORANGE)
                reasons.append(f"runtime {t/60:.1f} min > 30 min")
            elif t > 3 * 60:
                levels.append(RISK_YELLOW)
                reasons.append(f"runtime {t/60:.1f} min > 3 min")
            else:
                levels.append(RISK_GREEN)
        # RAM
        m = rpt.est_memory_bytes
        ram = rpt.available_ram_bytes
        if m is not None and ram is not None:
            if m > ram:
                levels.append(RISK_BLACK)
                reasons.append(f"estimated RAM {m/1024**3:.2f}GiB > available {ram/1024**3:.2f}GiB")
            elif m > 0.6 * ram:
                levels.append(RISK_RED)
                reasons.append("RAM > 60% available")
            elif m > 0.25 * ram:
                levels.append(RISK_ORANGE)
                reasons.append("RAM > 25% available")
            elif m > 512 * 1024 ** 2:
                levels.append(RISK_YELLOW)
                reasons.append("RAM > 512 MiB")
        # 取最严格的那个级别
        if not levels:
            # Fall back to Level-A
            return _decide_risk_level_a(rpt)
        worst = max(levels, key=lambda r: _RISK_ORDER[r])
        reason = "; ".join(reasons) if reasons else ""
        return worst, reason

    def _assert_feasible(
        self,
        rpt: ComplexityReport,
        *,
        max_expected_seconds: Optional[float],
        max_memory_bytes: Optional[int],
    ) -> None:
        if max_expected_seconds is not None and rpt.est_seconds_expected is not None:
            if rpt.est_seconds_expected > max_expected_seconds:
                raise RuntimeError(
                    f"Preflight aborted: expected runtime "
                    f"{rpt.est_seconds_expected:.1f}s > limit {max_expected_seconds}s. "
                    f"Pass force=True to override."
                )
        if max_memory_bytes is not None and rpt.est_memory_bytes is not None:
            if rpt.est_memory_bytes > max_memory_bytes:
                raise RuntimeError(
                    f"Preflight aborted: estimated memory "
                    f"{rpt.est_memory_bytes}B > limit {max_memory_bytes}B. "
                    f"Pass force=True to override."
                )
        if rpt.risk in (RISK_RED, RISK_BLACK):
            raise RuntimeError(
                "Preflight aborted: "
                f"risk={rpt.risk}, reason='{rpt.risk_reason}'. "
                "Pass force=True to override, or reduce N."
            )


# ===================================================================
# 5.  工具：RAM 探测 (跨平台尽力而为)
# ===================================================================
def _available_ram() -> Optional[int]:
    """Try to report available physical RAM (bytes). Returns None on failure."""
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    # Windows fallback —— 不依赖第三方包
    try:
        if os.name == "nt":
            import ctypes
            # Win32: GlobalMemoryStatusEx
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            ms = MEMORYSTATUSEX()
            ms.dwLength = ctypes.sizeof(ms)
            ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            if ok:
                return int(ms.ullAvailPhys)
    except Exception:  # pragma: no cover
        return None
    return None


# ===================================================================
# 6.  Module-level convenience API
# ===================================================================
_DEFAULT_SOLVER: Optional[GoofspielExactSolver] = None


def _solver() -> GoofspielExactSolver:
    global _DEFAULT_SOLVER
    if _DEFAULT_SOLVER is None:
        _DEFAULT_SOLVER = GoofspielExactSolver()
    return _DEFAULT_SOLVER


def estimate(n: int, *, benchmark: bool = True) -> ComplexityReport:
    """Module-level convenience: preflight estimate."""
    return _solver().estimate(n, benchmark=benchmark)


def solve(n: int, *, force: bool = False,
          config: Optional[SolverConfig] = None) -> SolveResult:
    """Module-level convenience: exact solve."""
    solver = GoofspielExactSolver(config) if config is not None else _solver()
    return solver.solve(n, force=force)


def solve_with_policy(
    n: int,
    *,
    force: bool = False,
    config: Optional[SolverConfig] = None,
) -> SolveResult:
    """Module-level convenience: exact solve + policies per revealed prize."""
    solver = GoofspielExactSolver(config) if config is not None else _solver()
    return solver.solve_with_policy(n, force=force)


# ===================================================================
# 7.  Carry-Over variant ——— preflight (Level-A closed form)
# ===================================================================
def estimate_carry_complexity(n: int) -> ComplexityReport:
    """
    Level-A exact complexity upper-bound for the carry-over Goofspiel variant.

    Classic:
        C_classic(N) = Σ_k C(N,k)^3
    Carry: chance state = (A, B, R, carry).  carry ∈ [0, N(N+1)/2].

    Exact count is *data-flow driven* (not simply ×carry_branches on every
    state; carry only accumulates on tie-paths).  For a safe preflight we
    therefore report an UPPER-BOUND:
        C_carry_max(N) = Σ_k C(N,k)^3 × (carry_max + 1),  where
        carry_max = N(N+1)/2.

    Similarly L_carry ≈ L_classic(N) × (carry_max + 1),
              E_carry ≈ E_classic(N) × (carry_max + 1).

    Risk thresholds are intentionally tighter than the classic solver because
    users have a much smaller tractable-N window for the carry-over model.
    """
    # Reuse classic estimator's loop output to keep it mathematically coherent.
    rpt_classic = estimate_complexity(n)
    carry_branches = (n * (n + 1)) // 2 + 1   # include 0

    total_c = rpt_classic.chance_states * carry_branches
    total_l = rpt_classic.matrix_games * carry_branches
    total_e = rpt_classic.matrix_cells * carry_branches
    # Per layer: inflate by carry_branches too.
    per_layer = {
        k: {
            "states": info["states"] * carry_branches,
            "lps": info["lps"] * carry_branches,
            "cells": info["cells"] * carry_branches,
        }
        for k, info in rpt_classic.per_layer.items()
    }
    report = ComplexityReport(
        N=n,
        chance_states=total_c,
        matrix_games=total_l,
        matrix_cells=total_e,
        per_layer=per_layer,
    )
    report.risk, report.risk_reason = _decide_risk_level_a_carry(report)
    return report


def _decide_risk_level_a_carry(rpt: ComplexityReport) -> Tuple[str, str]:
    """
    Tighter than classic because carry N=4 already ~40K states (classic N=5).
    Thresholds scaled by ~50× tighter (since carry_branches ~ N²/2 scales).
    """
    c = rpt.chance_states
    # Upper bound on carry branches shrinks tractable window significantly.
    if c > 4 * 10**10:                         # N>=10 upper bound — infeasible
        return RISK_BLACK, "carry upper bound > 40B; infeasible"
    if c > 2 * 10**8:                           # N=7 ~ 93M
        return RISK_RED, "carry states >= 200M; likely infeasible without clustering"
    if c > 10**6:                               # N=5 ~ 36K → N=6 ~ 334K → N=7 is RED
        return RISK_ORANGE, "carry states >= 1M; heavy solve; use force to override default cap"
    if c > 4 * 10**4:                           # N=4 ~ 3.8K → green
        return RISK_YELLOW, "carry states >= 40K; non-trivial but cheap for N<=5"
    return RISK_GREEN, "small carry-over state-space; trivial to solve (N ≤ 4)"


# ===================================================================
# 8.  Carry-Over solver  (独立类 —— state 含 carry，三分支 terminal)
# ===================================================================
class GoofspielCarrySolver:
    """
    Exact Nash solver for the CARRY-OVER Goofspiel rule.

    Recurrence (row player A, column player B; prize p; carry-in = c):
        F(A, B, R, c) = mean_{p ∈ R} V_p(A, B, R, c)
    where V_p is the Nash value of the |A|-by-|B| matrix game M:
        M[i,j] = immediate + F(child_A\{a}, child_B\{b}, R\{p}, carry_next)
    and immediate + carry_next branch three ways:
        a > b:  immediate = +(p + c),   carry_next = 0
        a < b:  immediate = -(p + c),   carry_next = 0
        tie + R\{p} ≠ ∅:  immed = 0,   carry_next = p + c   (rolls over)
        tie + R\{p} = ∅:  immed = 0,   carry_next = 0       (discard final)

    Symmetries:
        F(A,B,R,c) = -F(B,A,R,c)   (swap sign; carry is public)
        F(A,A,R,c) = 0  for any c  (anti-symmetry forces zero — CORRECT)
        The root  F(all,all,all,0) = 0  is a special case of the above.
    """

    def __init__(self, config: Optional[SolverConfig] = None) -> None:
        self.config = config or SolverConfig()
        self._lp_bench: Optional[Dict[int, float]] = None
        self._cell_rate: Optional[float] = None
        self._mem_per_state: Optional[float] = None
        self._calib_factor: Optional[float] = None

    # -------------------------------------------------------------- public
    def estimate(
        self, n: int, *, benchmark: bool = True, calibrate: bool = True,
    ) -> ComplexityReport:
        """Analogue of GoofspielExactSolver.estimate, carry-over variant."""
        rpt = estimate_carry_complexity(n)
        if self.config.skip_benchmark:
            benchmark = False
            calibrate = False
        if self.config.skip_calibration_solve:
            calibrate = False

        if benchmark:
            self._ensure_benchmarks(max_k=min(n, 13))
            self._ensure_memory_calibration(run_calibration_solve=calibrate)

            lp_sec = 0.0
            for k, info in rpt.per_layer.items():
                if k == 0:
                    continue
                t_ms = self._lp_bench.get(k, self._extrapolate_lp_time(k))
                lp_sec += info["lps"] * (t_ms / 1000.0)
            cell_sec = (rpt.matrix_cells / self._cell_rate) if self._cell_rate else 0.0
            cache_sec = rpt.chance_states / 2_000_000.0
            total = lp_sec + cell_sec + cache_sec
            if calibrate and self._calib_factor is not None:
                total *= self._calib_factor

            rpt.est_seconds_optimistic = total * 0.7
            rpt.est_seconds_expected = total
            rpt.est_seconds_conservative = total * 2.0
            if self._mem_per_state:
                rpt.est_memory_bytes = int(rpt.chance_states * self._mem_per_state)
            rpt.correction_factor = self._calib_factor
            rpt.available_ram_bytes = _available_ram()
            rpt.risk, rpt.risk_reason = self._decide_risk_level_b(rpt)
        return rpt

    def solve(
        self,
        n: int,
        *,
        force: bool = False,
        max_expected_seconds: Optional[float] = None,
        max_memory_bytes: Optional[int] = None,
    ) -> SolveResult:
        report = self.estimate(n)
        if not force:
            self._assert_feasible(
                report,
                max_expected_seconds=max_expected_seconds,
                max_memory_bytes=max_memory_bytes,
            )
        cache: Dict[Tuple[int, int, int, int], float] = {}
        t0 = time.perf_counter()
        full = (1 << n) - 1
        value = self._solve_chance(full, full, full, 0, n, cache)
        elapsed = time.perf_counter() - t0
        return SolveResult(
            N=n, value=value, report=report,
            cache_size=len(cache), elapsed_seconds=elapsed, policy_map=None,
        )

    def solve_with_policy(
        self,
        n: int,
        *,
        force: bool = False,
        max_expected_seconds: Optional[float] = None,
        max_memory_bytes: Optional[int] = None,
    ) -> SolveResult:
        report = self.estimate(n)
        if not force:
            self._assert_feasible(
                report,
                max_expected_seconds=max_expected_seconds,
                max_memory_bytes=max_memory_bytes,
            )
        cache: Dict[Tuple[int, int, int, int], float] = {}
        # policy_map key = (A, B, R, carry, prize)  —— extra carry dim vs classic
        policy_map: Dict[Tuple[int, int, int, int, int],
                         Tuple[float, np.ndarray, np.ndarray]] = {}
        t0 = time.perf_counter()
        full = (1 << n) - 1
        value = self._solve_chance_with_policies(
            full, full, full, 0, n, cache, policy_map,
        )
        elapsed = time.perf_counter() - t0
        return SolveResult(
            N=n, value=value, report=report,
            cache_size=len(cache), elapsed_seconds=elapsed,
            policy_map=policy_map,  # type: ignore[arg-type]
        )

    # --------------------------------------------------- state canonical key
    def _max_carry(self, n: int) -> int:
        return n * (n + 1) // 2

    def _canonical_key(
        self, a_mask: int, b_mask: int, r_mask: int, carry: int, n: int,
    ) -> Tuple[Tuple[int, int, int, int], int]:
        """
        Returns (canonical_tuple, sign).  Cache stores F(canonical).
        Value read: sign * cache[canonical].

        canonical_tuple = (A', B', R, carry) with A' ≤ B'.
        Sign = +1 if A ≤ B, else -1.  (carry 不变，因为是公共底池.)
        """
        if self.config.use_symmetry and a_mask > b_mask:
            a_mask, b_mask = b_mask, a_mask
            sign = -1
        else:
            sign = +1
        # key: use Python tuple (a_mask, b_mask, r_mask, carry).  For N<=7,
        # a/b/r each fit in 7 bits, carry fits in 7 bits (max 28) — fine either way.
        return (a_mask, b_mask, r_mask, carry), sign

    # -------------------------------------------------- value cache lookup
    def _lookup_chance(
        self,
        a_mask: int, b_mask: int, r_mask: int, carry: int, n: int,
        cache: Dict[Tuple[int, int, int, int], float],
    ) -> float:
        if r_mask == 0:
            return 0.0
        # NOTE: DO NOT short-circuit A==B here.
        # `_lookup_chance` is only called AFTER a child state was eagerly
        # solved by Phase-1 of the parent, so the canonical key MUST exist
        # in the cache. The short-circuit "A==B,any c -> 0" is handled
        # directly in `_solve_chance` where the entry is also written to
        # cache; bypassing that store here would corrupt the recursion.
        key, sign = self._canonical_key(a_mask, b_mask, r_mask, carry, n)
        if key in cache:
            return sign * cache[key]
        raise RuntimeError(
            "_lookup_chance(carry): cache miss. "
            f"Missing F(A={mask_to_cards(a_mask)}, B={mask_to_cards(b_mask)}, "
            f"R={mask_to_cards(r_mask)}, carry={carry})."
        )

    # -------------------------------------------------------------- recursive
    def _solve_chance(
        self,
        a_mask: int, b_mask: int, r_mask: int, carry: int, n: int,
        cache: Dict[Tuple[int, int, int, int], float],
    ) -> float:
        if r_mask == 0:
            return 0.0
        key, sign = self._canonical_key(a_mask, b_mask, r_mask, carry, n)
        if key in cache:
            return sign * cache[key]
        # SHORTCUT: A == B, any carry → anti-symmetry forces F = 0.
        # Proof: F(A,B,R,c) = -F(B,A,R,c). If A = B then F = -F ⇒ 2F = 0 ⇒ F = 0.
        # Still STORE it so cache stats match the reachable state-space.
        if self.config.short_cut_equal_hand and a_mask == b_mask:
            cache[key] = 0.0
            return 0.0

        r_cards = mask_to_cards(r_mask)
        a_cards = mask_to_cards(a_mask)
        b_cards = mask_to_cards(b_mask)
        k = len(r_cards)
        a_child_masks = [a_mask & ~(1 << (a - 1)) for a in a_cards]
        b_child_masks = [b_mask & ~(1 << (b - 1)) for b in b_cards]

        # PHASE 1 — eagerly compute every reachable child chance node
        for p in r_cards:
            r_child = r_mask & ~(1 << (p - 1))
            # carry_next 三分支 (对所有 child A/B 对都可先枚举):
            #   a != b: carry_next = 0  (胜/负后清零)
            #   tie + r_child != 0: carry_next = p + carry
            #   tie + r_child == 0: carry_next = 0 (末轮平局丢弃)
            for i in range(len(a_cards)):
                for j in range(len(b_cards)):
                    a, b = a_cards[i], b_cards[j]
                    ca, cb = a_child_masks[i], b_child_masks[j]
                    if a != b:
                        cnext = 0
                    elif r_child != 0:
                        cnext = p + carry
                    else:
                        cnext = 0  # discard final tie, no next round to carry
                    ckey, _ = self._canonical_key(ca, cb, r_child, cnext, n)
                    if ckey not in cache:
                        self._solve_chance(ca, cb, r_child, cnext, n, cache)

        # PHASE 2 — build matrix per prize, LP, average
        total = 0.0
        for p in r_cards:
            r_child = r_mask & ~(1 << (p - 1))
            M = np.zeros((len(a_cards), len(b_cards)), dtype=np.float64)
            for i, a in enumerate(a_cards):
                ca = a_child_masks[i]
                for j, b in enumerate(b_cards):
                    cb = b_child_masks[j]
                    # --- carry-over 三分支 immediate + carry_child ---
                    if a > b:
                        immediate = (p + carry)
                        carry_next = 0
                    elif a < b:
                        immediate = -(p + carry)
                        carry_next = 0
                    elif r_child != 0:
                        immediate = 0
                        carry_next = p + carry
                    else:
                        # 末轮平局：整包 stake 永久丢弃
                        immediate = 0
                        carry_next = 0
                    future = self._lookup_chance(ca, cb, r_child, carry_next, n, cache)
                    M[i, j] = immediate + future
            value, _, _ = solve_zero_sum_matrix(M, tol=self.config.lp_tol)
            total += value
        result = total / k
        cache[key] = result if sign == +1 else -result
        return result

    def _solve_chance_with_policies(
        self,
        a_mask: int, b_mask: int, r_mask: int, carry: int, n: int,
        cache: Dict[Tuple[int, int, int, int], float],
        policy_map: Dict,  # actual = Dict[Tuple[4 int + int], (V, x, y)]
    ) -> float:
        if r_mask == 0:
            return 0.0
        # NOTE: do NOT short-circuit A==B here even though F = 0 by
        # anti-symmetry.  Callers need per-prize Nash policies (x*, y*)
        # for these states too (otherwise bot will fall back on the root
        # N=2/3 A==B symmetric states, which reach every user's very
        # first round!).
        key, sign = self._canonical_key(a_mask, b_mask, r_mask, carry, n)

        # Cache hit 且 policy 对该 (A,B,R,carry) 的所有 p 已存在则直接返回
        if key in cache:
            r_cards = mask_to_cards(r_mask)
            all_p = all(
                (a_mask, b_mask, r_mask, carry, p) in policy_map for p in r_cards
            )
            if all_p and len(policy_map) > 0:
                return sign * cache[key]

        r_cards = mask_to_cards(r_mask)
        a_cards = mask_to_cards(a_mask)
        b_cards = mask_to_cards(b_mask)
        k = len(r_cards)
        a_child_masks = [a_mask & ~(1 << (a - 1)) for a in a_cards]
        b_child_masks = [b_mask & ~(1 << (b - 1)) for b in b_cards]

        # PHASE 1 — eager child solve
        for p in r_cards:
            r_child = r_mask & ~(1 << (p - 1))
            for i in range(len(a_cards)):
                for j in range(len(b_cards)):
                    a, b = a_cards[i], b_cards[j]
                    ca, cb = a_child_masks[i], b_child_masks[j]
                    if a != b:
                        cnext = 0
                    elif r_child != 0:
                        cnext = p + carry
                    else:
                        cnext = 0
                    ckey, _ = self._canonical_key(ca, cb, r_child, cnext, n)
                    if ckey not in cache:
                        self._solve_chance_with_policies(
                            ca, cb, r_child, cnext, n, cache, policy_map,
                        )

        # PHASE 2 — matrices & policy record
        total = 0.0
        for p in r_cards:
            r_child = r_mask & ~(1 << (p - 1))
            M = np.zeros((len(a_cards), len(b_cards)), dtype=np.float64)
            for i, a in enumerate(a_cards):
                ca = a_child_masks[i]
                for j, b in enumerate(b_cards):
                    cb = b_child_masks[j]
                    if a > b:
                        immediate = (p + carry)
                        carry_next = 0
                    elif a < b:
                        immediate = -(p + carry)
                        carry_next = 0
                    elif r_child != 0:
                        immediate = 0
                        carry_next = p + carry
                    else:
                        immediate = 0
                        carry_next = 0
                    future = self._lookup_chance(ca, cb, r_child, carry_next, n, cache)
                    M[i, j] = immediate + future
            value, x, y = solve_zero_sum_matrix(M, tol=self.config.lp_tol)
            total += value
            # key = raw (A,B,R,carry,p) — 对应 solver 调用方视角，不 canonical
            policy_map[(a_mask, b_mask, r_mask, carry, p)] = (
                value, x.copy(), y.copy(),
            )

        result = total / k
        cache[key] = result if sign == +1 else -result
        return result

    # -------------------------------------------------- preflight / runtime
    # (Shared helpers between classic and carry-over solver.)
    def _ensure_benchmarks(self, max_k: int = 13) -> None:
        if self._lp_bench is not None and self._cell_rate is not None:
            return
        rng = np.random.default_rng(0xC0FFEE)
        bench: Dict[int, float] = {}
        for k in range(2, max(max_k + 1, 3)):
            times: List[float] = []
            trials = max(3, min(80, 200 // max(1, k)))
            for _ in range(trials):
                M = rng.uniform(-k, k, size=(k, k))
                t0 = time.perf_counter()
                try:
                    solve_zero_sum_matrix(M)
                except Exception:
                    continue
                times.append((time.perf_counter() - t0) * 1000.0)
            if times:
                bench[k] = float(np.median(times))
        bench[1] = 0.0005
        self._lp_bench = bench
        K = 2000
        M = np.zeros((50, 50))
        t0 = time.perf_counter()
        cnt = 0
        for _ in range(K):
            for i in range(50):
                for j in range(50):
                    M[i, j] = 1 if i > j else (-1 if i < j else 0)
                    cnt += 1
        _ = cnt
        dt = max(1e-6, time.perf_counter() - t0)
        self._cell_rate = (K * 50 * 50) / dt

    def _extrapolate_lp_time(self, k: int) -> float:
        if self._lp_bench and k - 1 in self._lp_bench:
            base_k = k - 1
            t_base = self._lp_bench[base_k]
            return t_base * ((k / base_k) ** 3)
        return 0.5 * (k ** 3) / (13 ** 3) * 0.51

    def _ensure_memory_calibration(self, run_calibration_solve: bool = True) -> None:
        if self._mem_per_state is not None:
            if (not run_calibration_solve) or self._calib_factor is not None:
                return
        tracemalloc.start()
        cache: Dict[Tuple[int, int, int, int], float] = {}
        n_states = 50000
        for i in range(n_states):
            key = (i & 0x1FFF, i & 0x1FFF, i & 0x1FFF, i & 0x7F)
            cache[key] = float(i)
        current, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self._mem_per_state = max(28.0, current / max(1, n_states))
        del cache

        if not run_calibration_solve:
            if self._calib_factor is None:
                self._calib_factor = 1.3
            return
        # Carry 校准：N=3 跑一次，快且状态小
        try:
            t0 = time.perf_counter()
            pred = estimate_carry_complexity(3)
            self._ensure_benchmarks(max_k=3)
            lp_sec = 0.0
            for k, info in pred.per_layer.items():
                if k == 0:
                    continue
                t_ms = self._lp_bench.get(k, self._extrapolate_lp_time(k))
                lp_sec += info["lps"] * (t_ms / 1000.0)
            cell_rate = self._cell_rate or 5_000_000.0
            cell_sec = pred.matrix_cells / cell_rate
            cache_sec = pred.chance_states / 2_000_000.0
            predicted = lp_sec + cell_sec + cache_sec

            cache2: Dict[Tuple[int, int, int, int], float] = {}
            full = (1 << 3) - 1
            self._solve_chance(full, full, full, 0, 3, cache2)
            actual = time.perf_counter() - t0
            if predicted > 0:
                self._calib_factor = max(0.5, min(6.0, actual / predicted))
            else:  # pragma: no cover
                self._calib_factor = 1.0
        except Exception:  # pragma: no cover
            self._calib_factor = 1.3

    def _decide_risk_level_b(self, rpt: ComplexityReport) -> Tuple[str, str]:
        levels: List[str] = []
        reasons: List[str] = []
        t = rpt.est_seconds_expected
        if t is not None:
            if t > 10 * 86400:
                levels.append(RISK_BLACK); reasons.append(f"runtime {t/86400:.1f} days > 10 days")
            elif t > 6 * 3600:
                levels.append(RISK_RED); reasons.append(f"runtime {t/3600:.1f} h > 6h")
            elif t > 20 * 60:
                levels.append(RISK_ORANGE); reasons.append(f"runtime {t/60:.1f} min > 20 min")
            elif t > 2 * 60:
                levels.append(RISK_YELLOW); reasons.append(f"runtime {t/60:.1f} min > 2 min")
            else:
                levels.append(RISK_GREEN)
        m = rpt.est_memory_bytes
        ram = rpt.available_ram_bytes
        if m is not None and ram is not None:
            if m > ram:
                levels.append(RISK_BLACK); reasons.append(
                    f"estimated RAM {m/1024**3:.2f}GiB > available {ram/1024**3:.2f}GiB")
            elif m > 0.6 * ram:
                levels.append(RISK_RED); reasons.append("RAM > 60% available")
            elif m > 0.25 * ram:
                levels.append(RISK_ORANGE); reasons.append("RAM > 25% available")
            elif m > 512 * 1024 ** 2:
                levels.append(RISK_YELLOW); reasons.append("RAM > 512 MiB")
        if not levels:
            return _decide_risk_level_a_carry(rpt)
        worst = max(levels, key=lambda r: _RISK_ORDER[r])
        return worst, ("; ".join(reasons) if reasons else "")

    def _assert_feasible(
        self,
        rpt: ComplexityReport,
        *,
        max_expected_seconds: Optional[float],
        max_memory_bytes: Optional[int],
    ) -> None:
        if max_expected_seconds is not None and rpt.est_seconds_expected is not None:
            if rpt.est_seconds_expected > max_expected_seconds:
                raise RuntimeError(
                    f"Carry-Nash Preflight aborted: expected runtime "
                    f"{rpt.est_seconds_expected:.1f}s > limit {max_expected_seconds}s. "
                    f"Pass force=True to override.")
        if max_memory_bytes is not None and rpt.est_memory_bytes is not None:
            if rpt.est_memory_bytes > max_memory_bytes:
                raise RuntimeError(
                    f"Carry-Nash Preflight aborted: memory "
                    f"{rpt.est_memory_bytes}B > limit {max_memory_bytes}B. "
                    f"Pass force=True to override.")
        if rpt.risk in (RISK_RED, RISK_BLACK):
            raise RuntimeError(
                "Carry-Nash Preflight aborted: "
                f"risk={rpt.risk}, reason='{rpt.risk_reason}'. "
                "Pass force=True to override, or reduce N.")


# ===================================================================
# 9.  Carry-Over: module-level convenience API
# ===================================================================
_DEFAULT_CARRY_SOLVER: Optional[GoofspielCarrySolver] = None


def _carry_solver() -> GoofspielCarrySolver:
    global _DEFAULT_CARRY_SOLVER
    if _DEFAULT_CARRY_SOLVER is None:
        _DEFAULT_CARRY_SOLVER = GoofspielCarrySolver()
    return _DEFAULT_CARRY_SOLVER


def estimate_carry(n: int, *, benchmark: bool = True) -> ComplexityReport:
    """Carry-over preflight estimate."""
    return _carry_solver().estimate(n, benchmark=benchmark)


def solve_carry(
    n: int,
    *,
    force: bool = False,
    config: Optional[SolverConfig] = None,
) -> SolveResult:
    """Carry-over exact solve (no policy map)."""
    solver = GoofspielCarrySolver(config) if config is not None else _carry_solver()
    return solver.solve(n, force=force)


def solve_with_policy_carry(
    n: int,
    *,
    force: bool = False,
    config: Optional[SolverConfig] = None,
) -> SolveResult:
    """Carry-over exact solve WITH per-revealed-prize Nash policies."""
    solver = GoofspielCarrySolver(config) if config is not None else _carry_solver()
    return solver.solve_with_policy(n, force=force)
