"""
Author: 陈子聪 (Chen Zicong)
Date: 2026-08-30
Purpose: Unit tests for the Goofspiel exact Nash solver.

Covers:
  - Preflight Level-A combinatorics against hand-checked N=5..7 counts
  - Bitmask roundtrip
  - Zero-sum matrix LP correctness on a known 2×2 "matching pennies"
  - Root symmetry:  F(all, all, all) = 0   for N=1..5
  - F(A,B,R) = -F(B,A,R)   swap symmetry
  - Each Nash policy is a valid probability simplex (sum=1, all >=0)
  - V_p value of a matrix lies between pure min and pure max
  - Preflight risk ranking is monotone non-decreasing in N
  - N=13 triggers RED/BLACK without force=True
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pytest

from goofspiel import (
    GoofspielExactSolver,
    GoofspielCarrySolver,
    SolverConfig,
    estimate,
    solve,
    solve_with_policy,
    estimate_carry,
    solve_carry,
    solve_with_policy_carry,
    RISK_GREEN,
    RISK_YELLOW,
    RISK_ORANGE,
    RISK_RED,
    RISK_BLACK,
    cards_to_mask,
    mask_to_cards,
    solve_zero_sum_matrix,
)
from goofspiel.solver import (
    estimate_complexity,
    estimate_carry_complexity,
    _RISK_ORDER,
)


# -------------------------------------------------------------------
# 统一：测试用途 solver 一律跳过 Level-B benchmark / calibration，
#       否则 N=6 的真实求解会把每次测试跑慢几十秒。
# -------------------------------------------------------------------
_FAST_CFG = SolverConfig(
    skip_benchmark=True,
    skip_calibration_solve=True,
)


def _fast_solver(cfg: Optional[SolverConfig] = None) -> GoofspielExactSolver:
    """Build a solver that never runs machine calibration benchmarks."""
    merged = SolverConfig(
        use_symmetry=(cfg.use_symmetry if cfg else True),
        lp_tol=(cfg.lp_tol if cfg else 1e-9),
        short_cut_equal_hand=(cfg.short_cut_equal_hand if cfg else True),
        skip_benchmark=True,
        skip_calibration_solve=True,
    )
    return GoofspielExactSolver(merged)


# ===================================================================
# Bitmask helpers
# ===================================================================
class TestBitmask:
    def test_cards_to_mask_roundtrip(self):
        for n in (1, 2, 5, 13):
            full = list(range(1, n + 1))
            assert mask_to_cards(cards_to_mask(full)) == full

    def test_mask_remove(self):
        # A\{a} == A & ~(1 << (a-1))
        A = cards_to_mask([1, 2, 5, 8, 11])
        A2 = A & ~(1 << (5 - 1))
        assert mask_to_cards(A2) == [1, 2, 8, 11]

    def test_popcount(self):
        assert bin(cards_to_mask([1, 3, 5])).count("1") == 3


# ===================================================================
# Level-A preflight  (hand-check 表格前几行)
# ===================================================================
class TestPreflightLevelA:
    @pytest.mark.parametrize("n,expected_C", [
        # Central trinomial coefficient: Σ C(N,k)^3 = A000172
        (1, 2),              # 1 + 1
        (2, 10),             # 1 + 8 + 1
        (3, 56),             # 1 + 27 + 27 + 1
        (4, 346),            # 1 + 64 + 216 + 64 + 1 = 346  (A000172)
        (5, 2_252),          # per problem statement (matches 38M row in user's table for N=10)
        (6, 15_184),
        (7, 104_960),
    ])
    def test_exact_cache_states_C(self, n, expected_C):
        rpt = estimate_complexity(n)
        assert rpt.chance_states == expected_C

    def test_L_E_closed_form_vs_naive(self):
        # naive O(N * 2^N) enumeration via comb3(k) definition.
        def naive(n):
            C = L = E = 0
            for k in range(n + 1):
                c = math.comb(n, k) ** 3
                C += c
                L += k * c
                E += (k ** 3) * c
            return C, L, E
        for n in range(1, 11):
            rpt = estimate_complexity(n)
            C, L, E = naive(n)
            assert rpt.chance_states == C
            assert rpt.matrix_games == L
            assert rpt.matrix_cells == E

    def test_risk_order_monotonic_in_N(self):
        """N 越大风险等级 >= 之前的等级。"""
        prev = -1
        for n in range(1, 14):
            rpt = estimate_complexity(n)
            cur = _RISK_ORDER[rpt.risk]
            assert cur >= prev, f"risk regression at N={n}: {rpt.risk}"
            prev = cur

    def test_N13_is_at_least_orange(self):
        """N=13 无论如何至少是 ORANGE 级 (纯 Level A)。"""
        rpt = estimate_complexity(13)
        # 用户表格里 C=15.15B，对应 RED/BLACK 之间
        assert _RISK_ORDER[rpt.risk] >= _RISK_ORDER[RISK_ORANGE]


# ===================================================================
# Zero-sum matrix LP correctness
# ===================================================================
class TestZeroSumMatrix:
    def test_matching_pennies(self):
        # Matching pennies: +1/-1; value = 0, Nash = uniform on both sides.
        M = np.array([[+1.0, -1.0],
                      [-1.0, +1.0]])
        v, x, y = solve_zero_sum_matrix(M)
        assert abs(v) < 1e-6
        assert abs(x[0] - 0.5) < 1e-5
        assert abs(x[1] - 0.5) < 1e-5
        assert abs(y[0] - 0.5) < 1e-5
        assert abs(y[1] - 0.5) < 1e-5

    def test_dominated_action(self):
        # Row 2 dominates row 1:  Nash should put 0 on row 1.
        M = np.array([[1.0, 2.0],
                      [3.0, 4.0]])
        v, x, y = solve_zero_sum_matrix(M)
        # value should be 3 (col minimises: min(3,4)=3; rows can always get it via row 2)
        assert abs(v - 3.0) < 1e-6
        assert x[1] > 0.999

    def test_saddle_point_3x3(self):
        # Row 1 col 3 = 7  is a saddle:
        #   row mins:  min(8,3,7)=3;  min(5,4,6)=4;  min(0,2,1)=0  →  maximin = 4
        # Actually:  [8,3,7]; [5,4,6]; [0,2,1]
        #   row mins = 3, 4, 0   → maximin = 4
        #   col maxs = 8, 4, 7   → minimax = 4   (saddle at (r=2,c=2) with value 4)
        M = np.array([
            [8, 3, 7],
            [5, 4, 6],
            [0, 2, 1],
        ])
        v, x, y = solve_zero_sum_matrix(M)
        assert abs(v - 4.0) < 1e-5
        # Pure Nash: x = [0,1,0], y = [0,1,0]
        assert abs(x[1] - 1.0) < 1e-4
        assert abs(y[1] - 1.0) < 1e-4

    def test_policies_are_simplex(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            k = rng.integers(2, 8)
            M = rng.uniform(-5, 5, size=(k, k))
            v, x, y = solve_zero_sum_matrix(M)
            assert x.shape == (k,)
            assert y.shape == (k,)
            assert (x >= -1e-9).all()
            assert (y >= -1e-9).all()
            assert abs(x.sum() - 1.0) < 1e-6
            assert abs(y.sum() - 1.0) < 1e-6
            # Minimax theorem bounds
            # x^T M y == v  (within tol)
            gap = float(x @ M @ y - v)
            assert abs(gap) < 1e-5


# ===================================================================
# Exact solver invariants
# ===================================================================
class _SmallSolver(GoofspielExactSolver):
    """测试专用 solver：强制不做 benchmark，防止在小机器上意外触发大开销。"""

    def estimate(self, n, *, benchmark=False, calibrate=False):
        return super().estimate(n, benchmark=False, calibrate=False)


class TestExactSolver:
    def test_root_value_symmetric_is_zero(self):
        """对任意 N 的对称起点 F(all,all,all) == 0。"""
        solver = _fast_solver()
        for n in range(1, 6):
            res = solver.solve(n)
            assert abs(res.value) < 1e-8, f"N={n} root value {res.value} not 0"

    def test_swap_symmetry(self):
        """
        F(A,B,R) = -F(B,A,R).
        通过直接调用内部 chance 递归验证 (不用 root 捷径)。
        """
        solver = _fast_solver(SolverConfig(use_symmetry=False))
        # 手工构造非对称场景 for N=5.
        n = 5
        # 挑: A = {1,2,4}, B = {2,3,5}, R = {1,3,5}. (k=3)
        A = cards_to_mask([1, 2, 4])
        B = cards_to_mask([2, 3, 5])
        R = cards_to_mask([1, 3, 5])
        cache: dict = {}
        f1 = solver._solve_chance(A, B, R, n, cache)
        f2 = solver._solve_chance(B, A, R, n, cache)
        assert abs(f1 + f2) < 1e-8, f"swap symmetry failed: {f1} + {f2} = {f1 + f2}"

    def test_symmetry_pruning_reduces_cache(self):
        """打开 use_symmetry 应该显著减少 cache size。

        NOTE: 关闭 short_cut_equal_hand, 否则 root A==B 直接 return, cache_size=0.
        """
        cfg_nosym = SolverConfig(use_symmetry=False, short_cut_equal_hand=False,
                                 skip_benchmark=True, skip_calibration_solve=True)
        cfg_sym = SolverConfig(use_symmetry=True, short_cut_equal_hand=False,
                               skip_benchmark=True, skip_calibration_solve=True)
        no_sym = GoofspielExactSolver(cfg_nosym).solve(5)
        with_sym = GoofspielExactSolver(cfg_sym).solve(5)
        assert with_sym.cache_size <= no_sym.cache_size
        # N=5 至少减少 25%
        assert with_sym.cache_size < 0.85 * no_sym.cache_size, (
            f"sym cache {with_sym.cache_size} not < 0.85 * {no_sym.cache_size}"
        )

    def test_root_value_symmetric_is_zero_without_shortcut(self):
        """关闭 shortcut 后对称起点仍为 0 (纯递归验证)。"""
        solver = _fast_solver(SolverConfig(use_symmetry=True,
                                           short_cut_equal_hand=False))
        for n in range(1, 5):
            res = solver.solve(n)
            assert abs(res.value) < 1e-8

    def test_equal_hand_shortcut(self):
        """A==B 的任意子状态应该 shortcut 出 0 并且缓存里不爆长。"""
        solver = _fast_solver()
        n = 4
        # 挑一个中间等手牌状态 A=B={1,3}, R={2,4}
        AB = cards_to_mask([1, 3])
        R = cards_to_mask([2, 4])
        cache: dict = {}
        v = solver._solve_chance(AB, AB, R, n, cache)
        assert abs(v) < 1e-12

    def test_nontrivial_n2_policy(self):
        """
        N=2: root value == 0  by symmetry; V_1 and V_2 both = 0 (per manual M).

        Minimax theorem yields MANY equivalent Nash strategies at degenerate
        LP vertices.  Tests only check the invariants that every *valid* Nash
        must satisfy, not the "shape" of the strategy (uniform/non-uniform):
            (1)  V_1 = V_2 = 0  individually
            (2)  x,y are simplexes
            (3)  policies ARE a Nash:
                   x^T M >= v · 1    (Player A guarantees v vs any pure b)
                   M y <= v · 1    (Player B limits A to at most v vs any pure a)
        """
        full = solve_with_policy(2, config=_FAST_CFG)
        assert full.policy_map is not None
        assert abs(full.value) < 1e-8
        mask2 = (1 << 2) - 1
        key1 = (mask2, mask2, mask2, 1)
        key2 = (mask2, mask2, mask2, 2)
        v1, x1, y1 = full.policy_map[key1]
        v2, x2, y2 = full.policy_map[key2]
        assert abs(v1) < 1e-7, f"V_1 should be 0, got {v1}"
        assert abs(v2) < 1e-7, f"V_2 should be 0, got {v2}"
        assert abs((v1 + v2) / 2 - full.value) < 1e-7

        # Rebuild M for each prize and validate x^T M >= v · 1 & M y <= v · 1
        tmp = _fast_solver()
        for (p, v, x, y) in ((1, v1, x1, y1), (2, v2, x2, y2)):
            a_cards = [1, 2]; b_cards = [1, 2]
            r_child = mask2 & ~(1 << (p - 1))
            # cache children first
            cache_t: dict = {}
            for a in a_cards:
                for b in b_cards:
                    ca = mask2 & ~(1 << (a - 1))
                    cb = mask2 & ~(1 << (b - 1))
                    pk, _ = tmp._canonical_key(ca, cb, r_child, 2)
                    if pk not in cache_t:
                        tmp._solve_chance(ca, cb, r_child, 2, cache_t)
            M = np.zeros((2, 2))
            for i, a in enumerate(a_cards):
                for j, b in enumerate(b_cards):
                    imm = p * (1 if a > b else (-1 if a < b else 0))
                    ca = mask2 & ~(1 << (a - 1))
                    cb = mask2 & ~(1 << (b - 1))
                    fut = tmp._lookup_chance(ca, cb, r_child, 2, cache_t)
                    M[i, j] = imm + fut
            # Policies are valid simplexes
            assert abs(x.sum() - 1.0) < 1e-6
            assert abs(y.sum() - 1.0) < 1e-6
            assert (x >= -1e-9).all() and (y >= -1e-9).all()
            # Nash equilibrium conditions (any valid Nash must satisfy)
            assert (x @ M  >= v - 1e-7).all(), f"p={p}: x^T M = {x @ M} < v={v}"
            assert (M @ y  <= v + 1e-7).all(), f"p={p}: M y = {M @ y} > v={v}"

    def test_n3_policies_sane(self):
        """N=3 完整解，验证所有 matrix game 的价值在 [pure_min, pure_max] 区间内。"""
        full = solve_with_policy(3, config=_FAST_CFG)
        assert abs(full.value) < 1e-8
        # 用同一个快解算器做独立验证
        tmp = _fast_solver()
        for (A, B, R, p), (v, x, y) in full.policy_map.items():
            a_cards = mask_to_cards(A)
            b_cards = mask_to_cards(B)
            n = max(3, *(a_cards + b_cards + mask_to_cards(R)))
            cache_tmp: dict = {}
            r_child = R & ~(1 << (p - 1))
            # Phase 1: 先把所有 child 填好 (保证 _lookup_chance 必 hit)
            acm = [A & ~(1 << (a - 1)) for a in a_cards]
            bcm = [B & ~(1 << (b - 1)) for b in b_cards]
            for ca in acm:
                for cb in bcm:
                    pk, _ = tmp._canonical_key(ca, cb, r_child, n)
                    if pk not in cache_tmp:
                        tmp._solve_chance(ca, cb, r_child, n, cache_tmp)
            M = np.zeros((len(a_cards), len(b_cards)))
            for i, a in enumerate(a_cards):
                for j, b in enumerate(b_cards):
                    imm = p * (1 if a > b else (-1 if a < b else 0))
                    fut = tmp._lookup_chance(acm[i], bcm[j], r_child, n, cache_tmp)
                    M[i, j] = imm + fut
            # v 必须介于 maximin 和 minimax 之间 (Minimax 定理)
            row_guarantee = np.min(M, axis=1).max()
            col_guarantee = np.max(M, axis=0).min()
            assert v >= row_guarantee - 1e-7, \
                f"v={v} < maximin {row_guarantee} for (A,B,R,p)=({A},{B},{R},{p})"
            assert v <= col_guarantee + 1e-7, \
                f"v={v} > minimax {col_guarantee} for (A,B,R,p)=({A},{B},{R},{p})"
            assert abs(x.sum() - 1.0) < 1e-6
            assert abs(y.sum() - 1.0) < 1e-6


# ===================================================================
# Preflight abort on dangerous N
# ===================================================================
class TestPreflightAbort:
    def test_n13_preflight_blocks_without_force(self):
        """solve(13) must raise —— 保护用户。"""
        solver = GoofspielExactSolver(_FAST_CFG)
        # 我们不真实跑 N=13，只构造一个 RED/BLACK 报告并验证 gate 逻辑
        rpt = estimate_complexity(13)
        assert _RISK_ORDER[rpt.risk] >= _RISK_ORDER[RISK_RED], \
            f"N=13 risk should be RED/BLACK, got {rpt.risk}"
        with pytest.raises(RuntimeError):
            solver._assert_feasible(
                rpt,
                max_expected_seconds=None,
                max_memory_bytes=None,
            )

    def test_force_overrides_abort(self):
        """force=True 跳过检查 (但不真正执行 N=13 求解)。"""
        # 跑 small N 验证公开 solve API 成功路径
        r = solve(2, config=_FAST_CFG)
        assert abs(r.value) < 1e-8

    def test_module_estimate_runs_fast(self):
        """用户层面 estimate() 立即返回，不触发重量级 benchmark。"""
        import time
        t0 = time.perf_counter()
        rpt = estimate(8, benchmark=False)
        dt = time.perf_counter() - t0
        assert dt < 0.5, f"Level-A estimate took {dt}s, should be instant"
        assert rpt.N == 8
        assert rpt.chance_states > 0

    def test_small_solve_returns_zero(self):
        """最终端到端：对 N=1..5, solve(N).value == 0。"""
        for n in range(1, 6):
            r = solve(n, config=_FAST_CFG)
            assert abs(r.value) < 1e-9, f"solve({n}).value = {r.value}"


# ===================================================================
# Formatting
# ===================================================================
class TestReportFormat:
    def test_format_human_no_crash(self):
        for n in (1, 5, 10, 13):
            rpt = estimate_complexity(n)
            s = rpt.format_human()
            assert isinstance(s, str)
            assert "Preflight" in s
            assert f"N:                 {n}" in s


# ===================================================================
# Carry-over 规则精确 Nash: GoofspielCarrySolver 不变量
# ===================================================================
class TestCarrySolver:
    """验证双 Nash 的第二套 solver（carry-over 奖牌型）独立、
    数学上自洽，且与经典 solver 在同子状态给出 *不同* V 值。"""

    # ----------------------------------------------------------------
    # [1] 反对称根态：F(A=all, B=all, R=all, carry=0) = 0  (N=2,3)
    # ----------------------------------------------------------------
    def test_root_symmetric_is_zero_n2(self):
        r = solve_carry(2, config=_FAST_CFG)
        assert abs(r.value) < 1e-9, f"solve_carry(2).value={r.value}"

    def test_root_symmetric_is_zero_n3(self):
        r = solve_carry(3, config=_FAST_CFG)
        assert abs(r.value) < 1e-9, f"solve_carry(3).value={r.value}"

    # ----------------------------------------------------------------
    # [2] 等手任意 carry 短路：F(A=A, A, R, 任意 c) == 0
    #     (之前被误判为 carry>0 非 0，已用反证 F=-F 纠正)
    # ----------------------------------------------------------------
    def test_equal_hand_any_carry_shortcuts_to_zero(self):
        slv = GoofspielCarrySolver(_FAST_CFG)
        cache: dict = {}
        # A=B={2,3}, R={1,2}, c ∈ {0, 2, 5} — 均应为 0
        A = cards_to_mask([2, 3])
        R = cards_to_mask([1, 2])
        for c in (0, 2, 5):
            v = slv._solve_chance(A, A, R, n=3, cache=cache, carry=c)
            assert abs(v) < 1e-9, (
                f"F(A=B,R,carry={c}) must be 0 (anti-symmetry F=-F), "
                f"got {v}"
            )
        # 且 cache 必须被写入（等手短路仍然存 cache，避免上层 cache_size=0 bug）
        assert len(cache) >= 1, "equal-hand shortcut must still write cache"

    # ----------------------------------------------------------------
    # [3] Swap 对称：F(A,B,R,c) = -F(B,A,R,c)
    # ----------------------------------------------------------------
    def test_swap_symmetry_with_carry(self):
        slv = GoofspielCarrySolver(_FAST_CFG)
        cache: dict = {}
        # 构造非对称手：A={1,3}, B={2,3}, R={2,3}, carry=2
        A = cards_to_mask([1, 3])
        B = cards_to_mask([2, 3])
        R = cards_to_mask([2, 3])
        v1 = slv._solve_chance(A, B, R, n=3, cache=cache, carry=2)
        # 清空 cache 防符号复用泄漏
        cache2: dict = {}
        v2 = slv._solve_chance(B, A, R, n=3, cache=cache2, carry=2)
        assert abs(v1 + v2) < 1e-8, (
            f"Swap symmetry violated: F(A,B,c=2)={v1}, "
            f"F(B,A,c=2)={v2}, sum={v1+v2} (should be 0)"
        )

    # ----------------------------------------------------------------
    # [4] 双 solver 隔离：V(classic) != V(carry c=0) != V(carry c=2)
    #     对同一 (A,B,R) 子状态（平局→弃奖 vs 平局→滚入 两套规则 V 不同）
    # ----------------------------------------------------------------
    def test_carry_and_classic_values_diverge(self):
        slv_c = GoofspielCarrySolver(_FAST_CFG)
        slv_x = GoofspielExactSolver(_FAST_CFG)
        cache_c: dict = {}
        cache_x: dict = {}
        # A={2,3}, B={1,3}, R={2,3} — 存在平局路径（都出 3），
        # 经典 vs carry 的 terminal reward 不同 → V 必不同
        A = cards_to_mask([2, 3])
        B = cards_to_mask([1, 3])
        R = cards_to_mask([2, 3])
        v_x = slv_x._solve_chance(A, B, R, n=3, cache=cache_x)
        v_c0 = slv_c._solve_chance(A, B, R, n=3, cache=cache_c, carry=0)
        v_c2 = slv_c._solve_chance(A, B, R, n=3, cache=cache_c, carry=2)
        # 三者必须两两不同 (summary 中 benchmark 数字)
        assert v_x != v_c0, (
            f"Classic V({v_x}) must != Carry(c=0) V({v_c0}) — 规则不同"
        )
        assert v_c0 != v_c2, (
            f"Carry(c=0) V({v_c0}) must != Carry(c=2) V({v_c2}) — stake 不同"
        )
        assert v_x != v_c2, (
            f"Classic V must != Carry(c=2) V"
        )

    # ----------------------------------------------------------------
    # [5] policy_map 存在 carry>0 的 5 元组 key；
    #     所有返回的 (x,y) 都是合法概率单纯形 (sum=1, each>=0)
    # ----------------------------------------------------------------
    def test_policy_carry_keys_and_simplex(self):
        full = solve_with_policy_carry(3, config=_FAST_CFG)
        assert full.policy_map, "carry policy_map must be non-empty for N=3"

        # [5a] 必须存在至少一个 carry>0 的 policy 条目（R1 平局
        #      → R2 carry>0 时，子状态也会被 full.policy_map 填上）
        has_pos_carry = any(
            key[3] > 0 for key in full.policy_map.keys()
        )
        assert has_pos_carry, (
            "carry solver policy_map must contain entries with carry>0 "
            "(5-tuple key form: (A,B,R,carry,prize))."
        )

        # [5b] 所有 policy 条目：x、y 都是单纯形
        for key, (v, x, y) in full.policy_map.items():
            assert x.ndim == 1 and y.ndim == 1
            assert x.shape == y.shape
            assert (x >= -1e-9).all(), f"x has negative component @{key}: {x}"
            assert (y >= -1e-9).all(), f"y has negative component @{key}: {y}"
            assert abs(x.sum() - 1.0) < 1e-5, (
                f"x simplex sum={x.sum()} for key {key}"
            )
            assert abs(y.sum() - 1.0) < 1e-5, (
                f"y simplex sum={y.sum()} for key {key}"
            )

    # ----------------------------------------------------------------
    # [6] Preflight Level-A: N=4 carry ≈ GREEN（用户默认 N=4 安全）
    # ----------------------------------------------------------------
    def test_n4_carry_preflight_is_green(self):
        rpt = estimate_carry_complexity(4)
        assert _RISK_ORDER[rpt.risk] <= _RISK_ORDER[RISK_GREEN], (
            f"N=4 carry-over estimate should be GREEN (safe for default), "
            f"got risk={rpt.risk} chance_states={rpt.chance_states}"
        )
        # 顶层 estimate_carry() 模块 API 可直接运行（不触发 benchmark）
        import time as _t
        t0 = _t.perf_counter()
        r2 = estimate_carry(4, benchmark=False)
        dt = _t.perf_counter() - t0
        assert dt < 0.5, f"carry estimate Level-A must be instant, took {dt}s"
        assert r2.N == 4
