"""
Author: 陈子聪 (Chen Zicong)
Date: 2026-08-30
Purpose: Goofspiel bot library: Random / Heuristic / Exact Nash.

All bots implement a single method:

    choose_action_with_policy(env, player) -> Tuple[Action, Dict[str, Any]]

where the policy-info dict is rendered by the web frontend as a distribution
bar (让用户能看到 AI 是怎么想的)。

For backwards compatibility the classic:

    choose_action(env, player) -> Action

is still supported on every bot (random bots.py).
"""

from __future__ import annotations

import secrets
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .env import (
    GoofspielEnv, PlayerId, Action, Rng, PLAYER_0, PLAYER_1,
    INFO_LOW, INFO_HIGH,
)
from .solver import (
    SolverConfig,
    GoofspielExactSolver,
    GoofspielCarrySolver,   # NEW — carry-over rule exact solver
    SolveResult,
    cards_to_mask,
)

# Bot-type identifiers (公开字符串,用于前端下拉 & /api/game/new)
BOT_RANDOM = "random"
BOT_HEURISTIC = "heuristic"
BOT_NASH = "nash"                     # Classic "tie → discard" textbook Nash
BOT_NASH_CARRY = "nash_carry"         # NEW — "tie → rollover prize pool" Nash
EXACT_MODE_CLASSIC = "classic"
EXACT_MODE_CARRY = "carry"

BOT_TYPES: Tuple[str, ...] = (
    BOT_RANDOM, BOT_HEURISTIC, BOT_NASH, BOT_NASH_CARRY,
)

BOT_DESCRIPTIONS: Dict[str, str] = {
    BOT_RANDOM: "Random · 纯随机 (baseline)",
    BOT_HEURISTIC: "Heuristic · 启发式 (出价≈奖金比例 + carry 适配)",
    BOT_NASH: "Nash · 精确纳什 · 经典平局弃奖牌型 (仅 N ≤ 7, carry>0 会诚实回落)",
    BOT_NASH_CARRY: "Nash · 精确纳什 · Carry-Over 平局滚入奖牌型 (仅 N ≤ 4 默认)",
}

# Nash bot 只有 N 不太大才有真值;大 N 自动 fallback 到 Heuristic 并报告原因。
NASH_MAX_N = 7            # 经典平局弃奖牌型
NASH_CARRY_MAX_N = 4      # Carry-Over 平局滚入模型 (状态空间 blow-up 更大)

# ==========================================================================
# VEIL-mechanism helpers (for bot notes & honest-fallback decisions)
# ==========================================================================
def _veil_any_enabled(env: GoofspielEnv) -> bool:
    """True 当启用了任一 VEIL 机制 (hidden_prize/suit_tiebreak/info_reward).

    这些机制下, CLASSIC 与 CARRY 的精确 Nash solver 推导假设不成立,
    NashBot 必须诚实回落 Heuristic (详见 NashBot._choose 诚实回落 #0)。
    """
    return bool(
        getattr(env, "hidden_prize", False)
        or getattr(env, "suit_tiebreak", False)
        or getattr(env, "info_reward_enabled", False)
    )


def _veil_active_note(env: GoofspielEnv) -> str:
    """在 bot 的 note 末尾追加 "VEIL[HiddenPrize,SuitTiebreak,…]" 标签,
    让前端 GUI 可见当前 AI 运行在哪种规则变种上。"""
    tags: List[str] = []
    if getattr(env, "hidden_prize", False):
        tags.append("HiddenPrize")
    if getattr(env, "suit_tiebreak", False):
        tags.append("SuitTiebreak")
    if getattr(env, "info_reward_enabled", False):
        mode = getattr(env, "info_bits_mode", "half")
        tags.append(f"InfoReward({mode})")
    if not tags:
        return ""
    return "VEIL[" + ",".join(tags) + "]"


# ==========================================================================
# Base bot
# ==========================================================================
class BaseBot:
    """All bots inherit this.  Subclasses MUST override _choose."""

    def __init__(self, rng: Optional[Rng] = None) -> None:
        self._rng: Rng = rng if rng is not None else secrets.SystemRandom()

    # ------------------------------------------------------------------ API
    def choose_action(self, env: GoofspielEnv, player: PlayerId) -> Action:
        """Classic single-action API (legacy env.py compat)."""
        action, _ = self.choose_action_with_policy(env, player)
        return action

    def choose_action_with_policy(
        self,
        env: GoofspielEnv,
        player: PlayerId,
    ) -> Tuple[Action, Dict[str, Any]]:
        """
        Returns:
            action: the card the bot will play this round.
            info:   {"distribution": [[card_value, pct_0_to_100], …],
                     "value":      float (AI expected score diff; NaN for Random),
                     "bot_type":   str,
                     "note":       str  (额外前端可读说明)}
        """
        return self._choose(env, player)

    # -------------------------------------------------------------- to impl
    def _choose(
        self, env: GoofspielEnv, player: PlayerId,
    ) -> Tuple[Action, Dict[str, Any]]:  # pragma: no cover - abstract
        raise NotImplementedError


# ==========================================================================
# 1) RandomBot —— still matches legacy goofspiel.bots.RandomBot signature.
# ==========================================================================
class RandomBot(BaseBot):
    """Uniform random legal action chooser."""

    def __init__(self, rng: Optional[Rng] = None) -> None:
        super().__init__(rng=rng)

    def _choose(
        self, env: GoofspielEnv, player: PlayerId,
    ) -> Tuple[Action, Dict[str, Any]]:
        legal = sorted(env.legal_actions(player))
        if not legal:
            raise RuntimeError(f"RandomBot: no legal actions for {player}")
        n = len(legal)
        pct = 100.0 / n
        dist = [[c, pct] for c in legal]
        action = self._rng.choice(legal)
        veil_note = _veil_active_note(env)
        return int(action), {
            "distribution": dist,
            "value": float("nan"),
            "bot_type": BOT_RANDOM,
            "note": (f"Uniform random · 1/{n}" + (f" · {veil_note}" if veil_note else "")),
        }


# ==========================================================================
# 2) HeuristicBot —— deterministic-ish rule-based, strong vs casual humans.
# ==========================================================================
class HeuristicBot(BaseBot):
    """
    A lightweight rule-based agent.

    Heuristic (human-interpretable rules):
      1) Let `p = current prize`.
      2) Compute soft target rank: rank = percentile(p, sorted_prizes_still_left).
         We want a card roughly at the same quantile in our hand.
      3) Apply "small prize damping":  for p ≤ (mean prize)/2, cap bid to low 50%.
         This reserves high cards for the real big prizes.
      4) Add small randomness (choose among target±1 neighbours with weighted
         coin) to avoid becoming fully predictable.
      5) If we're already losing badly by the tail, "go for broke" on the
         biggest remaining prize.
    """

    def __init__(self, rng: Optional[Rng] = None) -> None:
        super().__init__(rng=rng)
        # Per-instance seeded numpy Generator so the neighbour-sampling below is
        # reproducible from ``self._rng`` — using the global ``np.random`` here
        # would make every match non-deterministic even under a fixed seed
        # (same bug NashBot already avoids via its own ``_np_rng``).
        self._np_rng = np.random.default_rng(self._rng.randint(0, 2**31 - 1))

    def _choose(
        self, env: GoofspielEnv, player: PlayerId,
    ) -> Tuple[Action, Dict[str, Any]]:
        legal = sorted(env.legal_actions(player))
        if not legal:
            raise RuntimeError(f"HeuristicBot: no legal actions for {player}")

        # ================= VEIL: 私人情报先验 (VEIL §12) =================
        # 如果 bot 自己持有 HIGH/LOW 情报, 则将"下一轮奖励"(即当前这一轮)的
        # belief 收缩到 R_HIGH / R_LOW 半区. hidden_prize=False 时该信息仅
        # 作为"cross-check", 不冲突 (obs 当前奖已知, 情报自然一致).
        my_private_signal: Optional[str] = None
        if hasattr(env, "private_info"):
            my_private_signal = env.private_info.get(player)
        # ==================================================================

        # 1) Current prize context
        #    经典可见: current_prize 是确定值
        #    Hidden Prize (VEIL §9): 可能 current_prize == None → 用期望值 +
        #      private_signal (如有) 做收缩 belief
        carry = int(env.carry_pool) if hasattr(env, "carry_pool") else 0
        raw_round_prize_raw = env.current_prize
        prizes_left_pool = sorted(
            list(env.remaining_prizes)
            + ([int(raw_round_prize_raw)] if raw_round_prize_raw is not None else [])
        )
        k = len(prizes_left_pool) or 1
        max_real_prize = max(prizes_left_pool) if prizes_left_pool else 0
        mean_real_prize = float(sum(prizes_left_pool)) / k

        if raw_round_prize_raw is None and getattr(env, "hidden_prize", False):
            # ---------- Hidden Prize + 私人情报 belief (VEIL §10-12) ----------
            # Belief: 默认是剩余奖励集上的均匀分布; 如果持有 HIGH/LOW 信号,
            # 则将 belief 限制在对应半区内。
            R_sorted = sorted(prizes_left_pool)
            if R_sorted:
                import math as _m
                size_low = int(_m.ceil(len(R_sorted) / 2))
                R_LOW_set = set(R_sorted[:size_low])
                R_HIGH_set = set(R_sorted[size_low:])
            else:
                R_LOW_set = set()
                R_HIGH_set = set()
            if my_private_signal == INFO_LOW and R_LOW_set:
                belief_support = sorted(R_LOW_set)
            elif my_private_signal == INFO_HIGH and R_HIGH_set:
                belief_support = sorted(R_HIGH_set)
            else:
                belief_support = list(R_sorted)  # 无情报 → 全支持
            if not belief_support:
                belief_support = [1]
            # 有效 round_prize = belief 的中位数
            mid = len(belief_support) // 2
            round_prize = int(belief_support[mid])
            # idx_p: 在剩余奖券中的百分位 (用中位数计算, 保守估计)
            idx_p = min(k - 1, R_sorted.index(round_prize)) if round_prize in R_sorted else 0
            prize_certainty = (
                "hidden(LOW-bias)" if my_private_signal == INFO_LOW else
                "hidden(HIGH-bias)" if my_private_signal == INFO_HIGH else
                "hidden(uniform-belief)"
            )
            p_median = round_prize + carry
            p = p_median
        else:
            # ---------- 经典: current_prize 已知 ----------
            round_prize = int(raw_round_prize_raw or 0)
            idx_p = prizes_left_pool.index(round_prize) if round_prize in prizes_left_pool else 0
            prize_certainty = "visible"
            p = round_prize + carry

        # 2) Score context —— tail "go for broke" rule
        opponent = PLAYER_1 if player == PLAYER_0 else PLAYER_0
        my_score = env.scores[player]
        op_score = env.scores[opponent]
        tail_losing = (
            k <= max(2, env.num_cards // 4)
            and my_score < op_score
            and (
                (raw_round_prize_raw == max_real_prize) if raw_round_prize_raw is not None
                else (round_prize >= max_real_prize)  # hidden 模式下 belief 中位数最大
            )
            or (carry > 0 and p >= max_real_prize)
        )

        # 3) Pick target rank in [0, len(legal)-1]
        if tail_losing:
            target_rank = len(legal) - 1  # 梭哈最大牌
            note = "Tail go-for-broke · 梭哈大牌抢最大奖"
        else:
            # raw rank: round_prize 的百分位映射（反映它在剩余奖券中的稀有度）
            raw_rank = (idx_p / max(1, (k - 1))) if k > 1 else 0.0
            # --- apply carry-aware adjustments on top of the percentile base ---
            if round_prize > 0:
                inflation = max(1.0, float(p) / float(round_prize))
                boost_from_carry = min(0.22, (inflation - 1.0) * 0.14)
                raw_rank = min(1.0, raw_rank + boost_from_carry)
            mean_eff_ref = (mean_real_prize + carry)
            if p < max(mean_real_prize, mean_eff_ref) * 0.75:
                raw_rank *= 0.55
            if p > max(mean_real_prize, mean_eff_ref) * 1.25:
                raw_rank = min(1.0, raw_rank * 1.12 + 0.04)
            target_rank = int(round(raw_rank * (len(legal) - 1)))
            target_rank = max(0, min(len(legal) - 1, target_rank))
            note = (
                f"Target rank {target_rank+1}/{len(legal)} "
                f"(prize percentile {idx_p}/{k-1 if k>1 else 0}, {prize_certainty})"
            )

        if carry > 0:
            note = f"[carry={carry}→eff_stake={p}] " + note
        if my_private_signal:
            note = f"[MY_PRIV_INFO={my_private_signal}] " + note
        veil_tag = _veil_active_note(env)
        if veil_tag:
            note = note + " · " + veil_tag

        # 4) Build distribution over target±1 neighbours
        weights = np.zeros(len(legal), dtype=float)
        weights[target_rank] = 0.70
        if target_rank - 1 >= 0:            weights[target_rank - 1] += 0.22
        if target_rank + 1 < len(weights):  weights[target_rank + 1] += 0.08
        if weights.sum() == 0:
            weights[:] = 1.0
        weights = weights / weights.sum()

        # Sample
        chosen_idx = int(self._np_rng.choice(len(legal), p=weights))
        action = legal[chosen_idx]

        dist = [[c, float(weights[i] * 100.0)] for i, c in enumerate(legal)]
        value = float("nan")  # 启发式没有 game-theoretic value
        return int(action), {
            "distribution": dist,
            "value": value,
            "bot_type": BOT_HEURISTIC,
            "note": note,
        }


# ==========================================================================
# 3) NashBot —— Exact ground-truth solver-backed policy.
#    离线预计算 solve_with_policy(N); 在线查表 + 按 Nash x* 抽样。
# ==========================================================================
class NashBot(BaseBot):
    """
    Exact-Nash Goofspiel bot.  Supports TWO rule-models side-by-side:

      exact_mode = EXACT_MODE_CLASSIC  (bot_type_id 'nash')
          → solver: GoofspielExactSolver, "tie → discard prize" textbook model.
            If env.carry_pool > 0 → HONESTLY fall back to HeuristicBot (this
            solver has no carry dimension; using it would be wrong).

      exact_mode = EXACT_MODE_CARRY  (bot_type_id 'nash_carry')
          → solver: GoofspielCarrySolver, "tie → rollover prize into carry_pool
            (discard only on final-round tie)" model (= 当前网页实际规则).
            Works for ANY carry value (0 and non-zero).

    Constraints (强制约束 —— 与 preflight 保持一致):
      - CLASSIC: N <= NASH_MAX_N (default 7) 才真正跑 exact solver;
      - CARRY:   N <= NASH_CARRY_MAX_N (default 4) 才真正跑 exact solver;
        (carry 状态空间 blow-up 更大，用更低的默认上限保护用户);
      - 超出自动 fallback 到 HeuristicBot (在 note 里报告原因)。
      - Swap 对称  F(A,B,R,c) = -F(B,A,R,c) 对两种模式都成立.
    """

    # Class-level caches: key = int N -> SolveResult
    # (Separated so we can NEVER accidentally serve the wrong rule-model.)
    _policy_cache_classic: Dict[int, SolveResult] = {}
    _policy_cache_carry:   Dict[int, SolveResult] = {}
    _fallback_cache: Dict[int, HeuristicBot] = {}

    def __init__(
        self,
        *,
        rng: Optional[Rng] = None,
        max_nash_n: Optional[int] = None,
        exact_mode: str = EXACT_MODE_CLASSIC,
        config: Optional[SolverConfig] = None,
    ) -> None:
        super().__init__(rng=rng)
        if exact_mode == EXACT_MODE_CLASSIC:
            self._exact_mode = EXACT_MODE_CLASSIC
            self._max_nash_n = (max_nash_n if max_nash_n is not None
                                else NASH_MAX_N)
            self._solver_cfg = config or SolverConfig(
                use_symmetry=True, short_cut_equal_hand=True,
                skip_benchmark=True, skip_calibration_solve=True,
            )
            self._solver: Union[GoofspielExactSolver, GoofspielCarrySolver] = (
                GoofspielExactSolver(self._solver_cfg)
            )
        elif exact_mode == EXACT_MODE_CARRY:
            self._exact_mode = EXACT_MODE_CARRY
            self._max_nash_n = (max_nash_n if max_nash_n is not None
                                else NASH_CARRY_MAX_N)
            self._solver_cfg = config or SolverConfig(
                use_symmetry=True, short_cut_equal_hand=True,
                skip_benchmark=True, skip_calibration_solve=True,
            )
            self._solver = GoofspielCarrySolver(self._solver_cfg)
        else:
            raise ValueError(
                f"NashBot: unknown exact_mode={exact_mode!r}. "
                f"Use one of {EXACT_MODE_CLASSIC!r}, {EXACT_MODE_CARRY!r}.")
        self._np_rng = np.random.default_rng(self._rng.randint(0, 2**31 - 1))

    # ---------------------------------------------------------------- core
    def _choose(
        self, env: GoofspielEnv, player: PlayerId,
    ) -> Tuple[Action, Dict[str, Any]]:
        num_cards = env.num_cards
        carry = int(env.carry_pool) if hasattr(env, "carry_pool") else 0

        # ---------------- 诚实回落 #0 (VEIL 机制 → solver 假设不成立) ----------------
        # CLASSIC / CARRY 的精确 Nash solver 都建立在:
        #   (a) 每轮奖励公开; (b) 平局由 rank-only 规则/经典-carry 处理;
        #   (c) 不存在"私人情报"影响 belief。
        # 一旦任一 VEIL 机制开启, solver 推导就不再是 ground truth.
        # 必须诚实回落 Heuristic, 绝不能输出"伪精确"分布。
        if _veil_any_enabled(env):
            flags_on: List[str] = []
            if getattr(env, "hidden_prize", False):          flags_on.append("hidden_prize")
            if getattr(env, "suit_tiebreak", False):         flags_on.append("suit_tiebreak")
            if getattr(env, "info_reward_enabled", False):   flags_on.append("info_reward")
            return self._fallback(
                num_cards, env, player,
                reason=(
                    f"VEIL 机制启用 ({','.join(flags_on)}) — "
                    f"Exact Nash solver 推导假设 (公开奖/无suit/无私信) 不成立, "
                    f"诚实回落 Heuristic。"),
            )

        # ---------------- 诚实回落 #1 (模式不兼容规则) ----------------
        # CLASSIC 模式 —— 如果 carry>0: solver 是 "平局弃奖" 训练的，
        #   绝对不能输出伪装精确分布。
        if self._exact_mode == EXACT_MODE_CLASSIC and carry > 0:
            return self._fallback(
                num_cards, env, player,
                reason=(
                    f"carry_pool={carry} 存在平局累计奖池；"
                    f"当前 Nash 精确 solver 为经典「平局弃奖」奖牌型推导，"
                    f"未适配 carry-over 规则。请切换到 bot_type='nash_carry'。"),
            )

        # ---------------- 诚实回落 #2 (N 超上限) ----------------
        if num_cards > self._max_nash_n:
            limit_label = (
                f"NASH_MAX_N={NASH_MAX_N}" if self._exact_mode == EXACT_MODE_CLASSIC
                else f"NASH_CARRY_MAX_N={NASH_CARRY_MAX_N}")
            return self._fallback(
                num_cards, env, player,
                reason=(f"N={num_cards} > {limit_label}"),
            )

        # ---------------- 精确查表 ----------------
        result = self._ensure_policy(num_cards)
        assert result is not None and result.policy_map is not None
        pm = result.policy_map

        opponent = PLAYER_1 if player == PLAYER_0 else PLAYER_0
        bot_cards: List[int] = sorted(env.legal_actions(player))
        opp_cards: List[int] = sorted(env.remaining_cards[opponent])
        current_prize: int     = int(env.current_prize or 0)
        prize_deck = sorted(list(env.remaining_prizes) + [current_prize])

        A_mask = cards_to_mask(bot_cards)
        B_mask = cards_to_mask(opp_cards)
        R_mask = cards_to_mask(prize_deck)

        # policy map key:
        #   CLASSIC: (A, B, R, prize)
        #   CARRY:   (A, B, R, carry, prize)   —— 5-tuple
        if self._exact_mode == EXACT_MODE_CLASSIC:
            key = (A_mask, B_mask, R_mask, current_prize)
        else:
            key = (A_mask, B_mask, R_mask, carry, current_prize)

        if key not in pm:
            return self._fallback(
                num_cards, env, player,
                reason=(
                    f"Nash({self._exact_mode}) policy miss for "
                    f"key={self._key_debug(key, current_prize, carry)}"),
            )

        value, x, _y = pm[key]
        if len(x) != len(bot_cards):
            return self._fallback(
                num_cards, env, player, reason="policy dim mismatch")

        xsafe = np.clip(x, 0.0, None)
        if xsafe.sum() <= 0:
            xsafe = np.ones_like(xsafe) / len(xsafe)
        else:
            xsafe = xsafe / xsafe.sum()

        chosen_idx = int(self._np_rng.choice(len(bot_cards), p=xsafe))
        action = bot_cards[chosen_idx]
        dist = [[c, float(xsafe[i] * 100.0)] for i, c in enumerate(bot_cards)]

        mode_tag = (
            "Nash-classic(tie-discard)"
            if self._exact_mode == EXACT_MODE_CLASSIC
            else f"Nash-carry-over(tie→rollover; carry_in={carry})"
        )
        note = (
            f"{mode_tag} · bot=row, opponent=col · "
            f"bot expected score-diff vs human from this state = {value:+.3f}"
        )
        bot_type_id = BOT_NASH if self._exact_mode == EXACT_MODE_CLASSIC \
            else BOT_NASH_CARRY
        return int(action), {
            "distribution": dist,
            "value": float(value),
            "bot_type": bot_type_id,
            "note": note,
        }

    # -------------------------------------------------------------- helpers
    def _key_debug(self, key, prize: int, carry: int) -> str:
        if self._exact_mode == EXACT_MODE_CLASSIC:
            A, B, R, _p = key  # type: ignore[misc]
            return f"(A={A:#x}, B={B:#x}, R={R:#x}, prize={prize})"
        A, B, R, c, _p = key  # type: ignore[misc]
        return f"(A={A:#x}, B={B:#x}, R={R:#x}, carry={c}, prize={prize})"

    def _ensure_policy(self, num_cards: int) -> SolveResult:
        if self._exact_mode == EXACT_MODE_CLASSIC:
            cache = NashBot._policy_cache_classic
            solver_for_cache = self._solver
            # solver already is GoofspielExactSolver (constructed correctly).
            if num_cards in cache:
                return cache[num_cards]
            # type ignore: both expose solve_with_policy with compatible signature
            result = solver_for_cache.solve_with_policy(num_cards, force=False)  # type: ignore[call-arg,arg-type]
            cache[num_cards] = result
            return result
        else:
            cache = NashBot._policy_cache_carry
            if num_cards in cache:
                return cache[num_cards]
            result = self._solver.solve_with_policy(num_cards, force=False)  # type: ignore[call-arg,arg-type]
            cache[num_cards] = result
            return result

    def _fallback(
        self,
        num_cards: int,
        env: GoofspielEnv,
        player: PlayerId,
        *,
        reason: str,
    ) -> Tuple[Action, Dict[str, Any]]:
        if num_cards not in NashBot._fallback_cache:
            NashBot._fallback_cache[num_cards] = HeuristicBot()
        action, info = NashBot._fallback_cache[num_cards].choose_action_with_policy(
            env, player,
        )
        # 覆盖说明,但保留 heuristic distribution
        fallback_model_label = (
            "Nash-classic"
            if self._exact_mode == EXACT_MODE_CLASSIC
            else "Nash-carry-over"
        )
        fallback_bot_type = (
            BOT_NASH if self._exact_mode == EXACT_MODE_CLASSIC
            else BOT_NASH_CARRY
        )
        info["note"] = (
            f"[{fallback_model_label} fallback to Heuristic · {reason}] "
            + (info.get("note") or "")
        )
        if "carry_pool=" in reason:
            info["value"] = 0.0
        # 用户选的哪款 Nash 就返回哪款 bot_type (带已解释的 fallback)
        info["bot_type"] = fallback_bot_type
        return action, info


# ==========================================================================
# Factory (app.py 后端使用)
# ==========================================================================
def create_bot(bot_type: str, *, seed: Optional[int] = None) -> BaseBot:
    """
    统一工厂。传入 seed=None → 用 secrets.SystemRandom()。
    """
    if bot_type not in BOT_TYPES:
        raise ValueError(
            f"Unknown bot_type '{bot_type}'. Must be one of {list(BOT_TYPES)}"
        )
    rng: Rng = random.Random(seed) if seed is not None else secrets.SystemRandom()
    if bot_type == BOT_RANDOM:
        return RandomBot(rng=rng)
    if bot_type == BOT_HEURISTIC:
        return HeuristicBot(rng=rng)
    if bot_type == BOT_NASH:
        return NashBot(rng=rng, exact_mode=EXACT_MODE_CLASSIC)
    if bot_type == BOT_NASH_CARRY:
        return NashBot(rng=rng, exact_mode=EXACT_MODE_CARRY)
    raise AssertionError("unreachable")  # pragma: no cover


# Keep legacy symbol exported — backwards compat with old tests / callers.
__all__ = [
    "RandomBot",
    "HeuristicBot",
    "NashBot",
    "BaseBot",
    "BOT_RANDOM",
    "BOT_HEURISTIC",
    "BOT_NASH",
    "BOT_NASH_CARRY",
    "EXACT_MODE_CLASSIC",
    "EXACT_MODE_CARRY",
    "BOT_TYPES",
    "BOT_DESCRIPTIONS",
    "NASH_MAX_N",
    "NASH_CARRY_MAX_N",
    "create_bot",
]
