"""
Author: 陈子聪 (Chen Zicong)
Date: 2026-08-30  (Last modified: 2026-08-31 — VEIL optional mechanisms added)
Purpose: Goofspiel (Game of Pure Strategy) core game environment.
         Provides a gym-like, RL-ready interface.

Rules (internal summary — 平局 carry-over 变体 + 可选 VEIL 机制):
  - Two players each hold cards 1..N (default N=13, where 11=J, 12=Q, 13=K).
  - A prize deck of 1..N is shuffled face-down.
  - Each round (CLASSIC rule, all VEIL flags = False):
      1. Reveal the next prize card.
      2. Compute prize_at_stake = current_prize + carry_pool (历史累计未发放)
      3. Both players SIMULTANEOUSLY pick one of their remaining cards.
      4. Higher card wins prize_at_stake → added to winner's score, carry_pool = 0.
      5. Tie and NOT the last round → both score 0, carry_pool becomes prize_at_stake
         (平局奖金不丢，累计入下一轮奖励池).
      6. Tie and IS the last round → both score 0, prize_at_stake is PERMANENTLY
         discarded (无下一轮可滚，这是唯一会真的丢弃奖品的场景).
      7. Played cards are permanently removed.
  - After N rounds, higher score wins; equal is a draw.

VEIL 可选机制 (全部默认关闭 — 开启即启用部分 VEIL 旗舰规则, §6-15 设计文档):
  - hidden_prize:
        出牌前 current_prize 在公开 observation 中隐藏 (出牌后联合揭晓)。
  - suit_tiebreak:
        每轮随机给 2 名玩家分配 2 个不重复 suit (♣<♦<♥<♠)。
        rank 相同 → suit 高者胜，保证唯一 Highest Bid，消除平局。
        开启时 carry_out 恒为 0 (无平局可滚)。
  - info_reward_enabled + info_bits_mode == "half":
        每轮 Lowest Bid 玩家获得下一轮奖励 HIGH/LOW 1-bit 私人信号。
        最后一轮不出情报 (§14 规则)。
"""

from __future__ import annotations

import secrets
import random
import math
from typing import Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Type aliases (类型别名)
# ---------------------------------------------------------------------------
PlayerId = str   # "player_0" / "player_1"
Action = int     # A card value 1..N
Rng = Union[random.Random, "secrets.SystemRandom"]  # Supports both seeded and crypto RNG

PLAYER_0: PlayerId = "player_0"
PLAYER_1: PlayerId = "player_1"
PLAYERS: Tuple[PlayerId, PlayerId] = (PLAYER_0, PLAYER_1)

# Suit 数值序:  ♣(0) < ♦(1) < ♥(2) < ♠(3)   (VEIL §6)
SUIT_RANKS: Dict[str, int] = {"C": 0, "D": 1, "H": 2, "S": 3}
SUIT_SYMBOLS: Dict[int, str] = {0: "\u2663", 1: "\u2666", 2: "\u2665", 3: "\u2660"}
SUIT_LABELS: Dict[int, str] = {0: "C", 1: "D", 2: "H", 3: "S"}

# Information signal 枚举 (VEIL §12 HIGH / LOW)
INFO_LOW = "LOW"
INFO_HIGH = "HIGH"
INFO_MODES = ("none", "half")  # 当前实现支持 none 和 half (VEIL 旗舰版); 其余(quartile/parity/exact/noisy_half)留待扩展

# Tie-rule 枚举 (Goofspiel 规则说明书 §29, 三类平局处理变体)
#   "rollover" = Variant A: 平局 → 整包 prize_at_stake 滚入下一轮 carry_pool
#                (末轮平局 = 丢弃, 即项目默认规则 = CARRY-OVER)
#   "discard"  = Variant C: 平局 = 奖金直接丢弃, 双方 0 分. 等同于 OpenSpiel /
#                经典教科书中的 Goofspiel 奖牌型 (Nash 经典 Solver 模型).
#                不产生 carry, carry_pool 恒为 0.
#   "split"    = Variant B: 平局 = 双方各拿 prize/2. 若 prize 为奇数, 向下取整
#                (即双方各拿 prize // 2), 剩余 1 分静默丢弃 (保证所有分数整数,
#                避免训练 reward 出 float 破坏现有训练管线契约).
#                不产生 carry, carry_pool 恒为 0.
TIE_RULE_ROLLOVER = "rollover"
TIE_RULE_DISCARD = "discard"
TIE_RULE_SPLIT = "split"
TIE_RULES = (TIE_RULE_ROLLOVER, TIE_RULE_DISCARD, TIE_RULE_SPLIT)
DEFAULT_TIE_RULE = TIE_RULE_ROLLOVER


class GoofspielEnv:
    """
    Goofspiel game environment (平局 carry-over 变体 + 可选 VEIL 机制作为 opt-in 开关)。

    Compatibility contract (兼容性铁律):
      GoofspielEnv()  默认构造 — 行为与旧版完全一致 (经典 carry-over tie 规则)。
      所有 VEIL 机制通过显式 keyword 参数开启; 默认值 = False/None。
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        num_cards: int = 13,
        rng: Optional[Rng] = None,
        # ---- 游戏规则变体 (§29): 平局处理方式 — 三选一 ----
        #      (属于核心规则参数, 与 VEIL 机制正交, 默认 rollover = 项目默认)
        tie_rule: str = DEFAULT_TIE_RULE,
        # ---- 以下全部为 VEIL 可选机制, 默认关闭 — 开启后行为向 VEIL 设计文档 §6-15 对齐 ----
        hidden_prize: bool = False,
        suit_tiebreak: bool = False,
        info_reward_enabled: bool = False,
        info_bits_mode: str = "none",   # "none" | "half"  (VEIL §15)
    ) -> None:
        """
        Args:
            num_cards: Number of cards per player (also size of prize deck), default 13.
            rng:       Random source. Default = secrets.SystemRandom() (crypto grade).
                       Tests may pass random.Random(seed) for reproducibility.
            tie_rule:
                平局规则 (规则说明书 §29 三选一，枚举 TIE_RULES)。
                rollover = Variant A (平局→carry 滚入下一轮，唯末轮平局丢弃) — 默认。
                discard  = Variant C (平局直接丢弃奖池，无 carry) — 经典教科书模型。
                split    = Variant B (平局双方各分 prize//2 分，向下取整，无 carry)。
            hidden_prize:
                出牌前隐藏当前奖励 (VEIL §9 Hidden Prize 模式)。
                出牌后联合揭晓前 observation.current_prize 为 None; step() 返回后揭晓。
            suit_tiebreak:
                每轮给 2 玩家随机分配 2 个不重复 suit; rank 平局时由 suit(♣<♦<♥<♠)
                决胜负, 消除平局 (VEIL §6)。开启时 carry_pool 恒为 0。
            info_reward_enabled:
                最低出牌者获得关于下一轮奖励的私人情报 (VEIL §11 Information Reward)。
            info_bits_mode:
                情报粒度 (VEIL §15)。当前实现支持 "none"(关闭) 和 "half"(§12 HIGH/LOW 二分)。
                未来扩展位: quartile / parity / exact / noisy_half。
        """
        if num_cards < 1:
            raise ValueError(f"num_cards must be >= 1, got {num_cards}")
        if tie_rule not in TIE_RULES:
            raise ValueError(
                f"tie_rule must be one of {TIE_RULES!r}, got {tie_rule!r}"
            )
        if info_bits_mode not in INFO_MODES:
            raise ValueError(
                f"info_bits_mode must be one of {INFO_MODES!r}, got {info_bits_mode!r}"
            )
        # 若 info_reward_enabled=True 但 info_bits_mode="none", 静默修正为 "half"
        if info_reward_enabled and info_bits_mode == "none":
            info_bits_mode = "half"
        # 若 info_bits_mode != "none" 视为 implicitly 开启 info_reward
        if info_bits_mode != "none":
            info_reward_enabled = True

        self.num_cards: int = num_cards
        self._rng: Rng = rng if rng is not None else secrets.SystemRandom()

        # ---- 规则变体 & VEIL 机制开关 (公共只读, app.py/bots.py 可查询) ----
        self.tie_rule: str = tie_rule
        self.hidden_prize: bool = bool(hidden_prize)
        self.suit_tiebreak: bool = bool(suit_tiebreak)
        self.info_reward_enabled: bool = bool(info_reward_enabled)
        self.info_bits_mode: str = info_bits_mode

        # Public state populated in reset() / step()
        self.round: int = 0
        self.done: bool = False
        self.scores: Dict[PlayerId, int] = {PLAYER_0: 0, PLAYER_1: 0}
        self.remaining_cards: Dict[PlayerId, List[Action]] = {PLAYER_0: [], PLAYER_1: []}
        self.prize_deck: List[int] = []
        self.current_prize: Optional[int] = None
        self.remaining_prizes: List[int] = []
        self.carry_pool: int = 0                          # 平局累计未发放奖金池

        # ---- VEIL suit-tiebreak 临时状态 ----
        # 本轮开始时 (reveal / seat-shuffle phase) 分配好的 suit 映射
        self.current_suits: Dict[PlayerId, int] = {PLAYER_0: 0, PLAYER_1: 0}
        # 历史中记录每轮 suit 分配 (便于 AI / UI / 审计)
        self._suit_history: List[Dict[PlayerId, int]] = []

        # ---- VEIL info-reward 状态 ----
        # 每个玩家"当前持有的"私人情报。获取当轮 step 返回后写入, 在下一轮 step
        # 之前的 observation 中对该玩家可见, step 之后清空 (消费掉或过期)。
        # value = None 表示无情报; "LOW"/"HIGH" 等表示该玩家拥有的信号。
        self.private_info: Dict[PlayerId, Optional[str]] = {
            PLAYER_0: None, PLAYER_1: None,
        }
        # 记录每轮 Lowest Bid 玩家 (供 UI 显示"谁获得了情报")
        # 元素: { "round": int, "lowest_player": PlayerId|None, "signal": str|None, "for_round": int }
        self.info_reward_log: List[Dict] = []

        # Each history entry (向后兼容, 字段扩展只增不改):
        #   round, prize, round_prize, carry_in, prize_at_stake,
        #   carry_out, discarded, actions, winner, rewards
        # + VEIL 扩展字段 (仅在对应机制开启时有值, 否则为 None):
        #   suits, lowest_bid_player, info_signal, info_for_round,
        #   prize_was_hidden_before_bid
        self.history: List[Dict] = []

    # ----------------------------------------------------------------- reset
    def reset(self) -> Dict:
        """
        Reset the environment for a new game.
        Returns the first observation (classic mode → with prize #1 revealed;
        hidden_prize mode → observation.current_prize = None until first step).
        """
        self.round = 0
        self.done = False
        self.scores = {PLAYER_0: 0, PLAYER_1: 0}
        self.history = []
        self.carry_pool = 0
        self.private_info = {PLAYER_0: None, PLAYER_1: None}
        self.info_reward_log = []
        self._suit_history = []

        # Both players start with cards 1..num_cards, sorted ascending.
        full_deck = list(range(1, self.num_cards + 1))
        self.remaining_cards = {
            PLAYER_0: list(full_deck),
            PLAYER_1: list(full_deck),
        }

        # Prize deck is a shuffled copy of 1..num_cards.
        self.prize_deck = list(full_deck)
        self._rng.shuffle(self.prize_deck)
        self.remaining_prizes = list(self.prize_deck)

        # 预取"第 1 轮"的 prize (内部已确定, 但 hidden_prize 模式下不在 obs 中公开)
        # NOTE: _reveal_next_prize 内部会 self.round += 1 并设置 self.current_prize。
        self._reveal_next_prize()
        # Round-1 seat shuffle (suit_tiebreak 模式)
        self._reshuffle_suits_if_enabled()
        return self.get_observation()

    # ------------------------------------------------------------------ step
    def step(self, actions: Dict[PlayerId, Action]) -> Tuple[Dict, Dict[PlayerId, int], bool, Dict]:
        """
        Advance one round using SIMULTANEOUS actions from both players.

        Signature 100% 与旧版一致, 保证训练代码 / bots.py / app.py 零修改即可运行。
        新增 VEIL 字段仅在 observation 与 info 中以额外 key 呈现。
        """
        if self.done:
            raise RuntimeError("Cannot call step() on a finished game. Call reset() first.")

        if 0 in actions or 1 in actions:  # type: ignore[operator]
            actions = {
                PLAYER_0: actions.get(PLAYER_0, actions.get(0)),  # type: ignore[arg-type]
                PLAYER_1: actions.get(PLAYER_1, actions.get(1)),  # type: ignore[arg-type]
            }

        # --- 1. Validate both actions present and legal -------------------
        if PLAYER_0 not in actions or PLAYER_1 not in actions:
            raise ValueError(
                f"step() requires actions for both {PLAYER_0} and {PLAYER_1}. Got: {actions}"
            )
        for pid in PLAYERS:
            self._assert_legal_action(pid, actions[pid])

        a0: Action = actions[PLAYER_0]
        a1: Action = actions[PLAYER_1]
        round_prize: int = self.current_prize          # 内部已确定 (hidden_prize 模式也已存在)
        carry_in: int = self.carry_pool
        prize_at_stake: int = round_prize + carry_in
        is_last_round: bool = (len(self.remaining_prizes) == 0)
        prize_was_hidden = bool(self.hidden_prize)
        suits_this_round: Dict[PlayerId, int] = dict(self.current_suits)

        # --- 2. Resolve winner / rewards / carry --------------------------
        #     VEIL §6 suit-tiebreak: 同一 rank → suit(♣<♦<♥<♠) 高者胜 (保证唯一winner)
        winner: Optional[PlayerId] = None
        rewards: Dict[PlayerId, int] = {PLAYER_0: 0, PLAYER_1: 0}
        carry_out: int = 0
        discarded: bool = False
        tie_occurred_rank_only: bool = (a0 == a1)

        # 计算"带 suit 的有效大小"
        def _effective(player_id: PlayerId, rank: int) -> Tuple[int, int]:
            if self.suit_tiebreak:
                return (rank, suits_this_round[player_id])
            return (rank, 0)  # 经典模式: suit 不参与 (因为 rank 已分胜负或会 tie)

        eff0 = _effective(PLAYER_0, a0)
        eff1 = _effective(PLAYER_1, a1)

        if eff0 > eff1:
            winner = PLAYER_0
            rewards[PLAYER_0] = prize_at_stake
            self.scores[PLAYER_0] += prize_at_stake
            carry_out = 0
        elif eff1 > eff0:
            winner = PLAYER_1
            rewards[PLAYER_1] = prize_at_stake
            self.scores[PLAYER_1] += prize_at_stake
            carry_out = 0
        else:  # tie (仅当 suit_tiebreak=False 且 a0==a1 时可达)
            winner = None
            rewards = {PLAYER_0: 0, PLAYER_1: 0}
            # ---- tie_rule 三分支 (§29 A/B/C) ----
            #   NOTE: 此处绝不直接依赖 NashBot / Solver 内部奖牌型，保持 env 纯规则层。
            #         rule / solver 的兼容由调用方 (app.py new_game 路由) 在开局时
            #         诚实回落至 heuristic，见 fallback_reason。
            if self.tie_rule == TIE_RULE_DISCARD:
                # Variant C: 平局一律丢弃
                carry_out = 0
                discarded = True
            elif self.tie_rule == TIE_RULE_SPLIT:
                # Variant B: 平局双方各分 prize//2 (向下取整, 余数静默丢弃)
                each = prize_at_stake // 2
                rewards = {PLAYER_0: each, PLAYER_1: each}
                self.scores[PLAYER_0] += each
                self.scores[PLAYER_1] += each
                carry_out = 0
                discarded = (prize_at_stake % 2) != 0  # 有"余数"就算一次丢弃 (便于审计)
            else:  # TIE_RULE_ROLLOVER (default)
                # Variant A: 非末轮平局 → carry_pool; 末轮平局 → 丢弃
                if is_last_round:
                    carry_out = 0
                    discarded = True
                else:
                    carry_out = prize_at_stake

        self.carry_pool = carry_out

        # --- 3. Resolve Lowest Bid + Information Reward (VEIL §11-15) ----
        #   - 信息奖励归属: 在 "effective 大小" 中更小的那个玩家
        #     (如果 suit_tiebreak=False 且 rank tie → 两人并列最低, 无人获得情报
        #      这是对 VEIL 2 人模式的合理退化, 因为没有唯一 Lowest Bid)
        #   - 最后一轮不出情报 (VEIL §14)
        lowest_player: Optional[PlayerId] = None
        info_signal: Optional[str] = None
        info_for_round: Optional[int] = None
        if self.info_reward_enabled and not is_last_round:
            if eff0 < eff1:
                lowest_player = PLAYER_0
            elif eff1 < eff0:
                lowest_player = PLAYER_1
            else:
                lowest_player = None  # 精确 tie → 无人获情报 (保守退化)
            if lowest_player is not None:
                info_signal = self._compute_next_prize_half_signal()
                info_for_round = self.round + 1  # 该情报"针对"的目标轮次
                self.private_info[lowest_player] = info_signal
                self.info_reward_log.append({
                    "round": self.round,
                    "lowest_player": lowest_player,
                    "signal": info_signal,
                    "for_round": info_for_round,
                })
        # (消费掉的情报在下一轮 step *之前* 仍有效; 之后我们在 _reveal 后清空过期)
        # -> 实际上 signal 的价值仅在"出价下一轮之前", 所以应在消费后 (下一次 step 开始时)
        #    清空. 这里写入 private_info, 在 _reveal_next_prize 消费后 (或目标轮揭晓后)
        #    清空. 简单策略: 每次 step 返回前, 清空 *已针对本轮揭晓过的* 那一条.
        #    因为 lowest_player 获得的是"下一轮情报", 也就是 *下一个 step* 才用.
        #    所以这次写进去的不会立刻过期.

        # --- 4. Remove played cards ---------------------------------------
        self.remaining_cards[PLAYER_0].remove(a0)
        self.remaining_cards[PLAYER_1].remove(a1)

        # --- 5. Record history (向后兼容: 原有字段全部保留 + VEIL 扩展字段)
        self._suit_history.append(dict(suits_this_round))
        self.history.append({
            # 经典字段
            "round": self.round,
            "prize": round_prize,
            "round_prize": round_prize,
            "carry_in": carry_in,
            "prize_at_stake": prize_at_stake,
            "carry_out": carry_out,
            "discarded": discarded,
            "actions": {PLAYER_0: a0, PLAYER_1: a1},
            "winner": winner,
            "rewards": dict(rewards),
            # VEIL 扩展字段 (未启用机制时为 None, 不影响旧代码判断)
            "suits": dict(suits_this_round) if self.suit_tiebreak else None,
            "tie_broken_by_suit": (
                True if self.suit_tiebreak and tie_occurred_rank_only else False
            ),
            "prize_was_hidden_before_bid": prize_was_hidden,
            "lowest_bid_player": lowest_player,
            "info_signal": info_signal,
            "info_for_round": info_for_round,
        })

        # --- 6. Advance / check done --------------------------------------
        # 清空"已过目标轮次"的私人情报 (情报消费掉了)
        self._expire_private_info_after_reveal()

        if is_last_round:
            # This was the last prize.
            self.done = True
            self.current_prize = None
        else:
            self._reveal_next_prize()
            # 下一轮重新分配 suit (VEIL §18 每轮重新匿名座位)
            self._reshuffle_suits_if_enabled()

        obs = self.get_observation()
        info = {
            # 经典字段
            "winner": winner,
            "round_prize": round_prize,
            "carry_in": carry_in,
            "prize_at_stake": prize_at_stake,
            "carry_out": carry_out,
            "discarded": discarded,
            # VEIL 扩展字段 (新增, 不破坏旧 key)
            "suits": dict(suits_this_round) if self.suit_tiebreak else None,
            "lowest_bid_player": lowest_player,
            "info_signal": info_signal,
            "info_for_round": info_for_round,
            "prize_was_hidden_before_bid": prize_was_hidden,
        }
        return obs, rewards, self.done, info

    # ------------------------------------------------------------- helpers
    def legal_actions(self, player: PlayerId) -> List[Action]:
        """Return the list of cards still available to `player`."""
        if player not in PLAYERS:
            raise ValueError(f"Unknown player '{player}'. Must be one of {PLAYERS}")
        return list(self.remaining_cards[player])

    def result(self) -> Optional[PlayerId]:
        """
        Return final game result (only valid after `done` is True):
          - PLAYER_0 / PLAYER_1 for a win,
          - "draw" for a tie,
          - None if game is not finished.
        """
        if not self.done:
            return None
        s0, s1 = self.scores[PLAYER_0], self.scores[PLAYER_1]
        if s0 > s1:
            return PLAYER_0
        if s1 > s0:
            return PLAYER_1
        return "draw"

    def get_observation(self, viewer: Optional[PlayerId] = None) -> Dict:
        """
        Build the observation dict.

        Signature change is BACKWARDS COMPATIBLE:
          - `viewer` is NEW optional kwarg.  Old callers who don't pass it
            get the OMNISCIENT public view (for backwards compat with existing
            training code that treats the env as fully observable 2-player).
          - When `viewer` is supplied, private_info["self"] is the info that
            THAT player is allowed to see; the OPPONENT's info signal is
            REDACTED to None (VEIL §12 private signal 仅得主可见).
          - When hidden_prize=True and the round hasn't been resolved yet
            (i.e. game not done AND still pre-bid phase): current_prize in obs
            = None to the players.  Server-omniscient view still shows it.
        """
        viewer_is_omniscient = (viewer is None)

        # 经典 observation (旧字段名 100% 保留)
        round = self.round
        scores = dict(self.scores)
        remaining_cards = {
            PLAYER_0: sorted(self.remaining_cards[PLAYER_0]),
            PLAYER_1: sorted(self.remaining_cards[PLAYER_1]),
        }
        remaining_prizes = sorted(self.remaining_prizes)
        carry_pool = int(self.carry_pool)
        done = self.done

        # current_prize 可见性规则:
        #   - hidden_prize=False → 永远可见 (经典)
        #   - hidden_prize=True 且 viewer 非 omniscient → 在 obs 中隐藏 (直到 bid 揭晓)
        #     揭晓时机: 本轮 step() 之后 current_prize 已经揭晓 → 此时应该可以让玩家看见
        #     简单判定: 在 step 返回后的 obs 里, 本轮已经过了 "prize reveal 阶段".
        #     由于 reset() 时 _reveal_next_prize 已执行但 step 未执行 → pre-bid.
        #     而 step() 结束时会 _reveal_next_prize 设好 *下一轮* 的 prize.
        #     因此: pre-bid 状态 = (self.current_prize is not None AND not done
        #                            AND self.history 中最大 round 索引 == self.round - 1)
        pre_bid_phase = (
            self.current_prize is not None
            and not done
            and (len(self.history) < self.round)
        )
        if self.hidden_prize and not viewer_is_omniscient and pre_bid_phase:
            obs_current_prize: Optional[int] = None
        else:
            obs_current_prize = self.current_prize

        # total_prize_at_stake 在 hidden_prize + pre-bid 时也应隐藏 (否则泄露信息)
        if self.hidden_prize and not viewer_is_omniscient and pre_bid_phase:
            # 不向玩家透露有效大小 (否则可以反推 current_prize)
            total_stake_visible: int = 0
        else:
            total_stake_visible = (
                (obs_current_prize + carry_pool) if obs_current_prize is not None else 0
            )

        # private_info 过滤 (VEIL §12 — 仅得主可见)
        if viewer_is_omniscient:
            obs_private_info: Dict[PlayerId, Optional[str]] = dict(self.private_info)
        else:
            # viewer 只能看见自己的情报; 对手情报强制 None
            opponent = PLAYER_1 if viewer == PLAYER_0 else PLAYER_0
            obs_private_info = {
                viewer: self.private_info.get(viewer),
                opponent: None,
            }

        obs: Dict = {
            # ---- 经典字段 (语义 100% 不变, 旧代码无需改) ----
            "round": round,
            "current_prize": obs_current_prize,
            "scores": scores,
            "remaining_cards": remaining_cards,
            "remaining_prizes": remaining_prizes,
            "carry_pool": carry_pool,
            "total_prize_at_stake": total_stake_visible,
            "done": done,
            "result": self.result(),
            "tie_rule": self.tie_rule,
            # ---- VEIL 扩展字段 (新增, 旧代码忽略即可) ----
            "veil": {
                "hidden_prize": self.hidden_prize,
                "suit_tiebreak": self.suit_tiebreak,
                "info_reward_enabled": self.info_reward_enabled,
                "info_bits_mode": self.info_bits_mode,
                # 当前轮 suit 分配 (suit_tiebreak=False 时为 None)
                "suits": dict(self.current_suits) if self.suit_tiebreak else None,
                "suits_display": (
                    {
                        pid: SUIT_SYMBOLS[self.current_suits[pid]]
                        for pid in PLAYERS
                    }
                    if self.suit_tiebreak else None
                ),
                # pre_bid 阶段 + hidden_prize → True 标记 (UI 显示"?"占位)
                "prize_is_currently_hidden": bool(
                    self.hidden_prize and pre_bid_phase and not viewer_is_omniscient
                ),
                # 每玩家私人情报 (已根据 viewer 过滤)
                "private_info": obs_private_info,
                # 历史"谁在上一轮获得了情报" (公开, 因为是最低出价 → 可从动作推断归属 suit)
                "last_info_awarded_to": (
                    dict(self.info_reward_log[-1]) if self.info_reward_log else None
                ),
            },
        }
        return obs

    # ------------------------------------------------------------- internal
    def _reveal_next_prize(self) -> None:
        """Pop the next prize off the deck and increment round."""
        if not self.remaining_prizes:
            self.current_prize = None
            return
        self.current_prize = self.remaining_prizes.pop(0)
        self.round += 1

    def _reshuffle_suits_if_enabled(self) -> None:
        """VEIL §6-7 / §18: suit_tiebreak 模式下每轮重新随机分配 2 个不重复 suit。"""
        if not self.suit_tiebreak:
            # 经典模式: 仍然设为默认值 (C=0), 但不参与比较
            self.current_suits = {PLAYER_0: 0, PLAYER_1: 0}
            return
        # 从 4 个 suit 里不重复抽 2 个 (VEIL §6 四人模式的 2 人退化)
        all_suits = [0, 1, 2, 3]  # C, D, H, S
        self._rng.shuffle(all_suits)
        s0, s1 = all_suits[0], all_suits[1]
        # 随机分配给 P0 / P1
        if self._rng.random() < 0.5:
            self.current_suits = {PLAYER_0: s0, PLAYER_1: s1}
        else:
            self.current_suits = {PLAYER_0: s1, PLAYER_1: s0}

    def _compute_next_prize_half_signal(self) -> str:
        """
        VEIL §12: 计算下一轮剩余奖励集 R_{t+1} 的 LOW / HIGH 二分信号。
        PRE: remaining_prizes 中第一个元素即为 p_{t+1} (下一轮将揭晓的奖励)。
        """
        # 排序 R_{t+1}
        R_sorted = sorted(self.remaining_prizes)
        k = len(R_sorted)
        if k <= 0:
            # 最后一轮 (不应触发: 调用方会检查 not is_last_round)
            return INFO_LOW
        # R_LOW = 前 ceil(k/2) 张 (VEIL §13)
        size_low = int(math.ceil(k / 2))
        R_LOW = set(R_sorted[:size_low])
        p_next = self.remaining_prizes[0]  # 注意: 未排序的 remaining_prizes[0] == p_{t+1}
        if p_next in R_LOW:
            return INFO_LOW
        return INFO_HIGH

    def _expire_private_info_after_reveal(self) -> None:
        """
        消费掉"针对目标轮次已揭晓"的私人情报。
        逻辑: info_for_round == current round (即本 step 执行前针对的"这一轮"揭晓)
              → 情报使命完成, 立即清空对应玩家的 private_info。
        """
        if not self.info_reward_log:
            return
        last = self.info_reward_log[-1]
        if int(last["for_round"]) == self.round:
            # 这条情报的目标轮就是当前正在进行的 round → 消费
            player = last["lowest_player"]
            if player in self.private_info:
                self.private_info[player] = None

    def _assert_legal_action(self, player: PlayerId, action: Action) -> None:
        """Raise ValueError if `action` is not legal for `player`."""
        legal = self.remaining_cards[player]
        if action not in legal:
            raise ValueError(
                f"Illegal action {action} for {player}. "
                f"Remaining cards: {sorted(legal)}"
            )


# Keep legacy symbols exported (backwards compat)
__all__ = [
    "GoofspielEnv",
    "PlayerId", "Action", "Rng",
    "PLAYER_0", "PLAYER_1", "PLAYERS",
    "SUIT_RANKS", "SUIT_SYMBOLS", "SUIT_LABELS",
    "INFO_LOW", "INFO_HIGH", "INFO_MODES",
    "TIE_RULE_ROLLOVER", "TIE_RULE_DISCARD", "TIE_RULE_SPLIT",
    "TIE_RULES", "DEFAULT_TIE_RULE",
]
