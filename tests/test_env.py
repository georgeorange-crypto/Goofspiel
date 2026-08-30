"""
Author: 陈子聪 (Chen Zicong)
Date: 2026-08-30
Purpose: Unit tests for the Goofspiel environment (平局 carry-over 变体).

Covers all mandatory scenarios — adapted to the new tie rule:
  - reset -> each player has 1..N
  - prize deck contains exactly 1..N
  - each card can be used only once
  - illegal action raises
  - higher card beats lower card
  - equal card -> tie
  - tie and NOT last round -> prize_at_stake ADDED TO carry_pool (not discarded!)
  - tie and LAST round -> prize_at_stake PERMANENTLY discarded (only case of discard)
  - a win after one or more ties takes prize_at_stake (= round_prize + Σ carry)
  - one game == exactly N rounds (default 13)
  - fixed seed is reproducible
  - step() requires both player actions at once
  - observation exposes carry_pool / total_prize_at_stake
  - every history entry carries: round_prize, carry_in, prize_at_stake, carry_out, discarded
"""

from __future__ import annotations

import random
import pytest

from goofspiel import GoofspielEnv, RandomBot, PLAYER_0, PLAYER_1


# ------------------------------------------------------------------- utils
def _seeded_env(seed: int = 42, num_cards: int = 13) -> GoofspielEnv:
    """Create an environment with a deterministic seeded RNG."""
    return GoofspielEnv(num_cards=num_cards, rng=random.Random(seed))


# ========================================================================
# RESET / INIT tests
# ========================================================================
class TestReset:
    def test_reset_gives_each_player_1_to_13(self):
        """reset 后双方都有 1~13"""
        env = _seeded_env()
        obs = env.reset()
        expected = list(range(1, 14))
        assert sorted(obs["remaining_cards"][PLAYER_0]) == expected
        assert sorted(obs["remaining_cards"][PLAYER_1]) == expected

    def test_prize_deck_contains_1_to_13(self):
        """prize deck 恰好包含 1~13 (shuffled but complete)"""
        env = _seeded_env()
        env.reset()
        assert sorted(env.prize_deck) == list(range(1, 14))
        all_prizes = ([env.current_prize] if env.current_prize else []) + env.remaining_prizes
        assert sorted(all_prizes) == list(range(1, 14))

    def test_reset_returns_valid_observation(self):
        """Observation has all required keys (incl. new carry fields) after reset."""
        env = _seeded_env()
        obs = env.reset()
        assert set(obs.keys()) >= {
            "round", "current_prize", "scores", "remaining_cards",
            "remaining_prizes", "carry_pool", "total_prize_at_stake",
            "done", "result",
        }
        assert obs["round"] == 1
        assert obs["done"] is False
        assert obs["result"] is None
        assert obs["current_prize"] is not None
        # At round 1 nothing has been tied yet -> carry must be 0, stake = just prize.
        assert obs["carry_pool"] == 0
        assert obs["total_prize_at_stake"] == obs["current_prize"]


# ========================================================================
# CARD USAGE tests
# ========================================================================
class TestCardUsage:
    def test_each_card_used_only_once(self):
        """每张牌只能使用一次 -> illegal after play."""
        env = _seeded_env(seed=1)
        env.reset()
        a0 = env.remaining_cards[PLAYER_0][0]
        a1 = env.remaining_cards[PLAYER_1][0]
        env.step({PLAYER_0: a0, PLAYER_1: a1})
        assert a0 not in env.remaining_cards[PLAYER_0]
        assert a1 not in env.remaining_cards[PLAYER_1]
        with pytest.raises(ValueError):
            env.step({PLAYER_0: a0, PLAYER_1: env.remaining_cards[PLAYER_1][0]})


# ========================================================================
# ILLEGAL ACTION tests
# ========================================================================
class TestIllegalActions:
    def test_illegal_action_raises(self):
        """非法动作报错 (out of range)."""
        env = _seeded_env()
        env.reset()
        with pytest.raises(ValueError):
            env.step({PLAYER_0: 99, PLAYER_1: 1})

    def test_step_requires_both_players(self):
        """step() 必须同时包含双方 action"""
        env = _seeded_env()
        env.reset()
        with pytest.raises(ValueError):
            env.step({PLAYER_0: 1})
        with pytest.raises(ValueError):
            env.step({PLAYER_1: 1})
        with pytest.raises(ValueError):
            env.step({})

    def test_step_after_done_raises(self):
        """Can't step on a finished game."""
        env = _seeded_env(num_cards=1)
        env.reset()
        obs, _, done, _ = env.step({PLAYER_0: 1, PLAYER_1: 1})
        assert done is True
        with pytest.raises(RuntimeError):
            env.step({PLAYER_0: 1, PLAYER_1: 1})


# ========================================================================
# SCORING / CARRY tests (核心 — 验证新规则)
# ========================================================================
class TestScoring:
    # ---------------------------------------------------------------- basic
    def test_higher_card_wins_clean(self):
        """无 carry 时：大牌赢 → 胜者得单 prize，carry_pool == 0，winner info 正确。"""
        env = GoofspielEnv(num_cards=3, rng=random.Random(0))
        env.reset()
        prize = env.current_prize
        assert env.carry_pool == 0
        obs, rewards, done, info = env.step({PLAYER_0: 3, PLAYER_1: 1})
        assert info["winner"] == PLAYER_0
        assert info["round_prize"] == prize
        assert info["carry_in"] == 0
        assert info["prize_at_stake"] == prize
        assert info["carry_out"] == 0
        assert info["discarded"] is False
        assert rewards == {PLAYER_0: prize, PLAYER_1: 0}
        assert obs["scores"] == {PLAYER_0: prize, PLAYER_1: 0}
        assert obs["carry_pool"] == 0
        assert not done  # num_cards=3, still 2 prizes left.

    def test_equal_card_tie_semantics(self):
        """相同牌平局：winner None / rewards 0 / 若不是末轮 carry_pool == prize。"""
        env = GoofspielEnv(num_cards=2, rng=random.Random(0))
        env.reset()
        prize_1 = env.current_prize
        assert prize_1 is not None and prize_1 > 0
        obs, rewards, done, info = env.step({PLAYER_0: 1, PLAYER_1: 1})
        assert info["winner"] is None
        assert rewards == {PLAYER_0: 0, PLAYER_1: 0}
        # R1 is NOT the last round (num_cards=2  => after R1, one more prize).
        assert done is False
        assert info["discarded"] is False
        assert info["carry_out"] == prize_1
        assert info["carry_in"] == 0
        assert info["prize_at_stake"] == prize_1
        # 平局不丢 → carry_pool == prize_1
        assert env.carry_pool == prize_1
        assert obs["carry_pool"] == prize_1
        # 下一轮观察：total_prize_at_stake = 新翻开 prize_2 + carry
        expected_stake = (obs["current_prize"] or 0) + obs["carry_pool"]
        assert obs["total_prize_at_stake"] == expected_stake

    # ------------------------------------------------------------- carry 核心
    def test_non_last_tie_accumulates_carry(self):
        """非末轮平局 → carry_pool = prize_at_stake，分数不变。"""
        env = GoofspielEnv(num_cards=3, rng=random.Random(5))
        env.reset()
        p1 = env.current_prize
        assert p1 > 0
        obs, _, _, _ = env.step({PLAYER_0: 2, PLAYER_1: 2})
        assert obs["scores"] == {PLAYER_0: 0, PLAYER_1: 0}
        assert env.carry_pool == p1
        assert env.history[-1]["carry_out"] == p1
        assert env.history[-1]["discarded"] is False

    def test_consecutive_ties_accumulate_sum(self):
        """连续两次平局 → carry 增长为 p1 + p2。"""
        env = GoofspielEnv(num_cards=3, rng=random.Random(5))
        env.reset()
        p1 = env.current_prize
        env.step({PLAYER_0: 1, PLAYER_1: 1})           # R1 tie -> carry = p1
        p2 = env.current_prize
        # R2 也平局，不是末轮（N=3，还有 R3）
        obs, _, _, info = env.step({PLAYER_0: 2, PLAYER_1: 2})
        assert info["winner"] is None
        assert info["carry_in"] == p1
        assert info["prize_at_stake"] == p2 + p1
        assert info["carry_out"] == p1 + p2
        assert info["discarded"] is False
        assert env.carry_pool == p1 + p2
        assert obs["carry_pool"] == p1 + p2

    def test_win_after_single_tie_takes_full_prize_at_stake(self):
        """平局后下一轮赢 → 胜者获得 p_prev (carry) + p_current 总和。"""
        env = GoofspielEnv(num_cards=3, rng=random.Random(5))
        env.reset()
        p1 = env.current_prize
        env.step({PLAYER_0: 1, PLAYER_1: 1})   # R1 tie, carry = p1
        p2 = env.current_prize
        assert env.carry_pool == p1
        obs, rewards, _, info = env.step({PLAYER_0: 3, PLAYER_1: 2})  # R2 player_0 wins
        assert info["winner"] == PLAYER_0
        stake = p1 + p2
        assert info["carry_in"] == p1
        assert info["prize_at_stake"] == stake
        assert info["carry_out"] == 0          # 清空
        assert rewards[PLAYER_0] == stake
        assert rewards[PLAYER_1] == 0
        assert obs["scores"][PLAYER_0] == stake
        assert obs["scores"][PLAYER_1] == 0
        assert env.carry_pool == 0

    def test_win_after_double_tie_takes_cumulative_stake(self):
        """两次平局后赢 → 胜者得 prize_R3 + carry = p1+p2+p3。"""
        # N=4 so we have at least one non-tie round after two ties.
        env = GoofspielEnv(num_cards=4, rng=random.Random(2))
        env.reset()
        p1 = env.current_prize
        env.step({PLAYER_0: 1, PLAYER_1: 1})   # R1 tie
        p2 = env.current_prize
        env.step({PLAYER_0: 2, PLAYER_1: 2})   # R2 tie, carry = p1 + p2
        p3 = env.current_prize
        stake_expected = p1 + p2 + p3
        obs, rewards, _, info = env.step({PLAYER_0: 4, PLAYER_1: 3})  # p0 wins R3
        assert info["prize_at_stake"] == stake_expected
        assert info["carry_out"] == 0
        assert rewards[PLAYER_0] == stake_expected
        assert obs["scores"][PLAYER_0] == stake_expected
        assert env.carry_pool == 0

    def test_last_round_tie_discards_total_prize_at_stake(self):
        """末轮平局 → prize_at_stake PERMANENTLY discarded（唯一会丢奖的场景）。"""
        # Simplest: N=1, only one round. Any action is a tie.
        env = GoofspielEnv(num_cards=1, rng=random.Random(0))
        env.reset()
        p1 = env.current_prize  # = 1
        assert p1 == 1
        obs, rewards, done, info = env.step({PLAYER_0: 1, PLAYER_1: 1})
        assert done is True
        assert info["winner"] is None
        assert rewards == {PLAYER_0: 0, PLAYER_1: 0}
        assert obs["scores"] == {PLAYER_0: 0, PLAYER_1: 0}
        assert info["prize_at_stake"] == 1
        assert info["carry_out"] == 0
        assert info["discarded"] is True
        assert env.carry_pool == 0
        assert env.result() == "draw"

    def test_last_round_tie_also_discards_previous_carry(self):
        """N=2: R1 tie (carry=p1) + R2 tie (末轮) → total=p1+p2 整包丢弃。"""
        env = GoofspielEnv(num_cards=2, rng=random.Random(0))
        env.reset()
        p1 = env.current_prize
        env.step({PLAYER_0: 1, PLAYER_1: 1})   # R1 tie carry = p1
        assert env.carry_pool == p1
        p2 = env.current_prize
        obs, rewards, done, info = env.step({PLAYER_0: 2, PLAYER_1: 2})  # R2 tie LAST
        assert done is True
        total = p1 + p2
        assert info["carry_in"] == p1
        assert info["prize_at_stake"] == total
        assert info["carry_out"] == 0
        assert info["discarded"] is True
        assert rewards == {PLAYER_0: 0, PLAYER_1: 0}
        assert obs["scores"] == {PLAYER_0: 0, PLAYER_1: 0}
        assert env.carry_pool == 0
        assert env.result() == "draw"


# ========================================================================
# OBSERVATION / HISTORY contract tests
# ========================================================================
class TestObsHistoryContracts:
    def test_history_every_entry_has_carry_fields(self):
        """每一条 history 条目必须包含：round_prize, carry_in, prize_at_stake, carry_out, discarded."""
        env = _seeded_env(seed=7, num_cards=3)
        env.reset()
        while not env.done:
            env.step({
                PLAYER_0: env.remaining_cards[PLAYER_0][0],
                PLAYER_1: env.remaining_cards[PLAYER_1][0],
            })
        assert len(env.history) == 3
        for entry in env.history:
            for key in ("round_prize", "carry_in", "prize_at_stake",
                        "carry_out", "discarded"):
                assert key in entry, f"history entry missing key {key}: {entry}"
            # prize_at_stake invariant: always round_prize + carry_in for this round.
            assert entry["prize_at_stake"] == entry["round_prize"] + entry["carry_in"]

    def test_observation_carry_matches_env_carry(self):
        """get_observation() 的 carry_pool 字段永远等价于 env.carry_pool；total_prize_at_stake == prize + carry (或 game over 时 0)."""
        env = GoofspielEnv(num_cards=3, rng=random.Random(4))
        env.reset()
        # R1 tie → carry = prize_1
        env.step({PLAYER_0: 1, PLAYER_1: 1})
        obs = env.get_observation()
        assert obs["carry_pool"] == env.carry_pool == env.history[-1]["carry_out"]
        assert obs["total_prize_at_stake"] == (obs["current_prize"] or 0) + obs["carry_pool"]
        # R2 player_0 wins → carry reset to 0
        env.step({PLAYER_0: 3, PLAYER_1: 2})
        obs = env.get_observation()
        assert obs["carry_pool"] == 0
        assert env.carry_pool == 0
        # 最后一轮：如果 done，current_prize 为 None → stake = 0 + 0 = 0
        if obs["done"]:
            assert obs["current_prize"] is None
            assert obs["total_prize_at_stake"] == 0

    def test_prize_conservation_no_final_discard(self):
        """若没有触发末轮平局丢弃，则玩家总分 ≡ N(N+1)/2（奖池严格守恒）。"""
        # 最简单：N=1 且胜负有分（non-tie impossible 因为只剩 1 对 1，所以换 N=2，且保证不是末轮平局）。
        # 构造 N=2：R1 一胜一负（保证 R2 前 carry=0；R2 再一胜一负）→ 全程无末轮平局 discard → 总分 3
        env = GoofspielEnv(num_cards=2, rng=random.Random(1))
        env.reset()
        # R1: winner p0, prize p1
        env.step({PLAYER_0: 2, PLAYER_1: 1})
        # R2: one card left each; one must win since hand sizes differ? No: N=2 after one action, hands are [1] and [2]; tie is impossible.
        assert sorted(env.remaining_cards[PLAYER_0]) == [1]
        assert sorted(env.remaining_cards[PLAYER_1]) == [2]
        env.step({PLAYER_0: 1, PLAYER_1: 2})  # p1 wins prize_2
        total_score = env.scores[PLAYER_0] + env.scores[PLAYER_1]
        assert total_score == 2 * 3 // 2, f"Expected total=3 (no final discard), got {total_score}"
        # 验证未触发 discard
        discarded_rounds = [h for h in env.history if h["discarded"]]
        assert discarded_rounds == []

    def test_prize_conservation_with_final_discard_matches(self):
        """触发末轮平局 discard 时：玩家总分 + discarded_prize = N(N+1)/2。"""
        env = GoofspielEnv(num_cards=2, rng=random.Random(0))
        env.reset()
        env.step({PLAYER_0: 1, PLAYER_1: 1})  # R1 tie carry=p1
        env.step({PLAYER_0: 2, PLAYER_1: 2})  # R2 tie LAST discard total=p1+p2=3
        sum_scores = env.scores[PLAYER_0] + env.scores[PLAYER_1]
        discarded = sum(h["prize_at_stake"] for h in env.history if h["discarded"])
        # 末轮平局 discard 的就是最后一条的 prize_at_stake（已含 carry）
        assert sum_scores + discarded == 2 * 3 // 2


# ========================================================================
# FULL GAME duration tests
# ========================================================================
class TestGameDuration:
    def test_13_rounds_exactly(self):
        """一局恰好 13 轮 for default num_cards=13."""
        env = _seeded_env(seed=7)
        obs = env.reset()
        assert obs["round"] == 1
        rounds_played = 0
        while not env.done:
            a0 = env.remaining_cards[PLAYER_0][0]
            a1 = env.remaining_cards[PLAYER_1][0]
            obs, _, _, _ = env.step({PLAYER_0: a0, PLAYER_1: a1})
            rounds_played += 1
        assert rounds_played == 13
        assert env.round == 13
        assert len(env.history) == 13
        assert env.result() is not None

    def test_num_cards_param(self):
        """Arbitrary num_cards = N rounds."""
        for N in [1, 2, 5, 13]:
            env = _seeded_env(seed=N, num_cards=N)
            env.reset()
            played = 0
            while not env.done:
                a0 = env.remaining_cards[PLAYER_0][0]
                a1 = env.remaining_cards[PLAYER_1][0]
                env.step({PLAYER_0: a0, PLAYER_1: a1})
                played += 1
            assert played == N, f"num_cards={N} expected {N} rounds, got {played}"


# ========================================================================
# REPRODUCIBILITY tests
# ========================================================================
class TestReproducibility:
    def test_fixed_seed_reproduces_prize_order_and_outcome(self):
        """fixed seed 可以复现: two envs with same seed produce identical prize deck, history, scores, result."""
        def run_game(seed: int):
            bot0 = RandomBot(rng=random.Random(seed))
            bot1 = RandomBot(rng=random.Random(seed + 1))
            env = GoofspielEnv(num_cards=13, rng=random.Random(seed))
            env.reset()
            while not env.done:
                a0 = bot0.choose_action(env, PLAYER_0)
                a1 = bot1.choose_action(env, PLAYER_1)
                env.step({PLAYER_0: a0, PLAYER_1: a1})
            return env.prize_deck, env.history, env.scores, env.result()

        prize_deck_a, history_a, scores_a, result_a = run_game(seed=123)
        prize_deck_b, history_b, scores_b, result_b = run_game(seed=123)

        assert prize_deck_a == prize_deck_b
        assert history_a == history_b
        assert scores_a == scores_b
        assert result_a == result_b


# ========================================================================
# LEGAL ACTIONS / RESULT tests
# ========================================================================
class TestLegalAndResult:
    def test_legal_actions_matches_remaining(self):
        env = _seeded_env()
        env.reset()
        for pid in (PLAYER_0, PLAYER_1):
            assert env.legal_actions(pid) == sorted(env.remaining_cards[pid])

    def test_result_before_done_is_none(self):
        env = _seeded_env()
        env.reset()
        assert env.result() is None

    def test_result_is_draw_if_equal_scores(self):
        """N=1, final tie (discard) → scores are 0-0 → draw."""
        env = GoofspielEnv(num_cards=1, rng=random.Random(0))
        env.reset()
        env.step({PLAYER_0: 1, PLAYER_1: 1})
        assert env.result() == "draw"

    def test_result_distinguishes_winner(self):
        """N=2, R1 player_0 wins, R2 player_1 wins → compare totals match result()."""
        env = GoofspielEnv(num_cards=2, rng=random.Random(0))
        env.reset()
        env.step({PLAYER_0: 2, PLAYER_1: 1})   # R1 p0 wins prize_1
        env.step({PLAYER_0: 1, PLAYER_1: 2})   # R2 p1 wins prize_2
        s0, s1 = env.scores[PLAYER_0], env.scores[PLAYER_1]
        expected = PLAYER_0 if s0 > s1 else (PLAYER_1 if s1 > s0 else "draw")
        assert env.result() == expected
