"""
Author: 陈子聪 (Chen Zicong)
Date: 2026-08-30
Purpose: Tests for the configurable Goofspiel web app + new bot family.

Coverage:
  (A) RandomBot / HeuristicBot / NashBot:
        - choose_action always returns a LEGAL card from remaining set
        - distributions sum to ~100%
        - NashBot on N=5 returns finite game-theoretic values
  (B) NashBot on N > NASH_MAX_N silently falls back to HeuristicBot.
  (C) FastAPI TestClient endpoints:
        - GET /api/game/config
        - POST /api/game/new  with various (N, bot)
        - POST /api/game/play  full game loop (Heuristic vs fixed human action)
        - HTTP 422 on invalid num_cards / invalid bot_type
        - HTTP 400 on illegal action / no active game
  (D) Play 100 games of (Random vs Heuristic) N=13:
        Sanity-check Heuristic is strictly stronger than Random
        (score-diff p-value via one-sided t-test on win ratio is not required,
         we only assert *all actions were legal* + reasonable mean diff).

Fast tests (not touching N=5 exact Nash solve) use pytest skipif so the
default test run stays sub-minute.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Tell pytest about our custom "slow" marker to avoid warnings
pytest_plugins: List[str] = []

def pytest_configure(config):  # pragma: no cover - side-effect at test session start
    config.addinivalue_line("markers", "slow: mark test as slow (Nash exact solve runs)")

# ---- UUT imports --------------------------------------------------------
from goofspiel import (
    GoofspielEnv,
    RandomBot,
    HeuristicBot,
    NashBot,
    create_bot,
    BOT_RANDOM,
    BOT_HEURISTIC,
    BOT_NASH,
    BOT_NASH_CARRY,
    BOT_TYPES,
    NASH_MAX_N,
    NASH_CARRY_MAX_N,
    EXACT_MODE_CLASSIC,
    EXACT_MODE_CARRY,
    PLAYER_0,
    PLAYER_1,
)
from goofspiel.bots import BaseBot
import app as app_module


# =========================================================================
# A. Bot-level tests
# =========================================================================
def _play_full_game(bot0: BaseBot, bot1: BaseBot, n: int,
                    seed: int = 0) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Self-play helper, returns final-scores + per-round policy info lists.

    Prize-deck is made deterministic by constructing the GoofspielEnv with a
    seeded random.Random (it uses self._rng.shuffle() in reset()).
    """
    import random as _r
    rng = _r.Random(seed)
    env_rng = _r.Random(rng.randint(0, 2**31 - 1))
    env = GoofspielEnv(num_cards=n, rng=env_rng)
    env.reset()
    p0_infos = []
    p1_infos = []
    while not env.done:
        a0, i0 = bot0.choose_action_with_policy(env, PLAYER_0)
        a1, i1 = bot1.choose_action_with_policy(env, PLAYER_1)
        p0_infos.append(i0)
        p1_infos.append(i1)
        env.step({PLAYER_0: a0, PLAYER_1: a1})
    result_val = env.result()   # env.result is the *method* —— previously was
    # mistakenly returning the bound method itself rather than its return value.
    return {
        "s0": env.scores[PLAYER_0],
        "s1": env.scores[PLAYER_1],
        "result": result_val,
        "done": env.done,
    }, p0_infos, p1_infos


class TestRandomBot:
    def test_legal_actions_uniform_dist(self):
        b = RandomBot()
        env = GoofspielEnv(num_cards=13)
        env.reset()
        # Check 3 times
        for _ in range(3):
            action, info = b.choose_action_with_policy(env, PLAYER_0)
            assert action in env.legal_actions(PLAYER_0)
            assert info["bot_type"] == BOT_RANDOM
            dist = info["distribution"]
            vals, pcts = zip(*dist)
            assert set(vals) == set(env.legal_actions(PLAYER_0))
            # Uniform 1/k => sum ~ 100, each equal
            assert abs(sum(pcts) - 100.0) < 1e-6
            assert len(set(pcts)) == 1  # all equal

    def test_legacy_choose_action(self):
        """Legacy single-action API compat."""
        b = RandomBot()
        env = GoofspielEnv(5)
        env.reset()
        a = b.choose_action(env, PLAYER_1)
        assert a in env.legal_actions(PLAYER_1)


class TestHeuristicBot:
    @pytest.mark.parametrize("n", [3, 5, 13])
    def test_all_actions_legal_full_game(self, n):
        import random as _r
        h0 = HeuristicBot(rng=_r.Random(10))
        h1 = RandomBot(rng=_r.Random(11))
        final, infos, _ = _play_full_game(h0, h1, n, seed=0)
        # env.result() returns PLAYER_0/PLAYER_1/"draw" when done
        assert final["result"] in ("player_0", "player_1", "draw"), (
            f"Unexpected result: {final['result']!r}  scores={final}"
        )
        # All policy distributions sum ~100 and values are NaN (no GT value)
        for info in infos:
            assert info["bot_type"] == BOT_HEURISTIC
            assert math.isnan(info["value"])
            total = sum(p for _, p in info["distribution"])
            assert abs(total - 100.0) < 1e-6

    def test_heuristic_better_than_random_on_average(self):
        """Sanity: Heuristic beats Random most games."""
        import random as _r
        wins, losses, draws = 0, 0, 0
        for s in range(50):
            h = HeuristicBot(rng=_r.Random(1000 + s))
            r = RandomBot(rng=_r.Random(2000 + s))
            final, _, _ = _play_full_game(h, r, 13, seed=s)
            r0 = final["result"]
            assert r0 in ("player_0", "player_1", "draw"), f"unexpected result={r0}"
            if r0 == "player_0": wins += 1
            elif r0 == "player_1": losses += 1
            else: draws += 1
        # Expected: Heuristic clearly dominates Random
        assert wins > losses, f"Heuristic W/D/L = {wins}/{draws}/{losses}"


class TestNashBot:
    def test_factory_unknown_bot_raises(self):
        with pytest.raises(ValueError):
            create_bot("foo")

    def test_factory_random_heuristic(self):
        assert isinstance(create_bot(BOT_RANDOM, seed=1), RandomBot)
        assert isinstance(create_bot(BOT_HEURISTIC, seed=1), HeuristicBot)
        nb = create_bot(BOT_NASH, seed=1)
        assert isinstance(nb, NashBot)
        # Make sure constructors don't raise when seed=None (default case)
        nb_no_seed = NashBot()
        assert isinstance(nb_no_seed, NashBot)

    def test_nash_fallback_when_n_too_big(self):
        """N=13 must NOT attempt the exact solve; returns heuristic distribution."""
        nb = NashBot(max_nash_n=3)
        env = GoofspielEnv(num_cards=13)
        env.reset()
        a, info = nb.choose_action_with_policy(env, PLAYER_1)
        assert a in env.legal_actions(PLAYER_1)
        # Note clearly shows fallback
        assert "fallback" in info["note"].lower() or "回落" in info["note"]
        # Value is NaN (fallback to heuristic)
        assert math.isnan(info["value"])

    @pytest.mark.slow
    def test_nash_n5_exact_policy_values(self):
        """Run N=5 exact solve, every step has finite policy value."""
        import random as _r
        nb = NashBot(rng=_r.Random(42))
        opp = RandomBot(rng=_r.Random(43))
        final, _, p1_infos = _play_full_game(opp, nb, 5, seed=1)
        assert final["result"] in ("player_0", "player_1", "draw")
        for info in p1_infos:
            assert math.isfinite(info["value"]), f"bad Nash value: {info}"
            total = sum(p for _, p in info["distribution"])
            assert abs(total - 100.0) < 1e-3


# =========================================================================
# B. API-level tests via FastAPI TestClient
# =========================================================================
@pytest.fixture
def client() -> TestClient:
    return TestClient(app_module.app)


class TestConfigEndpoint:
    def test_config_shape(self, client: TestClient):
        r = client.get("/api/game/config")
        assert r.status_code == 200
        body = r.json()
        assert "num_cards" in body
        assert body["num_cards"]["min"] == 1
        assert body["num_cards"]["max"] == 13
        assert set(b["id"] for b in body["bots"]) == set(BOT_TYPES)


class TestNewGameEndpoint:
    def test_new_game_default_empty_body(self, client: TestClient):
        """Compat: old-style empty POST -> starts 13/Random."""
        r = client.post("/api/game/new")
        assert r.status_code == 200
        body = r.json()
        assert body["state"]["num_cards"] == 13
        # With no body, backend uses DEFAULT_BOT = BOT_RANDOM
        assert body["meta"]["requested_bot"] == BOT_RANDOM
        assert body["meta"]["actual_bot"] == BOT_RANDOM

    @pytest.mark.parametrize("n", [1, 5, 7, 13])
    def test_new_game_heuristic_any_n(self, client: TestClient, n: int):
        r = client.post("/api/game/new",
                        json={"num_cards": n, "bot_type": BOT_HEURISTIC})
        assert r.status_code == 200
        body = r.json()
        assert body["state"]["num_cards"] == n
        assert body["meta"]["actual_bot"] == BOT_HEURISTIC

    def test_new_game_nash_n13_fallback(self, client: TestClient):
        """N=13 Nash -> backend auto uses Heuristic, fallback_reason present."""
        r = client.post("/api/game/new",
                        json={"num_cards": 13, "bot_type": BOT_NASH})
        assert r.status_code == 200
        body = r.json()
        # requested_bot is still Nash
        assert body["meta"]["requested_bot"] == BOT_NASH
        # Backend records why a fallback happened
        reason = str(body["meta"]["fallback_reason"])
        assert "fallback" in reason.lower() or "回落" in reason
        # The actual bot object deployed IS Heuristic (not a NashBot that
        # happens to fall back at play-time —— 直接落成实例就应该是
        # HeuristicBot，这样后续出牌才能0延迟。
        assert isinstance(app_module._bot, HeuristicBot)
        # `meta.actual_bot` keeps showing Heuristic because the user's real
        # experience IS Heuristic play; the label describes actual behavior.
        assert body["meta"]["actual_bot"] == BOT_HEURISTIC

    @pytest.mark.parametrize("bad_n,bad_bot", [
        (0, BOT_RANDOM),
        (14, BOT_RANDOM),
        (5, "invalid-bot-name"),
    ])
    def test_new_game_validation_errors(self, client: TestClient, bad_n, bad_bot):
        r = client.post("/api/game/new",
                        json={"num_cards": bad_n, "bot_type": bad_bot})
        assert r.status_code == 422

    def test_new_game_veil_flags_roundtrip_to_state_and_fallback(self, client: TestClient):
        """
        /api/game/new 三个 VEIL 机制 flag 作为 top-level JSON 字段提交，
        后端必须：
          (1) state.veil.enabled == any(flags)（实际真值在 state.veil）。
          (2) 每个开启的 flag 在 state.veil 里对应字段为 True（hidden_prize /
              suit_tiebreak / info_reward_enabled）。
          (3) meta.veil.any_enabled / active_tags 同步；同时如果选 Nash +
              任一 VEIL 开，meta.actual_bot 诚实回落为 Heuristic 且
              fallback_reason 非空（回落语义红线）。
        这条测试把前端 collectVeilFlags → payload {...veil} 铺开后的 JSON
        契约固定下来。GUI toggle-switch 改版后只要后端契约不变就仍 PASSED。
        """
        # Case A: no veil flags at all → classic mode
        r = client.post("/api/game/new",
                        json={"num_cards": 5, "bot_type": BOT_HEURISTIC})
        assert r.status_code == 200
        body = r.json()
        s_veil = body["state"].get("veil") or {}
        m_veil = body["meta"].get("veil") or {}
        assert s_veil.get("enabled") in (False, None, 0) or not s_veil.get("enabled")
        assert m_veil.get("any_enabled") in (False, None)

        # Case B: hidden_prize + info_reward on → 2 flags on, suit off
        r = client.post("/api/game/new", json={
            "num_cards": 5,
            "bot_type": BOT_HEURISTIC,
            "veil_hidden_prize": True,
            "veil_info_reward":  True,
            "veil_suit_tiebreak": False,
        })
        assert r.status_code == 200, r.json()
        body = r.json()
        s_veil = body["state"]["veil"]
        m_veil = body["meta"]["veil"]
        assert s_veil.get("enabled") is True, f"state.veil.enabled expected True; got {s_veil}"
        assert s_veil.get("hidden_prize") is True
        assert s_veil.get("info_reward_enabled") is True
        assert s_veil.get("suit_tiebreak") is False
        assert m_veil["any_enabled"] is True
        assert m_veil["hidden_prize"] is True
        assert m_veil["suit_tiebreak"] is False
        tags = m_veil.get("active_tags") or []
        assert len(tags) >= 2, f"至少 2 个激活 tag, 得 {tags}"
        assert "HiddenPrize" in tags
        assert any("InfoReward" in t for t in tags)

        # Case C: Nash + suit_tiebreak → auto fallback to Heuristic honest
        r = client.post("/api/game/new", json={
            "num_cards": 5,
            "bot_type": BOT_NASH,
            "veil_suit_tiebreak": True,
        })
        assert r.status_code == 200, r.json()
        body = r.json()
        meta = body["meta"]
        assert meta["requested_bot"] == BOT_NASH
        assert meta["actual_bot"] == BOT_HEURISTIC, \
            f"Nash + VEIL → 必须诚实回落 Heuristic，实际 meta={meta}"
        reason = meta.get("fallback_reason") or ""
        assert len(reason) > 0, f"fallback_reason 不能为空 (诚实回落语义红线)"
        m_veil = body["meta"]["veil"]
        s_veil = body["state"]["veil"]
        assert m_veil["any_enabled"] is True
        assert m_veil["suit_tiebreak"] is True
        assert m_veil["hidden_prize"] is False
        assert m_veil["info_reward_enabled"] is False
        assert s_veil["enabled"] is True
        assert s_veil["suit_tiebreak"] is True


class TestPlayEndpoint:
    def test_play_no_game_400(self, client: TestClient):
        # Clear session by resetting globals: the easiest way is call new then
        # set module-level env variable to None via a fresh TestClient + new.
        app_module._env = None
        r = client.post("/api/game/play", json={"action": 1})
        assert r.status_code == 400

    def test_full_game_play_with_heuristic(self, client: TestClient):
        """
        完整走完一局 N=5 + Heuristic:
          - every /play returns 200
          - human action is always just min(remaining)
          - at end, state.done=True + result one of three valid strings
          - every round returns BOTH ai_policy (distribution) AND
            human_policy (counterfactual win/tie/lose bars)
        """
        r = client.post("/api/game/new",
                        json={"num_cards": 5, "bot_type": BOT_HEURISTIC})
        assert r.status_code == 200
        state = r.json()["state"]
        n = state["num_cards"]
        played_human: List[int] = []
        for _ in range(n):
            remaining = [c["value"] for c in state["remaining_cards"]["human"]]
            pick = min(remaining)
            while pick in played_human:
                remaining.remove(pick)
                pick = min(remaining)
            played_human.append(pick)
            resp = client.post("/api/game/play", json={"action": pick})
            assert resp.status_code == 200
            body = resp.json()
            lr = body["last_round"]

            # AI policy
            assert "ai_policy" in lr
            assert "distribution" in lr["ai_policy"]
            assert 1 <= int(lr["bot_action"]) <= n

            # Human counterfactual
            assert "human_policy" in lr
            bars = lr["human_policy"]["bars"]
            # Every bar corresponds to a pre-step human card we just had
            pre_step_cards_set = {int(b["card_value"]) for b in bars}
            # 'remaining' here is pre-pick human hand; pick is still in it.
            assert pre_step_cards_set == set(remaining) | {pick}
            # Exactly one `played=True` per round — matches the card we sent
            played_cf = [b for b in bars if b["played"]]
            assert len(played_cf) == 1
            assert int(played_cf[0]["card_value"]) == pick
            # Outcome values are well-formed + delta consistent
            for b in bars:
                assert b["outcome"] in ("win", "tie", "lose")
                delta = int(b["delta"])
                # NOTE: carry-over 规则下 win delta = prize_at_stake
                # (= round_prize + carry_in)，不再是单轮 prize。
                # 后端 last_round 同时暴露 prize(兼容) 和
                # prize_at_stake(新字段)，取后者做严格相等断言。
                stake = int(lr.get("prize_at_stake", lr["prize"]))
                if b["outcome"] == "win":
                    assert delta == stake, (
                        f"R{lr['round']}: win delta={delta} != "
                        f"prize_at_stake={stake}  (carry_in="
                        f"{lr.get('carry_in', 0)}, round_prize={lr['prize']})"
                    )
                else:
                    assert delta == 0

            state = body["state"]
        assert state["done"] is True
        assert state["result"] in ("player_0", "player_1", "draw")
        max_possible = n * (n + 1) // 2
        assert state["scores"]["human"] + state["scores"]["bot"] <= max_possible

    def test_human_counterfactual_agrees_with_env(self, client: TestClient):
        """
        Force a fixed-bot (RandomBot with seed=7) N=3 game, and on each round
        verify: the API-reported human_policy outcome for the actually played
        human card matches env.history[*].winner semantics.
        """
        import app as _am
        from goofspiel import RandomBot
        import random as _r
        r = client.post("/api/game/new",
                        json={"num_cards": 3, "bot_type": BOT_RANDOM})
        assert r.status_code == 200
        # Force seeded RandomBot bot instance to make this test deterministic
        seeded_bot = RandomBot(rng=_r.Random(7))
        _am._bot = seeded_bot
        # Re-set to a fresh env instance with deterministic prize shuffle
        class _Fixed:
            def shuffle(self, x):
                x[:] = [3, 1, 2]
        env = _am.GoofspielEnv(3)
        env._rng = _Fixed()  # type: ignore[attr-defined]
        env.reset()
        _am._env = env

        # Now drive the game from client; API still uses the seeded bot we injected
        moves = [1, 2, 3]
        for m in moves:
            resp = client.post("/api/game/play", json={"action": m})
            assert resp.status_code == 200
            lr = resp.json()["last_round"]
            bars_by_value = {int(b["card_value"]): b for b in lr["human_policy"]["bars"]}
            played_bar = bars_by_value[m]
            # Outcome vs winner
            winner = lr["winner"]
            expected = {"player_0": "win", "player_1": "lose", "draw": "tie",
                        None: "tie"}[winner]
            assert played_bar["outcome"] == expected, (
                f"R{lr['round']}: got outcome={played_bar['outcome']} "
                f"for played card {m}, winner={winner}"
            )

    def test_counterfactual_bot_waste_field_semantics(self, client: TestClient):
        """
        New dimension "对手亏牌 (bot waste / efficiency)" — verify each
        counterfactual bar carries the 4 expected opponent-analysis fields
        and their values respect the invariant formulas:
          bot_prize_delta  ∈ {0, prize_at_stake}  (tie/lose → 0, bot win → full stake)
          bot_net_gain     == bot_prize_delta − bot_card_value  (may be negative)
          bot_efficiency   ∈ {"wasted","even","profitable"}   matches sign(net)
          bot_efficiency_label matches efficiency class (Chinese label)
          my_net_gain      == delta − hc  (your own card value subtractions)
        """
        import app as _am
        from goofspiel import RandomBot
        import random as _r
        r = client.post("/api/game/new",
                        json={"num_cards": 5, "bot_type": BOT_RANDOM})
        assert r.status_code == 200
        seeded_bot = RandomBot(rng=_r.Random(123))
        _am._bot = seeded_bot

        class _Fixed:
            def shuffle(self, x):
                x[:] = [3, 5, 1, 4, 2]    # prize order: round1 prize=3
        env = _am.GoofspielEnv(5)
        env._rng = _Fixed()  # type: ignore[attr-defined]
        env.reset()
        _am._env = env

        resp = client.post("/api/game/play", json={"action": 1})
        assert resp.status_code == 200, resp.json()
        lr = resp.json()["last_round"]
        bars = lr["human_policy"]["bars"]

        # All bars must share the SAME frozen bot card (counterfactual invariant).
        bot_cards = {int(b["bot_card_value"]) for b in bars}
        assert len(bot_cards) == 1, f"all counterfactuals share one bot card; got {bot_cards}"
        bot_cv = next(iter(bot_cards))
        assert 1 <= bot_cv <= 5

        prize_at_stake = int(lr["human_policy"].get("prize_at_stake", 0))
        assert prize_at_stake >= 1

        for bar in bars:
            hc = int(bar["card_value"])
            delta = int(bar["delta"])
            bpd = int(bar["bot_prize_delta"])
            bng = int(bar["bot_net_gain"])
            mng = int(bar["my_net_gain"])
            eff = bar["bot_efficiency"]
            lbl = bar["bot_efficiency_label"]

            # Formulas invariant
            assert bng == bpd - bot_cv, (
                f"card {hc}: bot_net_gain {bng} should equal bot_prize_delta{bpd} − bot_card{bot_cv}"
            )
            assert mng == delta - hc, (
                f"card {hc}: my_net_gain {mng} should equal delta{delta} − my_card{hc}"
            )

            # eff vs sign(net)
            if bng > 0:
                assert eff == "profitable", f"card {hc} bng={bng} -> expected profitable"
                assert "赚牌" in lbl
            elif bng < 0:
                assert eff == "wasted", f"card {hc} bng={bng} -> expected wasted"
                assert "亏牌" in lbl
            else:
                assert eff == "even", f"card {hc} bng={bng} -> expected even"
                assert "平本" in lbl

            # outcome → bot_prize_delta must match rule
            if bar["outcome"] == "win":
                assert bpd == 0, f"you win → bot gets 0, got {bpd}"
                # When YOU win, bot receives 0 and still loses its card → net<0
                assert bng < 0, f"you win hc={hc}: bot wasted its card, bng should be <0"
            elif bar["outcome"] == "lose":
                assert bpd == prize_at_stake, f"bot wins → bot gets prize_at_stake"
            else:  # tie
                assert bpd == 0, f"tie → bot receives 0 prize this round"
                assert bng < 0, f"tie hc={hc}: bot used a card for 0 prize → wasted"

            # bot_eff_desc must exist and contain the opponent's card display
            desc = str(bar["bot_eff_desc"])
            assert desc, f"card {hc}: bot_eff_desc missing"
            assert bar["bot_card_display"] in desc, (
                f"bot_eff_desc must mention the bot card display; got {desc}"
            )

    def test_play_illegal_action_400(self, client: TestClient):
        client.post("/api/game/new",
                    json={"num_cards": 3, "bot_type": BOT_RANDOM})
        # 99 is never legal (N=3 -> {1,2,3}).
        # 422 = pydantic schema rejects out-of-range (ge/le bounds on action field);
        # 400 = server-level game-rule rejection.  Both are acceptable here so long
        # as the action is refused.
        r = client.post("/api/game/play", json={"action": 99})
        assert r.status_code in (400, 422)


# =========================================================================
# D. 双精确 Nash：HTTP 路由 + 回落透明性契约
# =========================================================================
class TestDualNash:
    """验证两套 Nash 模型的 HTTP 契约：
    (a) config.bots 暴露第 4 项 nash_carry，且 max_n / nash_rule_model 正确。
    (b) new nash_carry N=4 → actual_bot=nash_carry, nash_rule_model=carry。
    (c) new nash_carry N=5 超过上限 → 诚实回落 Heuristic + fallback_reason
        含 NASH_CARRY_MAX_N=4。
    (d) **关键契约**：N=3 R1 强制平局 (env.carry>0)，
        - Nash(CLASSIC) 下一回合 ai_policy.note 明确含 "fallback"+"carry"
          (诚实回落：模型不兼容，绝不输出假分布)
        - Nash(CARRY) 下一回合 ai_policy.bot_type == nash_carry 且 note 含
          "Nash-carry-over" tag（真正精确：carry>0 不回落）。"""

    # ------------------------------------------------------------------
    # [D1] /api/game/config：共 4 个 bot，nash_carry 带正确元数据
    # ------------------------------------------------------------------
    def test_config_exposes_nash_carry(self, client: TestClient):
        r = client.get("/api/game/config")
        assert r.status_code == 200
        body = r.json()
        bots = body["bots"]
        assert len(bots) == 4, f"expect 4 bot entries, got {len(bots)}: {bots}"
        nash_carry_cfg = next(
            (b for b in bots if b["id"] == BOT_NASH_CARRY), None
        )
        nash_cfg = next(
            (b for b in bots if b["id"] == BOT_NASH), None
        )
        assert nash_carry_cfg is not None, f"nash_carry missing from config: {bots}"
        # nash_rule_model 区分（两 exact 有不同 model；random/heuristic None）
        assert nash_cfg["nash_rule_model"] == "classic"
        assert nash_carry_cfg["nash_rule_model"] == "carry"
        # max_n 上限：nash 7 vs nash_carry 4
        assert nash_cfg["max_n_for_exact_nash"] == NASH_MAX_N, (
            f"classic nash max_n should be {NASH_MAX_N}, got "
            f"{nash_cfg['max_n_for_exact_nash']}"
        )
        assert nash_carry_cfg["max_n_for_exact_nash"] == NASH_CARRY_MAX_N, (
            f"carry nash max_n should be {NASH_CARRY_MAX_N}, got "
            f"{nash_carry_cfg['max_n_for_exact_nash']}"
        )

    # ------------------------------------------------------------------
    # [D2] new nash_carry N=4：actual_bot=nash_carry 未回落
    # ------------------------------------------------------------------
    def test_new_nash_carry_n4_runs_exact(self, client: TestClient):
        r = client.post(
            "/api/game/new",
            json={"num_cards": 4, "bot_type": BOT_NASH_CARRY},
        )
        assert r.status_code == 200
        body = r.json()
        meta = body["meta"]
        assert meta["requested_bot"] == BOT_NASH_CARRY
        assert meta["actual_bot"] == BOT_NASH_CARRY, (
            f"N=4 nash_carry must stay exact, actual={meta['actual_bot']} "
            f"fallback_reason={meta.get('fallback_reason')}"
        )
        assert meta["nash_rule_model"] == EXACT_MODE_CARRY
        assert meta.get("fallback_reason") is None

    # ------------------------------------------------------------------
    # [D3] new nash_carry N=5：超 NASH_CARRY_MAX_N=4 回落 Heuristic
    # ------------------------------------------------------------------
    def test_new_nash_carry_n5_fallback_to_heuristic(self, client: TestClient):
        r = client.post(
            "/api/game/new",
            json={"num_cards": 5, "bot_type": BOT_NASH_CARRY},
        )
        assert r.status_code == 200
        meta = r.json()["meta"]
        assert meta["requested_bot"] == BOT_NASH_CARRY
        assert meta["actual_bot"] == BOT_HEURISTIC, (
            f"N=5 nash_carry must fall back to heuristic, got "
            f"actual={meta['actual_bot']}"
        )
        # 回落原因必须明示是 NASH_CARRY_MAX_N=4 超阈值
        reason = meta.get("fallback_reason") or ""
        assert "NASH_CARRY_MAX_N" in reason or (
            f"N>{NASH_CARRY_MAX_N}" in reason
        ), (
            f"fallback_reason must mention NASH_CARRY_MAX_N threshold, "
            f"got '{reason}'"
        )
        # 已回落 → nash_rule_model 必须为 None（不能假装是 exact）
        assert meta.get("nash_rule_model") in (None, "null"), (
            f"fallback state must NOT carry an exact nash_rule_model, "
            f"got {meta.get('nash_rule_model')}"
        )

    # ------------------------------------------------------------------
    # [D4] **关键契约对照**：R1 平局 carry>0 后
    #      Nash(CLASSIC) → 下一回合诚实回落 + note 含 fallback
    #      Nash(CARRY)  → 下一回合仍精确 + bot_type=nash_carry + tag
    # ------------------------------------------------------------------
    def test_carry_over_split_contract_tie_then_round2(
        self, client: TestClient
    ):
        # ============================================================
        # Helper: 对给定 bot_type 强制造一个 R1 tie (carry>0) 的环境，
        # 返回 R2 第一次 /play 的 ai_policy dict。
        # ============================================================
        def _r2_policy_after_r1_tie(bot_type: str) -> Dict[str, Any]:
            r = client.post(
                "/api/game/new",
                json={"num_cards": 3, "bot_type": bot_type},
            )
            assert r.status_code == 200, f"new failed: {r.content}"
            # 直接用 app module 的全局对象注入一个“已 R1 tie” 的 env：
            #   奖品牌顺序 [2,1,3]；第 1 轮 prize=2，human出3 / bot出3 → 平局
            #   → 非末轮 carry_next = 2，进入 R2 时 env.carry_pool==2 > 0。
            class _FixedShuffle:
                def shuffle(self_, x):  # noqa: N805
                    x[:] = [2, 1, 3]
            new_env = GoofspielEnv(3)
            new_env._rng = _FixedShuffle()  # type: ignore[attr-defined]
            new_env.reset()
            # prize_order = [2,1,3];  round 1 prize = 2.
            # Both play same card 3 -> tie, non-final, carry rolls over.
            # NOTE: GoofspielEnv.step() takes a Dict[PlayerId, Action], not kw args.
            _ = new_env.step({0: 3, 1: 3})
            assert new_env.carry_pool == 2, (
                f"post-R1 tie carry_pool should be 2 (=R1 prize), "
                f"got {new_env.carry_pool}"
            )
            assert not new_env.done
            # 把 session 全局 env 替换成我们准备好的这个（bot 不变）
            app_module._env = new_env

            # 人类在 R2 随便出一张合法卡（比如 2）
            r2 = client.post("/api/game/play", json={"action": 2})
            assert r2.status_code == 200, f"R2 play failed: {r2.content}"
            return r2.json()["ai_policy"]

        # ---- 两种模式分别跑（共享同一个 TestClient，会自动 clean） ----
        # (a) CLASSIC 模式: R1 tie→carry>0 → NashBot 诚实回落
        pol_classic = _r2_policy_after_r1_tie(BOT_NASH)
        note_classic = pol_classic.get("note") or ""
        # fallback 痕迹：必须明确告知用户当前回合 *非* 精确 Nash（规则模型不兼容）
        assert "fallback" in note_classic.lower() or (
            "回落" in note_classic
        ), (
            f"CLASSIC Nash with carry>0 MUST honest-fallback, "
            f"but got note='{note_classic}' (no fallback tag)"
        )
        assert "carry" in note_classic.lower() or (
            "carry" in note_classic
        ), (
            f"fallback note must mention the cause 'carry', "
            f"got '{note_classic}'"
        )

        # (b) CARRY 模式：R1 tie→carry>0 → 仍精确，bot_type 保持 nash_carry
        pol_carry = _r2_policy_after_r1_tie(BOT_NASH_CARRY)
        note_carry = pol_carry.get("note") or ""
        # bot_type 必须还是 nash_carry（不能偷偷换成 heuristic）
        assert pol_carry.get("bot_type") == BOT_NASH_CARRY, (
            f"CARRY Nash with carry>0 must stay exact, but got "
            f"bot_type={pol_carry.get('bot_type')} note='{note_carry}'"
        )
        # note 必须携带 carry-over 模型标签（给前端展示"本轮是 carry 精确解"）
        assert "carry-over" in note_carry.lower().replace(" ", "-") or (
            "carry" in note_carry.lower() and "exact" in note_carry.lower()
        ), (
            f"CARRY mode R2 exact policy must be tagged with carry-over, "
            f"got note='{note_carry}'"
        )
        # 绝对不能出现 fallback 字样（证明 carry>0 下 *未* 回落）
        assert "fallback" not in note_carry.lower() and (
            "回落" not in note_carry
        ), (
            f"CARRY Nash R2 must NOT fallback, but note has fallback tag: "
            f"'{note_carry}'"
        )
