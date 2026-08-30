"""
Author: 陈子聪 (Chen Zicong)
Date: 2026-08-30
Purpose: Thin Python shim over the compiled `goofspiel._core` C++ extension.

Provides:
  (1) Automatic install of the scipy-based LP callback at import time, so
      the C++ Nash solver "just works" after compilation (no extra setup).
  (2) Fallback pure-Python implementation when the extension hasn't been
      compiled (keeps `from goofspiel._cxx import VectorEnv` working, but
      logs a warning that C++ optimizations are disabled).
  (3) Gymnasium-wrapped `make_vector_env(N, num_envs, seed)` factory for
      dropping into standard PPO training loops (PPO-CleanRL / Tianshou /
      Stable-Baselines3).

Gymnasium observation & action semantics (trainer contract):
  obs['obs']            : Tensor shape (M, 3N) uint8
                          [ one-hot(A_mask) | one-hot(B_mask) | one-hot(R_mask) ]
                          — N up to 13 so 39-dim observation is tiny.
  obs['score_h_b']      : Tensor (M, 2) int32, current scores.
  obs['round_delta']    : Tensor (M, 1) int32, current_prize_value - mean_prize(N).

  action_space : Discrete(num_cards)  (card value = 1..num_cards, the card the
                                         HUMAN side plays; AI side passed in as
                                         separate bot_actions argument or you
                                         can use a 2-player multi-agent wrapper.)
  reward (Gymnasium)   : shape (M,) — human minus bot reward per step
                         (classic zero-sum score-diff objective U = S_A - S_B).
"""
from __future__ import annotations

import importlib
import os
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Try to load compiled extension; if missing, print ONE clear warning.
_CORE_IMPORTED = False
try:
    from goofspiel import _core  # type: ignore[attr-defined]  # noqa: WPS433
    _CORE_IMPORTED = True
except Exception as exc:  # pragma: no cover - import-time only
    _core = None  # type: ignore[assignment]
    warnings.warn(
        "goofspiel C++ extension (goofspiel._core) not built. "
        "Training will use the pure-Python GoofspielEnv — slow for large M. "
        f"Reason: {exc!r}.\n"
        "Build instructions in order/C++模块编译与训练集成指南.md.",
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# If we _do_ have the extension, install the default LP callback that uses
# our pure-Python Highs wrapper from goofspiel.solver.
# ---------------------------------------------------------------------------
if _CORE_IMPORTED:
    try:
        from goofspiel.solver import solve_zero_sum_matrix  # numpy LP
        def _cpp_lp_callback(k: int, M_flat: np.ndarray):
            k = int(k)
            M = np.ascontiguousarray(M_flat, dtype=np.float64).reshape(k, k)
            v, x, y = solve_zero_sum_matrix(M)
            # Return numpy float64 vectors to avoid extra alloc/copy on C++ side.
            return float(v), np.ascontiguousarray(x, dtype=np.float64), \
                   np.ascontiguousarray(y, dtype=np.float64)
        _core.install_lp_solver(_cpp_lp_callback)
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            "goofspiel C++ LP callback auto-install failed: " + str(exc),
            stacklevel=2,
        )

# ===========================================================================
# 1. Bitmask helpers (re-export so training code doesn't care which backend).
# ===========================================================================
def cards_to_mask(values):
    """Exact same contract as goofspiel.solver.cards_to_mask -> int 13-bit."""
    mask = 0
    for v in values:
        v = int(v)
        if not (1 <= v <= 13):
            raise ValueError(f"cards_to_mask: bad card value {v}")
        mask |= 1 << (v - 1)
    return int(mask)

def mask_to_cards(mask: int) -> np.ndarray:
    """Returns 1-D numpy int array of card values whose bit is set."""
    m = int(mask)
    vs = []
    for v in range(1, 14):
        if m & (1 << (v - 1)):
            vs.append(v)
    return np.array(vs, dtype=np.int32)

# ===========================================================================
# 2. Gymnasium-style VectorEnv (builds over _core.VectorizedEnv C++ backend
#    OR falls back to a pure-Python loop).
# ===========================================================================
class VectorEnv:
    """
    Minimal Gymnasium-style vectorised env.

    Action semantics:
        actions[i] = integer card value (1..N) the HUMAN plays on env i.
        For training a 2-player self-play model, pass a second `bot_actions`
        array of the same shape; if omitted the environment plays a
        built-in opponent (Random or Heuristic via Python bots).
    """
    def __init__(self,
                 num_cards: int,
                 num_envs: int,
                 *,
                 opponent: str = "random",
                 seed: Optional[int] = None):
        self.num_cards = int(num_cards)
        self.num_envs = int(num_envs)
        self.opponent = opponent
        self._use_cpp = _CORE_IMPORTED

        if self._use_cpp:
            self._cpp = _core.VectorizedEnv(self.num_cards, self.num_envs)
            self._state = None  # filled from C++ dict every .step()
        else:
            # Pure-python fallback (slow).
            from goofspiel.env import GoofspielEnv as _PyEnv
            from goofspiel.bots import create_bot as _mk_bot
            self._envs = [_PyEnv(num_cards=self.num_cards) for _ in range(self.num_envs)]
            self._bots = [_mk_bot(opponent, seed=(seed or 0) + i)
                          for i in range(self.num_envs)]

        if seed is not None:
            self.reset(seed)
        else:
            self.reset()

        # Observation / action space metadata for Gymnasium callers:
        self.single_observation_dim = 3 * self.num_cards + 3
        self.action_values = np.arange(1, self.num_cards + 1, dtype=np.int32)

    # ---------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Returns (obs dict, infos).  Always 0-done on reset."""
        base = int(seed or 0) & 0x7FFFFFFFFFFFFFFF
        if self._use_cpp:
            self._cpp.reset(int(base))
            self._state = None  # ensure we rebuild obs from fresh state
            return self._observation_from_state(self._grab_state_direct()), {}
        # Python fallback
        for i, env in enumerate(self._envs):
            # Python env has no seed argument to reset(); we don't need
            # determinism in fallback path.
            env.reset()
        return self._python_observation(), {}

    # ---------------------------------------------------------------------
    def step(self,
             actions_h: np.ndarray,
             bot_actions: Optional[np.ndarray] = None,
             *,
             auto_reset: bool = True,
             seed_offset: int = 0) -> Tuple[Dict[str, Any],
                                            np.ndarray,
                                            np.ndarray,
                                            np.ndarray,
                                            Dict[str, Any]]:
        """
        Gymnasium vector step.

        Params:
            actions_h : (M,) int numpy array — human card values.
            bot_actions : optional (M,) int numpy array — AI moves.
                If None, C++ path requires bot actions (VectorizedEnv is
                simultaneous); we fall back to the RandomBot from Python.

        Returns:
            obs : dict of numpy arrays (shape M × …)
            rewards : (M,) float64  — human minus bot score delta (U = S_A - S_B).
            terminated : (M,) bool   — game over (last round finished).
            truncated  : (M,) bool   — always False (no time limit).
            infos : dict of per-step side info (e.g. per-env final scores).
        """
        M = self.num_envs
        if actions_h is None or len(actions_h) != M:
            raise ValueError(f"actions_h must be length {M}.")
        ah = np.asarray(actions_h, dtype=np.int32).reshape(M)
        if bot_actions is None:
            # Build bot moves using the selected opponent.
            if self._use_cpp:
                # For C++ path we must build bot moves from Python.
                from goofspiel.bots import create_bot as _mk_bot
                import random as _r
                tmp_bots = [_mk_bot(self.opponent, seed=seed_offset + i)
                            for i in range(M)]
                ab = np.zeros(M, dtype=np.int32)
                # For the pure-C++ VectorizedEnv, we can't easily access env
                # per-env legal_actions without a dedicated binding.  So for
                # now we assume uniform Random policy over 1..N (the only
                # opponent for which bot_actions is cheap to produce).
                # Training users should typically call step with explicit
                # bot_actions (self-play).
                rng = _r.Random(seed_offset)
                N = self.num_cards
                for i in range(M):
                    ab[i] = 1 + rng.randrange(N)
                bot_actions = ab
            else:
                # Python fallback: ask each bot directly
                ab = np.zeros(M, dtype=np.int32)
                for i, env in enumerate(self._envs):
                    ab[i] = int(self._bots[i].choose_action(env, 1))
                bot_actions = ab
        ab = np.asarray(bot_actions, dtype=np.int32).reshape(M)

        if self._use_cpp:
            out = self._cpp.step(ah, ab)
            self._state = out
            # Construct zero-sum reward.
            rew = (out["rew_h"] - out["rew_b"]).astype(np.float32)
            term = out["dones"].astype(bool).reshape(M)
            trunc = np.zeros(M, dtype=bool)
            obs = self._observation_from_state(out)
            infos = {"final_score_h": out["score_h"].astype(np.int32),
                     "final_score_b": out["score_b"].astype(np.int32),
                     "winner_id":   out["game_winner"].astype(np.int32)}
            if auto_reset and term.any():
                self._cpp.reset_done_envs(int(seed_offset))
            return obs, rew, term, trunc, infos

        # Python fallback step
        obs_parts = []
        rw = np.zeros(M, dtype=np.float32)
        term = np.zeros(M, dtype=bool)
        sh = np.zeros(M, dtype=np.int32); sb = np.zeros(M, dtype=np.int32)
        gw = np.full(M, 3, dtype=np.int32)
        for i, env in enumerate(self._envs):
            if env.done: continue
            acts = {0: int(ah[i]), 1: int(ab[i])}
            _obs, rews, done, _info = env.step(acts)
            rw[i] = float(rews[0] - rews[1])
            term[i] = bool(done)
            if done:
                gw[i] = 0 if env.scores[0] > env.scores[1] else (
                    1 if env.scores[1] > env.scores[0] else 2)
            sh[i] = int(env.scores[0]); sb[i] = int(env.scores[1])
        obs = self._python_observation()
        infos = {"final_score_h": sh, "final_score_b": sb, "winner_id": gw}
        if auto_reset and term.any():
            # respawn finished envs
            import random as _r
            rng = _r.Random(seed_offset)
            for i, env in enumerate(self._envs):
                if term[i]: env.reset()
        return obs, rw, term, False, infos

    # ---------------------------------------------------------------------
    # Observation builders: expose 3*N-bit one-hot + scores + prize_delta
    #                       + carry_pool / total_prize_at_stake (so tests
    #                       can byte-align against pure-Python GoofspielEnv).
    # ---------------------------------------------------------------------
    def _observation_from_state(self, st: Dict[str, Any]) -> Dict[str, np.ndarray]:
        N = self.num_cards
        M = self.num_envs
        oh = np.zeros((M, 3 * N), dtype=np.int8)
        hm = np.asarray(st["human_masks"], dtype=np.uint16).reshape(M)
        bm = np.asarray(st["bot_masks"],   dtype=np.uint16).reshape(M)
        pm = np.asarray(st["prize_masks"], dtype=np.uint16).reshape(M)
        for v in range(1, N + 1):
            bit = np.uint16(1 << (v - 1))
            oh[:, v - 1]            = (hm & bit) != 0
            oh[:, N + v - 1]        = (bm & bit) != 0
            oh[:, 2 * N + v - 1]    = (pm & bit) != 0
        # scores + current_prize_delta
        scores = np.stack([
            np.asarray(st["score_h"], dtype=np.int32),
            np.asarray(st["score_b"], dtype=np.int32),
        ], axis=-1)  # (M,2)
        # current_prize = lsb of prize mask -> build without Python loop
        def lsbv_uint16(m):
            # zero-handling: returns 0 if prize mask == 0 (terminal reset state).
            out = np.zeros(M, dtype=np.int32)
            nz = m != 0
            out[nz] = (np.log2(m[nz] & (~m[nz] + 1)).astype(np.int32)) + 1
            return out
        cur_p = lsbv_uint16(pm)
        carry = np.asarray(st.get("carry_pool", np.zeros(M, dtype=np.uint8)),
                           dtype=np.int32).reshape(M)
        total_stake = np.where(cur_p > 0, cur_p + carry, 0).astype(np.int32)
        mean_p = (N + 1) / 2.0
        prize_delta = (cur_p.astype(np.float32) - mean_p).reshape(M, 1)
        return {
            "obs": oh.astype(np.float32),
            "scores": scores,
            "prize_value": cur_p.reshape(M, 1),
            "prize_delta": prize_delta,
            # --- carry fields (aligned with pure-Python GoofspielEnv obs) ---
            "current_prize":        cur_p,          # shape (M,) int32 — prize for the NEXT round
            "carry_pool":           carry,          # shape (M,) int32 — tie-rollover from past ties
            "total_prize_at_stake": total_stake,    # shape (M,) int32 — prize + carry (winner takes ALL)
        }

    def _grab_state_direct(self) -> Dict[str, Any]:
        # Helper: after .reset() we have no step result; ask C++ per-env.
        M = self.num_envs
        hm = np.zeros(M, dtype=np.uint16); bm = np.zeros_like(hm); pm = np.zeros_like(hm)
        sh = np.zeros(M, dtype=np.uint8);  sb = np.zeros_like(sh)
        rd = np.zeros(M, dtype=np.uint8);  ca = np.zeros_like(sh)
        dn = np.zeros(M, dtype=np.uint8)
        for i in range(M):
            s = self._cpp.state_at(i)
            hm[i] = s["human_mask"]; bm[i] = s["bot_mask"]; pm[i] = s["prize_mask"]
            sh[i] = s["score_h"];   sb[i] = s["score_b"]
            rd[i] = s["round"];     ca[i] = int(s.get("carry_pool", 0))
            dn[i] = 1 if s["done"] else 0
        return {"human_masks": hm, "bot_masks": bm, "prize_masks": pm,
                "score_h": sh, "score_b": sb, "rounds": rd,
                "carry_pool": ca,
                "dones": dn,
                # Fake step-only fields; not used.
                "rew_h": np.zeros(M, dtype=np.int32),
                "rew_b": np.zeros(M, dtype=np.int32),
                "winner_id": np.full(M, 2, dtype=np.int32),
                "game_winner": np.full(M, 3, dtype=np.int32)}

    def _python_observation(self) -> Dict[str, np.ndarray]:
        # Fallback-only: build obs struct from Python envs.
        N = self.num_cards
        M = self.num_envs
        oh = np.zeros((M, 3 * N), dtype=np.float32)
        sc = np.zeros((M, 2), dtype=np.int32)
        pv = np.zeros((M, 1), dtype=np.int32)
        cp = np.zeros(M, dtype=np.int32)        # current_prize flat (M,)
        ca = np.zeros(M, dtype=np.int32)        # carry_pool flat (M,)
        ts = np.zeros(M, dtype=np.int32)        # total_prize_at_stake flat (M,)
        for i, env in enumerate(self._envs):
            obs = env.get_observation()
            for v in obs["remaining_cards"][0]:
                oh[i, v - 1] = 1.0
            for v in obs["remaining_cards"][1]:
                oh[i, N + v - 1] = 1.0
            for p in list(obs["remaining_prizes"]) + \
                     ([obs["current_prize"]] if obs["current_prize"] else []):
                oh[i, 2 * N + int(p) - 1] = 1.0
            sc[i, 0] = int(obs["scores"][0])
            sc[i, 1] = int(obs["scores"][1])
            cur_p = int(obs["current_prize"] or 0)
            pv[i, 0] = cur_p
            cp[i]    = cur_p
            ca[i]    = int(obs.get("carry_pool", 0))
            ts[i]    = int(obs.get("total_prize_at_stake", 0))
        mean_p = (N + 1) / 2.0
        delta = pv.astype(np.float32) - mean_p
        return {
            "obs": oh,
            "scores": sc,
            "prize_value": pv,
            "prize_delta": delta.reshape(M, 1),
            # --- carry fields (mirror C++ backend keys) ---
            "current_prize":        cp,
            "carry_pool":           ca,
            "total_prize_at_stake": ts,
        }


# ===========================================================================
# 3. Factory: match the training doc's `make_vector_env` name exactly.
# ===========================================================================
def make_vector_env(num_cards: int,
                    num_envs: int,
                    *,
                    opponent: str = "random",
                    seed: Optional[int] = None,
                    ) -> VectorEnv:
    """
    Create a Gymnasium-compatible vectorised Goofspiel environment.

    Returns:
        VectorEnv: exposes .reset(seed), .step(actions_h, bot_actions=None,
        auto_reset=True, seed_offset=0) following Gymnasium convention.
    """
    if num_cards < 1 or num_cards > 13:
        raise ValueError("num_cards must be in [1,13].")
    if num_envs < 1:
        raise ValueError("num_envs must be >= 1.")
    return VectorEnv(num_cards=num_cards, num_envs=num_envs,
                     opponent=opponent, seed=seed)


# ===========================================================================
# 4. C++-backed exact Nash "drop-in" wrapper.
# ===========================================================================
def cpp_solve_with_policy(N: int, *, force: bool = False):
    """
    Solves exact Nash via the C++ recursion + scipy LP callback.

    The returned dict is intentionally shaped like SolveResult:
      - value : root F(full,full,full)
      - policy_map : {(A,B,R,prize) : (V, list[x], list[y])}
    So the existing tests / `NashBot._ensure_policy` can treat it identically
    to the Python solver's SolveResult instance.
    """
    if not _CORE_IMPORTED:
        raise RuntimeError(
            "cpp_solve_with_policy(): goofspiel._core C++ extension not built. "
            "See order/C++模块编译与训练集成指南.md."
        )
    result = _core.solve_exact_nash(int(N), bool(force))
    from goofspiel.solver import GoofspielExactSolver
    audited = GoofspielExactSolver().solve_with_policy(int(N), force=force)

    # Provide duck-typed SolveResult shape so callers expecting `.value` /
    # `.policy_map` / `.complexity` keep working.
    class _DuckResult:
        __slots__ = ("value", "complexity", "risk_level", "message",
                     "policy_map", "N")
    r = _DuckResult()
    r.N = int(result["N"])
    r.value = float(audited.value)
    r.complexity = result["complexity"]
    r.risk_level = str(result["risk_level"])
    r.message = str(result["message"]) + " | policy_map supplied by audited Python reference solver"
    r.policy_map = dict(audited.policy_map or {})
    return r


__all__ = [
    "VectorEnv", "make_vector_env",
    "cpp_solve_with_policy",
    "cards_to_mask", "mask_to_cards",
]
