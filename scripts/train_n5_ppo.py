"""
Author: 陈子聪 (Chen Zicong)
Date: 2026-08-30
Purpose: Minimal end-to-end PPO training demo for Goofspiel on top of the
         C++-accelerated `make_vector_env` Gymnasium-like vector env.

Goal — PROOF OF PRINCIPLE ONLY:
    - Show the C++ backend (4096 parallel envs) can feed a small PPO model
      directly, with no extra marshalling beyond `torch.from_numpy()`.
    - Print a learning curve every 5 updates (episode returns, policy loss,
      value loss, entropy).
    - N=5 5-card variant, 100k environment interactions (~10 updates at 4096
      rollout steps / 1 update).  This is a *demo*; production config lives
      in order/Goofspiel-13 智能体完整训练流程实施规范.md.

Run:
    python scripts/train_n5_ppo.py
        [--num-envs 4096]
        [--total-timesteps 100000]
        [--lr 2.5e-4]
        [--seed 1]

If goofspiel._core C++ extension isn't built, the script falls back to the
pure-Python vector env (prints a warning and uses num_envs=256 default).
"""
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from goofspiel._cxx import make_vector_env  # C++ 或 fallback Python env

# --------- tiny PPO network (1M params max) --------------------------------
class ActorCriticNet(nn.Module):
    """
    Inputs -> concat(obs_flat:3N, scores/2, prize_delta/1).
    Two hidden layers of 256 units each, tanh.
    Actor logits: num_cards outputs (masked by 0 where card already played).
    Value scalar output.
    """
    def __init__(self, obs_dim: int, num_actions: int,
                 hidden: int = 256):
        super().__init__()
        self.num_actions = int(num_actions)
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),   nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden, self.num_actions)
        self.critic_head = nn.Linear(hidden, 1)

    def _featurise(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        flat = obs["obs"].flatten(start_dim=1)          # (B, 3N)
        sc   = obs["scores"].flatten(start_dim=1)        # (B, 2)
        pd   = obs["prize_delta"].flatten(start_dim=1)   # (B, 1)
        return torch.cat([flat, sc, pd], dim=1).float()

    def get_value(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.critic_head(self.trunk(self._featurise(obs))).squeeze(-1)

    def get_action_and_value(self, obs: Dict[str, torch.Tensor],
                             actions: torch.Tensor | None = None):
        feats = self.trunk(self._featurise(obs))
        logits = self.actor_head(feats)
        # Apply legal-action mask: cards already played get -1e8 logits so
        # their probability becomes exactly 0.  The legal mask is derived
        # directly from obs["obs"][:, 0:num_actions] (human mask one-hot).
        legal = obs["obs"][:, : self.num_actions] > 0.5
        INF = 1e8
        logits = logits.masked_fill(~legal, -INF)
        dist = Categorical(logits=logits)
        if actions is None:
            actions = dist.sample()
        # Card values (1-based) vs action index (0-based): return separate
        # array of card values for the env step.
        card_values = actions.int() + 1
        return (
            actions,
            card_values,
            dist.log_prob(actions),
            dist.entropy(),
            self.critic_head(feats).squeeze(-1),
        )


# --------- storage: a rollout buffer of fixed length T over M envs ---------
class RolloutBuffer:
    def __init__(self, T: int, M: int, num_cards: int, device):
        self.T, self.M, self.N = T, M, num_cards
        obs_dim = 3 * num_cards
        self.obs = torch.zeros((T, M, obs_dim), dtype=torch.float32, device=device)
        self.scores = torch.zeros((T, M, 2), dtype=torch.int32, device=device)
        self.prize_delta = torch.zeros((T, M, 1), dtype=torch.float32, device=device)
        self.actions = torch.zeros((T, M), dtype=torch.int64, device=device)
        self.logprobs = torch.zeros((T, M), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((T, M), dtype=torch.float32, device=device)
        self.dones = torch.zeros((T, M), dtype=torch.bool, device=device)
        self.values = torch.zeros((T, M), dtype=torch.float32, device=device)
        # For legal mask during policy update we need the per-step legal mask,
        # keep it as a separate tensor:
        self.legal = torch.zeros((T, M, num_cards), dtype=torch.bool, device=device)

    @staticmethod
    def _to(d, k, device):
        if isinstance(d[k], np.ndarray):
            return torch.from_numpy(d[k]).to(device)
        return d[k].to(device)

    def assign(self, t: int, obs_dict, actions_idx, logprobs, rewards,
               dones, values):
        device = self.obs.device
        self.obs[t] = self._to(obs_dict, "obs", device).float()
        self.scores[t] = self._to(obs_dict, "scores", device).int()
        self.prize_delta[t] = self._to(obs_dict, "prize_delta", device).float()
        self.legal[t] = self.obs[t][:, : self.N] > 0.5
        self.actions[t] = actions_idx.to(device)
        self.logprobs[t] = logprobs.to(device)
        self.rewards[t] = torch.as_tensor(rewards, device=device).float()
        self.dones[t] = torch.as_tensor(dones, device=device).bool()
        self.values[t] = values.to(device)

    def obs_at(self, t):
        return {
            "obs": self.obs[t],
            "scores": self.scores[t].float(),
            "prize_delta": self.prize_delta[t],
        }


# --------- minimal PPO update (CleanRL inspired 60 lines) ------------------
def ppo_update(
    net: ActorCriticNet,
    opt: optim.Optimizer,
    buf: RolloutBuffer,
    next_value: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    n_epochs: int,
    n_minibatch: int,
    max_grad_norm: float,
):
    T, M, N = buf.T, buf.M, buf.N
    # GAE
    advantages = torch.zeros_like(buf.rewards)
    lastgaelam = 0
    nextnonterminal = (~buf.dones[T - 1]).float()
    nextvalues = next_value
    for t in reversed(range(T)):
        if t == T - 1:
            nextnonterm = nextnonterminal; nextv = nextvalues
        else:
            nextnonterm = (~buf.dones[t + 1]).float()
            nextv = buf.values[t + 1]
        delta = buf.rewards[t] + gamma * nextv * nextnonterm - buf.values[t]
        advantages[t] = lastgaelam = \
            delta + gamma * gae_lambda * nextnonterm * lastgaelam
    returns = advantages + buf.values

    # Flatten
    b_obs    = {"obs": buf.obs.reshape(T*M, 3*N),
                "scores": buf.scores.reshape(T*M, 2).float(),
                "prize_delta": buf.prize_delta.reshape(T*M, 1)}
    b_actions = buf.actions.reshape(-1)
    b_logp    = buf.logprobs.reshape(-1)
    b_adv     = advantages.reshape(-1)
    b_ret     = returns.reshape(-1)
    b_vals    = buf.values.reshape(-1)
    b_legal   = buf.legal.reshape(T*M, N)
    batch_size = T * M
    micro = batch_size // n_minibatch
    assert micro > 0
    # Normalize advantage (per batch)
    b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

    # Track losses for stdout logging
    losses_pg, losses_v, losses_e = [], [], []
    for _ in range(n_epochs):
        idx = torch.randperm(batch_size, device=b_obs["obs"].device)
        for start in range(0, batch_size, micro):
            mb = idx[start:start + micro]
            mb_obs = {k: v[mb] for k, v in b_obs.items()}
            mb_acts = b_actions[mb]
            mb_logp_old = b_logp[mb]
            mb_adv  = b_adv[mb]
            mb_ret  = b_ret[mb]
            mb_vals = b_vals[mb]
            mb_legal = b_legal[mb]
            # Forward
            feats = net.trunk(torch.cat([mb_obs["obs"], mb_obs["scores"], mb_obs["prize_delta"]], dim=1))
            logits = net.actor_head(feats)
            logits = logits.masked_fill(~mb_legal, -1e8)
            dist = Categorical(logits=logits)
            new_logp = dist.log_prob(mb_acts)
            entropy = dist.entropy().mean()
            new_value = net.critic_head(feats).squeeze(-1)

            logr = new_logp - mb_logp_old
            ratio = logr.exp()
            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            v_loss_unclipped = (new_value - mb_ret) ** 2
            v_clipped = mb_vals + torch.clamp(new_value - mb_vals, -clip_coef, clip_coef)
            v_loss_clipped = (v_clipped - mb_ret) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            loss = pg_loss + vf_coef * v_loss - ent_coef * entropy

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            opt.step()

            losses_pg.append(float(pg_loss.item()))
            losses_v.append(float(v_loss.item()))
            losses_e.append(float(entropy.item()))
    return (
        float(np.mean(losses_pg)),
        float(np.mean(losses_v)),
        float(np.mean(losses_e)),
    )


# ==========================================================================
# main
# ==========================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-cards", type=int, default=5, help="1..13")
    p.add_argument("--num-envs",  type=int, default=None,
                   help="parallel envs.  Default 4096 if C++ backend is built, else 256.")
    p.add_argument("--total-timesteps", type=int, default=100_000)
    p.add_argument("--rollout-steps", type=int, default=256,
                   help="T steps per env per PPO update.")
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=8)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="auto",
                   help="auto / cpu / cuda / cuda:0")
    p.add_argument("--opponent", type=str, default="random",
                   help="opponent used when bot_actions is not provided.  This "
                        "demo uses uniform self-play: both human & bot use the "
                        "current policy, so the --opponent flag is unused unless "
                        "no C++ backend is built.")
    p.add_argument("--log-dir", type=str, default=None,
                   help="Optional directory for tensorboard-style logs (not used).")
    args = p.parse_args()

    # --- reproducibility ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- device ---
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    # --- env ---
    from goofspiel import _core  # noqa: F401 (force import to print warning once)
    cpp_built = False
    try:
        cpp_built = _core is not None and hasattr(_core, "VectorizedEnv")
    except Exception:
        cpp_built = False
    if args.num_envs is None:
        args.num_envs = 4096 if cpp_built else 256
    venv = make_vector_env(args.num_cards, args.num_envs,
                           opponent=args.opponent, seed=args.seed)
    print(f"[goof] env N={args.num_cards} | M={args.num_envs} | "
          f"C++ backend = {cpp_built} | device = {device}")

    # --- model ---
    obs_dim = 3 * args.num_cards + 3  # 3N one-hot + 2 scores + prize_delta
    net = ActorCriticNet(obs_dim=obs_dim, num_actions=args.num_cards).to(device)
    opt = optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    T = int(args.rollout_steps)
    buf = RolloutBuffer(T, args.num_envs, args.num_cards, device=device)

    # Obs init -> torch dict
    np_obs, _ = venv.reset(seed=args.seed)
    cur_obs = {k: torch.from_numpy(v.copy()).to(device) for k, v in np_obs.items()}

    global_step = 0
    n_updates = args.total_timesteps // (args.num_envs * T)
    if n_updates <= 0:
        n_updates = 1
    print(f"[goof] total timesteps target {args.total_timesteps}, "
          f"so {n_updates} PPO updates × {T}×{args.num_envs} rollouts")
    t0 = time.perf_counter()

    episode_returns: List[float] = []
    episode_lengths: List[int] = []

    for update_idx in range(n_updates):
        # ----- rollout -----
        for t in range(T):
            with torch.no_grad():
                act_idx, card_vals, logp, ent, val = net.get_action_and_value(cur_obs)
                # Self-play: bot side uses the SAME policy (with its own mask =
                # the bot mask = obs[ N : 2N ]  one-hot row).
                bot_obs = {
                    "obs": torch.cat([
                        cur_obs["obs"][:, args.num_cards:2 * args.num_cards],
                        cur_obs["obs"][:, 0:args.num_cards],
                        cur_obs["obs"][:, 2 * args.num_cards:],
                    ], dim=1),
                    "scores": cur_obs["scores"][:, [1, 0]].clone(),
                    "prize_delta": cur_obs["prize_delta"].clone(),
                }
                _b_idx, bot_cards, _blp, _be, _bv = net.get_action_and_value(bot_obs)

            np_step_obs, rew, term, _trunc, infos = venv.step(
                actions_h=card_vals.cpu().numpy().astype(np.int32),
                bot_actions=bot_cards.cpu().numpy().astype(np.int32),
                auto_reset=True,
                seed_offset=update_idx * T + t,
            )
            global_step += args.num_envs

            buf.assign(t, cur_obs, act_idx.detach(), logp.detach(),
                       rew, term, val.detach())

            cur_obs = {k: torch.from_numpy(v.copy()).to(device)
                       for k, v in np_step_obs.items()}
            # Track finished episodes for log
            done_idx = np.flatnonzero(term)
            for i in done_idx:
                sh = int(infos["final_score_h"][i])
                sb = int(infos["final_score_b"][i])
                episode_returns.append(float(sh - sb))
                episode_lengths.append(args.num_cards)  # episodes always N steps

        # --- bootstrap value ---
        with torch.no_grad():
            next_val = net.get_value(cur_obs)

        # --- ppo update ---
        pg_l, v_l, ent_l = ppo_update(
            net, opt, buf, next_val,
            gamma=args.gamma, gae_lambda=args.gae_lambda,
            clip_coef=args.clip_coef, vf_coef=args.vf_coef,
            ent_coef=args.ent_coef, n_epochs=args.update_epochs,
            n_minibatch=args.num_minibatches,
            max_grad_norm=args.max_grad_norm,
        )

        # --- log ---
        sps = int(global_step / max(1e-6, time.perf_counter() - t0))
        lastN = min(200, len(episode_returns))
        avg_ret = float(np.mean(episode_returns[-lastN:])) if lastN else 0.0
        print(
            f"[upd {update_idx+1:>3}/{n_updates}]  steps={global_step:>7d}  "
            f"SPS={sps:>5d}  |  avg_return(±200)={avg_ret:+.3f}  "
            f"|  pg_loss={pg_l:+.4f} v_loss={v_l:+.4f} entropy={ent_l:+.4f}"
        )

    # --- final save (lightweight: just state_dict to ./checkpoints/ if dir exists) ---
    save_dir = Path(os.environ.get("GOOFSPIEL_CHECKPOINT_DIR", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"ppo_n{args.num_cards}_seed{args.seed}.pt"
    torch.save({
        "args": vars(args),
        "model_state_dict": net.state_dict(),
        "optim_state_dict": opt.state_dict(),
        "global_step": global_step,
        "last_ep_return_avg": (
            float(np.mean(episode_returns[-200:])) if episode_returns else 0.0
        ),
    }, path)
    print(f"\n[goof] done. last 200-ep avg return = "
          f"{float(np.mean(episode_returns[-200:])) if episode_returns else 0.0:+.3f}")
    print(f"[goof] checkpoint saved to {path}")


if __name__ == "__main__":
    main()
