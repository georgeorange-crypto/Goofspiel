"""Self-play trajectory generation for robust training."""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass

from goofspiel.game import GameState, transition
from goofspiel.training.buffers import ReplayItem
from goofspiel.training.data import RobustTrajectorySample, RoundRecord, state_record_from_game_state
from goofspiel.training.teachers import immediate_q_matrix


@dataclass
class SelfPlayBatch:
    trajectories: list[RobustTrajectorySample]
    replay_items: list[ReplayItem]
    games: int
    mean_score_diff: float


def _next_prize_after(state: GameState) -> int | None:
    if state.prize_mask == 0:
        return None
    for card in range(1, state.n + 1):
        if state.prize_mask & (1 << (card - 1)):
            return card
    return None


def _sample_action_from_policy(policy, legal: list[int], rng: random.Random) -> int:
    probs = [max(0.0, float(policy[card - 1])) for card in legal]
    total = sum(probs)
    if total <= 0:
        return rng.choice(legal)
    threshold = rng.random() * total
    accum = 0.0
    for card, prob in zip(legal, probs):
        accum += prob
        if accum >= threshold:
            return card
    return legal[-1]


def generate_selfplay_batch(
    model,
    *,
    games: int,
    n_cards: int,
    device: str,
    seed: int,
    model_version: str,
) -> SelfPlayBatch:
    import torch
    from goofspiel.models import public_state_from_game

    rng = random.Random(seed)
    trajectories: list[RobustTrajectorySample] = []
    replay_items: list[ReplayItem] = []
    score_diffs: list[int] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(int(games)):
            state = GameState.initial(n_cards, current_prize=1)
            states = []
            rounds = []
            self_policies = []
            opp_policies = []
            self_probs = []
            opp_probs = []
            while not state.done:
                batch = public_state_from_game([state], device=device)
                out = model(batch)
                legal_self = state.self_actions
                legal_opp = state.opponent_actions
                pi_self = torch.softmax(out.robust_policy_logits[0].masked_fill(~batch.self_action_mask[0], -1e9), dim=-1).detach().cpu()
                pi_opp = torch.softmax((-out.robust_policy_logits[0]).masked_fill(~batch.opponent_action_mask[0], -1e9), dim=-1).detach().cpu()
                a = _sample_action_from_policy(pi_self, legal_self, rng)
                b = _sample_action_from_policy(pi_opp, legal_opp, rng)
                before = state
                after = transition(state, a, b, next_prize=_next_prize_after(state))
                states.append(state_record_from_game_state(before))
                rounds.append(
                    RoundRecord(
                        round_index=before.round_index,
                        prize=before.current_prize,
                        self_action=a,
                        opponent_action=b,
                        reward_self=after.reward_self,
                        reward_opponent=after.reward_opp,
                        carry_in=before.carry_pool,
                        carry_out=after.state.carry_pool,
                        done=after.state.done,
                    )
                )
                self_policies.append([float(x) for x in pi_self.tolist()])
                opp_policies.append([float(x) for x in pi_opp.tolist()])
                self_probs.append(float(pi_self[a - 1]))
                opp_probs.append(float(pi_opp[b - 1]))
                q_target, _self_cards, _opp_cards = immediate_q_matrix(before)
                replay_items.append(
                    ReplayItem(
                        item_id=uuid.uuid4().hex,
                        state=state_record_from_game_state(before).__dict__,
                        q_target=q_target.tolist(),
                        policy_target=[float(x) for x in pi_self.tolist()],
                        final_score_diff=0.0,
                        priority=1.0 + abs(after.normalized_reward),
                        source="selfplay",
                    )
                )
                state = after.state
            diff = int(state.self_score - state.opp_score)
            score_diffs.append(diff)
            for item in replay_items[-len(rounds) :]:
                item.final_score_diff = float(diff)
            trajectories.append(
                RobustTrajectorySample(
                    sample_id=uuid.uuid4().hex,
                    states=states,
                    rounds=rounds,
                    behavior_policy_self=self_policies,
                    behavior_policy_opponent=opp_policies,
                    action_prob_self=self_probs,
                    action_prob_opponent=opp_probs,
                    final_score_diff=diff,
                    model_version=model_version,
                    opponent_version=model_version,
                    n=n_cards,
                )
            )
    if was_training:
        model.train()
    mean_diff = sum(score_diffs) / max(1, len(score_diffs))
    return SelfPlayBatch(trajectories, replay_items, int(games), float(mean_diff))
