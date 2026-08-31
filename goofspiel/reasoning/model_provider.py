"""Trained-model provider for the reasoning router (Phase 4.1 / 4.2).

Before this the reasoning layer and the neural net were *adjacent, not
integrated*: every Q the router used came from the handcrafted
``immediate_q_matrix``.  This provider is the one seam that lets a loaded
:class:`~goofspiel.models.GoofspielModel` supply

* the **robust** joint-Q matrix (opponent-agnostic — computed from the public
  state alone, exactly as the model computes ``q_robust`` before any opponent
  history is encoded), and
* an **opponent belief** over the opponent's next action (opponent-conditioned —
  computed from the trained opponent head, run on the adaptive view's memory /
  history).

Both are optional: when no provider is attached, or the model is out of budget,
the router falls back to the handcrafted matrix.  The robust path here **never**
reads opponent history — it is fed only the public state, so ``Q_R ⊥ opponent``
holds for the model just as it does for the handcrafted matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from goofspiel.game import GameState
from goofspiel.reasoning.state import ReasoningState


class TrainedModelProvider:
    """Adapts a loaded ``GoofspielModel`` to the router's Q / belief seams."""

    def __init__(self, model: Any, *, device: str = "cpu", max_cards: int | None = None) -> None:
        self.model = model
        self.device = device
        self.max_cards = int(max_cards if max_cards is not None else getattr(model, "max_cards", 13))
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, device: str = "cpu") -> "TrainedModelProvider":
        from goofspiel.training.model_eval import load_model_from_checkpoint

        model, _metadata = load_model_from_checkpoint(path, device=device)
        return cls(model, device=device)

    # ------------------------------------------------------------------
    # Robust Q — opponent-agnostic (public state only, no history/memory).
    # ------------------------------------------------------------------
    def robust_q13(self, state: GameState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(q[1,13,13], self_mask[1,13], opp_mask[1,13])`` from the model.

        The model is run on the public state alone (no ``current_game_history``,
        no ``long_term_memory``), so ``q_robust`` cannot depend on the opponent.
        Illegal joint cells are explicitly zeroed to match the handcrafted
        ``_immediate_q13`` convention (the downstream Nash solver masks them
        regardless, but zeroing keeps the two Q sources byte-comparable).
        """
        from goofspiel.models import public_state_from_game

        batch = public_state_from_game([state], max_cards=self.max_cards, device=self.device)
        with torch.no_grad():
            out = self.model(batch)
        q = out.q_robust.detach().float()  # (1, 13, 13)
        self_mask = out.self_action_mask.detach().bool()  # (1, 13)
        opp_mask = out.opponent_action_mask.detach().bool()  # (1, 13)
        joint = self_mask[:, :, None] & opp_mask[:, None, :]
        q = q * joint.to(q.dtype)
        return q, self_mask, opp_mask

    # ------------------------------------------------------------------
    # Opponent belief — opponent-conditioned (adaptive view only).
    # ------------------------------------------------------------------
    def opponent_belief(self, adaptive_state: ReasoningState) -> list[float] | None:
        """Return a length-13 belief over the opponent's next action, or ``None``.

        Uses the trained opponent head (``opponent_fused_logits``) run on the
        adaptive view's session memory / current-game history.  Returns ``None``
        when there is no memory/history to condition on (the router then leaves
        the adaptive tier idle rather than inventing a belief).
        """
        memory = adaptive_state.opponent_memory
        history = adaptive_state.current_game_history
        if memory is None and history is None:
            return None
        from goofspiel.models import public_state_from_game

        state = adaptive_state.public_state
        batch = public_state_from_game([state], max_cards=self.max_cards, device=self.device)
        with torch.no_grad():
            out = self.model(batch, current_game_history=history, long_term_memory=memory)
        logits = out.opponent_fused_logits[0].detach().float()  # (13,), illegal = -1e9
        opp_mask = out.opponent_action_mask[0].detach().bool()
        masked = logits.masked_fill(~opp_mask, float("-inf"))
        probs = torch.softmax(masked, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        return [float(p) for p in probs]


__all__ = ["TrainedModelProvider"]
