"""Hard-rule tool router for Goofspiel decisions."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from goofspiel.game import GameState
from goofspiel.reasoning.decision import FinalDecision, final_decision
from goofspiel.reasoning.exact_br import solve_exact_best_response
from goofspiel.reasoning.exact_tool import solve_exact_tool
from goofspiel.reasoning.matrix_tools import solve_matrix_nash_tool
from goofspiel.reasoning.search import SearchBudget, run_gt_cfr, run_sm_mcts
from goofspiel.reasoning.state import ReasoningState
from goofspiel.reasoning.types import GameToolResult, ToolMode


@dataclass(frozen=True)
class DecisionBudget:
    max_wall_ms: int = 200
    exact_max_remaining: int = 4
    sm_mcts_low: int = 128
    sm_mcts_mid: int = 512
    sm_mcts_high: int = 2048
    gt_cfr_iterations: int = 256
    # Robust floor slack for the safe adaptive mixture: the mixed policy must
    # keep its worst-case value within ``epsilon`` of the pure-robust guarantee.
    exploit_epsilon: float = 0.0


@dataclass
class AgentReasoningResult:
    robust_result: GameToolResult
    tool_results: list[GameToolResult]
    final: FinalDecision
    traces: list[dict[str, object]] = field(default_factory=list)


def _immediate_q13(state: GameState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from goofspiel.training.teachers import immediate_q_matrix

    q = torch.zeros(1, 13, 13)
    q_small, self_cards, opp_cards = immediate_q_matrix(state)
    q_small_t = torch.as_tensor(q_small, dtype=torch.float32)
    for i, a in enumerate(self_cards):
        for j, b in enumerate(opp_cards):
            q[0, a - 1, b - 1] = q_small_t[i, j]
    self_mask = torch.zeros(1, 13, dtype=torch.bool)
    opp_mask = torch.zeros(1, 13, dtype=torch.bool)
    self_mask[0, [a - 1 for a in state.self_actions]] = True
    opp_mask[0, [a - 1 for a in state.opponent_actions]] = True
    return q, self_mask, opp_mask


def strategic_importance(state: GameState) -> float:
    future_mass = sum(state.opponent_actions) / max(1, state.total_prize_mass)
    current = state.stake / max(1, state.total_prize_mass)
    early = 1.0 - (state.round_index - 1) / max(1, state.n)
    return float(min(1.0, 0.5 * current + 0.25 * future_mass + 0.25 * early))


class ToolRouter:
    """Implements the fixed router order from the design documents."""

    def __init__(self, budget: DecisionBudget | None = None, *, model_provider: object | None = None) -> None:
        self.budget = budget or DecisionBudget()
        # Optional trained-model provider (Phase 4.1).  When present, the robust
        # Q/masks come from the model; when absent the handcrafted immediate
        # matrix is used.  It must never be fed opponent history for the robust
        # path — it is handed only the opponent-agnostic public state.
        self.model_provider = model_provider

    def _robust_q13(self, state: GameState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Robust joint-Q from the model provider if available, else handcrafted."""
        if self.model_provider is not None:
            try:
                q, self_mask, opp_mask = self.model_provider.robust_q13(state)
                return q, self_mask, opp_mask, "model_q_robust"
            except Exception:
                # A provider failure must never break decision making — fall back
                # to the handcrafted matrix rather than crash the router.
                pass
        q, self_mask, opp_mask = _immediate_q13(state)
        return q, self_mask, opp_mask, "immediate_q_matrix"

    def _opponent_belief(self, reasoning_state: ReasoningState) -> list[float] | None:
        """Belief over the opponent's next action, from the *adaptive* view only.

        Priority: an explicit belief the caller injected on the adaptive view
        (a known opponent bias) wins; otherwise the trained opponent head is run
        on the adaptive view's memory/history.  Returns ``None`` when neither is
        available, leaving the adaptive tier idle.
        """
        adaptive = reasoning_state.adaptive_view()
        if adaptive.opponent_belief is not None:
            return [float(p) for p in adaptive.opponent_belief]
        if self.model_provider is not None:
            try:
                return self.model_provider.opponent_belief(adaptive)
            except Exception:
                return None
        return None

    def think(
        self,
        reasoning_state: ReasoningState,
        *,
        mode: ToolMode = ToolMode.PLAY,
        generator: torch.Generator | None = None,
    ) -> AgentReasoningResult:
        robust_state = reasoning_state.robust_view().public_state
        q, self_mask, opp_mask, q_source = self._robust_q13(robust_state)
        matrix = solve_matrix_nash_tool(
            q,
            self_mask,
            opp_mask,
            iterations=128,
            mode=mode,
            state_key=reasoning_state.canonical_key,
            model_version=reasoning_state.model_version,
        )
        matrix.policy_self = matrix.policy_self.squeeze(0)
        matrix.policy_opponent = matrix.policy_opponent.squeeze(0)
        matrix.q_matrix = matrix.q_matrix.squeeze(0)
        matrix.valid_self_mask = matrix.valid_self_mask.squeeze(0)
        matrix.valid_opponent_mask = matrix.valid_opponent_mask.squeeze(0)
        tools = [matrix]
        traces = [{"step": "matrix_nash", "valid": matrix.valid, "q_source": q_source}]

        exact = solve_exact_tool(robust_state, max_remaining=self.budget.exact_max_remaining, mode=mode)
        exact.state_key = reasoning_state.canonical_key
        tools.append(exact)
        traces.append({"step": "exact_preflight_and_solve", "valid": exact.valid, "exactness": exact.exactness})
        if not exact.valid and mode != ToolMode.PLAY:
            gt = run_gt_cfr(robust_state, iterations=self.budget.gt_cfr_iterations, mode=mode)
            gt.state_key = reasoning_state.canonical_key
            tools.append(gt)
            traces.append({"step": "gt_cfr", "valid": gt.valid, "iterations": gt.simulations})
        if not exact.valid and strategic_importance(robust_state) >= 0.35:
            sims = self.budget.sm_mcts_mid if mode == ToolMode.PLAY else self.budget.sm_mcts_high
            sm = run_sm_mcts(robust_state, budget=SearchBudget(simulations=sims), mode=mode)
            sm.state_key = reasoning_state.canonical_key
            tools.append(sm)
            traces.append({"step": "sm_mcts", "valid": sm.valid, "simulations": sm.simulations})

        # ---- adaptive tier (Phase 4.2): opponent-conditioned, safe-mixed -----
        # The belief is derived only from the adaptive view; the robust tools
        # above never saw it.  The adaptive candidate is an exact best response
        # to that belief on the current-state matrix.
        adaptive_result: GameToolResult | None = None
        opponent_belief: torch.Tensor | None = None
        belief = self._opponent_belief(reasoning_state)
        if belief is not None:
            adaptive_result = solve_exact_best_response(robust_state, belief, mode=mode)
            adaptive_result.state_key = reasoning_state.canonical_key
            opponent_belief = torch.zeros(13, dtype=torch.float32)
            for idx, prob in enumerate(belief[:13]):
                opponent_belief[idx] = float(prob)
            traces.append({"step": "adaptive_exact_best_response", "valid": adaptive_result.valid})

        final = final_decision(
            tools,
            adaptive_result=adaptive_result,
            opponent_belief=opponent_belief,
            epsilon=self.budget.exploit_epsilon,
            generator=generator,
            state_key=reasoning_state.canonical_key,
        )
        robust = next(tool for tool in tools if tool.source == final.robust_source and tool.valid)
        return AgentReasoningResult(robust_result=robust, tool_results=tools, final=final, traces=traces)
