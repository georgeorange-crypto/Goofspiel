"""Reasoning layer primitives.

Heavy numerical tools are imported lazily so dataclass/schema inspection works
before torch/CUDA is configured on a fresh machine.
"""

from .state import CanonicalStateKey, ChanceStateKey, ReasoningState
from .types import Exactness, GameToolResult, ToolMode

_LAZY = {
    "FinalDecision": ("goofspiel.reasoning.decision", "FinalDecision"),
    "GameAgent": ("goofspiel.reasoning.agent", "GameAgent"),
    "AgentReasoningResult": ("goofspiel.reasoning.router", "AgentReasoningResult"),
    "DecisionBudget": ("goofspiel.reasoning.router", "DecisionBudget"),
    "LeafEvaluator": ("goofspiel.reasoning.leaf", "LeafEvaluator"),
    "SearchBudget": ("goofspiel.reasoning.search", "SearchBudget"),
    "ToolRouter": ("goofspiel.reasoning.router", "ToolRouter"),
    "final_decision": ("goofspiel.reasoning.decision", "final_decision"),
    "run_gt_cfr": ("goofspiel.reasoning.search", "run_gt_cfr"),
    "run_sm_mcts": ("goofspiel.reasoning.search", "run_sm_mcts"),
    "safe_exploit_mixture": ("goofspiel.reasoning.safe_mixture", "safe_exploit_mixture"),
    "select_robust_result": ("goofspiel.reasoning.decision", "select_robust_result"),
    "solve_exact_best_response": ("goofspiel.reasoning.exact_br", "solve_exact_best_response"),
    "solve_exact_tool": ("goofspiel.reasoning.exact_tool", "solve_exact_tool"),
    "solve_matrix_nash_tool": ("goofspiel.reasoning.matrix_tools", "solve_matrix_nash_tool"),
    "validate_tool_result": ("goofspiel.reasoning.decision", "validate_tool_result"),
}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(name)
    import importlib

    module_name, attr = _LAZY[name]
    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value
    return value

__all__ = [
    "Exactness",
    "FinalDecision",
    "GameAgent",
    "GameToolResult",
    "AgentReasoningResult",
    "CanonicalStateKey",
    "ChanceStateKey",
    "DecisionBudget",
    "LeafEvaluator",
    "ReasoningState",
    "SearchBudget",
    "ToolMode",
    "ToolRouter",
    "final_decision",
    "run_gt_cfr",
    "run_sm_mcts",
    "safe_exploit_mixture",
    "select_robust_result",
    "solve_exact_best_response",
    "solve_exact_tool",
    "solve_matrix_nash_tool",
    "validate_tool_result",
]
