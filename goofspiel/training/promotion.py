"""Promotion gates for candidate checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromotionThresholds:
    min_replay_samples: int = 1
    max_q_loss: float = 10.0
    min_entropy: float = 0.05
    require_stage0: bool = True


@dataclass
class PromotionGateReport:
    decision: str
    gates: dict[str, bool]
    metrics: dict[str, float]
    thresholds: PromotionThresholds = field(default_factory=PromotionThresholds)
    notes: list[str] = field(default_factory=list)


def evaluate_promotion_candidate(
    metrics: dict[str, float],
    *,
    stage0_ok: bool = True,
    thresholds: PromotionThresholds | None = None,
) -> PromotionGateReport:
    thresholds = thresholds or PromotionThresholds()
    gates = {
        "stage0_ok": bool(stage0_ok) if thresholds.require_stage0 else True,
        "replay_samples": float(metrics.get("replay_samples", 0.0)) >= thresholds.min_replay_samples,
        "q_loss_finite": abs(float(metrics.get("q_loss_last", 0.0))) <= thresholds.max_q_loss,
        "entropy_floor": float(metrics.get("entropy_last", 0.0)) >= thresholds.min_entropy,
    }
    decision = "PROMOTE_CANDIDATE" if all(gates.values()) else "REJECT_CANDIDATE"
    notes = []
    if decision != "PROMOTE_CANDIDATE":
        notes.append("candidate failed one or more hard gates")
    return PromotionGateReport(decision=decision, gates=gates, metrics={k: float(v) for k, v in metrics.items()}, thresholds=thresholds, notes=notes)


def write_promotion_report(report: PromotionGateReport, out_path: str | Path) -> dict[str, Any]:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"path": str(path), "decision": report.decision, "gates": report.gates}
