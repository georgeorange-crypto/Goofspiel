"""Priority ① — Standalone vs Full-sequence lineage contract.

`test_coordinator_autowire.py` proves the θ-producing chain (1→3→4→5) hard-fails
in the full sequence when a parent's weights are missing.  That protection stops
at the θ boundary, though: the DOWNSTREAM stages (league / red-team / evaluate)
are not θ-producers, and `_resolve_checkpoint` lets them degrade to a
freshly-minted seed when their upstream product is absent.

Honest degrade is correct in ONE mode and a lie in the other:

  * Standalone (single-stage `run`): the user asked for exactly one stage in
    isolation, so evaluating/leaguing a fresh seed is the honest thing to do.
  * Full-sequence (`run_full_sequence`): every downstream stage is contractually
    built on THIS run's real snapshots.  Silently substituting a throwaway seed
    and labelling the result as this run's league/eval is exactly the kind of
    dishonesty the remediation exists to kill.

So the same code path must behave differently by mode.  These tests pin both
sides of that fork by RE-EXECUTING it, never by reading a mode flag back:

  1. Full-sequence RAISES when a *downstream-required* product is missing — and
     names the specific missing role, not a generic failure.  We sabotage the
     TERMINAL θ-stage (stage5/P5) so the θ-chain itself stays intact (nothing
     inherits from P5) and only the downstream contract (stage6 needs P5 as the
     exploiter) is violated — isolating this path from the θ hard-fail.
  2. The very same missing product, in STANDALONE mode, does NOT raise: the
     stage runs and reports honestly that it consumed no upstream checkpoint.
"""

from __future__ import annotations

import pytest

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - machine environment guard
    pytest.skip(f"torch cannot be imported in this environment: {exc}", allow_module_level=True)

from goofspiel.training import TrainingCoordinator, TrainingRunConfig


def _tiny_config(tmp_path, stage: str, **overrides) -> TrainingRunConfig:
    base = dict(
        artifact_dir=str(tmp_path),
        stage=stage,
        steps=1,
        batch_size=2,
        n_cards=2,
        num_corpus_games=1,
        device="cpu",
    )
    base.update(overrides)
    return TrainingRunConfig(**base)


# ---------------------------------------------------------------------------
# 1. Full-sequence mode: a missing DOWNSTREAM product is a broken lineage.
# ---------------------------------------------------------------------------
def test_full_sequence_raises_when_downstream_product_missing(tmp_path, monkeypatch):
    """Sabotage P5 (terminal θ-stage) → stage6_league must refuse in strict mode.

    P5 feeds no θ-stage, so the θ auto-wiring chain is untouched and the run
    reaches stage6_league with P4/P3 present but P5 absent.  In full-sequence
    (strict) mode the league REQUIRES P5 as the exploiter role, so it must raise
    — naming P5 — rather than mint a throwaway exploiter seed and pass the match
    off as this run's league result.
    """
    from goofspiel.training import coordinator as coord_mod

    real_dispatch = coord_mod.TrainingCoordinator._dispatch_stage

    def sabotaged(self, stage, *, init_from_checkpoint, produced=None, strict=False):
        out = real_dispatch(
            self, stage, init_from_checkpoint=init_from_checkpoint, produced=produced, strict=strict
        )
        if stage == "stage5_adaptive":
            # Simulate P5's checkpoint write vanishing AFTER the θ-chain has
            # already consumed P4 (P5 inherits P4; nothing inherits P5).
            if isinstance(out.get("metrics"), dict):
                out["metrics"]["checkpoint"] = None
        return out

    monkeypatch.setattr(coord_mod.TrainingCoordinator, "_dispatch_stage", sabotaged)

    # Also hide any on-disk P5 the resolver could fall back to, so "missing"
    # really means missing and the strict raise is the ONLY possible outcome.
    real_resolve = coord_mod.TrainingCoordinator._resolve_checkpoint

    def resolve_no_p5(self, stage, produced):
        if stage == "stage5_adaptive":
            return None
        return real_resolve(self, stage, produced)

    monkeypatch.setattr(coord_mod.TrainingCoordinator, "_resolve_checkpoint", resolve_no_p5)

    with pytest.raises(RuntimeError, match=r"lineage broken.*stage6_league.*stage5_adaptive"):
        TrainingCoordinator(_tiny_config(tmp_path, "all")).run_full_sequence()


def test_full_sequence_raises_name_the_consumer_and_role(tmp_path):
    """A direct strict dispatch with an empty produced-map names WHAT is missing.

    Calling the downstream dispatch strictly with nothing produced must raise
    before doing any league/eval work, and the message must identify both the
    consumer stage and the concrete role/producer it needed — an operator has to
    know which product to go rebuild.
    """
    coord = TrainingCoordinator(_tiny_config(tmp_path, "stage6_league"))
    with pytest.raises(RuntimeError, match=r"robust backbone \(P4\).*stage4_robust_rl"):
        coord._dispatch_stage("stage6_league", init_from_checkpoint=None, produced={}, strict=True)

    with pytest.raises(RuntimeError, match=r"stage7_redteam.*robust backbone \(P4\)"):
        coord._dispatch_stage("stage7_redteam", init_from_checkpoint=None, produced={}, strict=True)

    with pytest.raises(RuntimeError, match=r"'evaluate'.*robust backbone \(P4\)"):
        coord._dispatch_stage("evaluate", init_from_checkpoint=None, produced={}, strict=True)


# ---------------------------------------------------------------------------
# 2. Standalone mode: the SAME missing product degrades honestly, no raise.
# ---------------------------------------------------------------------------
def test_standalone_evaluate_degrades_without_checkpoint(tmp_path):
    """`--stage evaluate` alone (no upstream) must succeed and say ckpt is None.

    This is the contrast to the strict raise above: identical missing product,
    opposite contract.  Standalone deliberately runs one stage in isolation, so
    the honest thing is to evaluate the heuristic reference and REPORT that no
    trained checkpoint was consumed — not to raise.
    """
    result = TrainingCoordinator(_tiny_config(tmp_path, "evaluate")).run()
    assert result["ok"] is True
    # Degraded honestly: no upstream P4 existed, so none was consumed — and the
    # result says so rather than pretending or crashing.
    assert result["evaluated_checkpoint"] is None
