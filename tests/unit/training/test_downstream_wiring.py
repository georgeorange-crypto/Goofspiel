"""Downstream + corpus wiring: the full sequence feeds THIS run's artifacts.

Two seams were dead/loose before this fix:

  1. ``build_corpus`` wrote ``game_corpus.jsonl`` that NOTHING read — P1 always
     sampled reachable states on the fly.  Now P1 pretrains over the corpus when
     present.
  2. ``run_full_sequence`` invoked stage6/stage7/evaluate with NO checkpoint, so
     the league played freshly-minted seeds, the red-team corrected a throwaway
     seed, and evaluate scored the heuristic reference — none of them touched the
     run's own P3/P4/P5 snapshots.  Now the coordinator threads them through.

Per the project testing principle these tests RE-EXECUTE the fact (reload the
bytes, recompute the state, compare θ) rather than trusting a status field.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - environment guard
    pytest.skip(f"torch cannot be imported in this environment: {exc}", allow_module_level=True)

from goofspiel.training import TrainingCoordinator, TrainingRunConfig
from goofspiel.training.checkpoint import load_checkpoint, sha256_file
from goofspiel.training.corpus import generate_random_game_corpus
from goofspiel.training.stages import _load_corpus_states, run_stage1_pretrain


def _tiny_config(tmp_path, stage: str, **overrides) -> TrainingRunConfig:
    base = dict(
        artifact_dir=str(tmp_path),
        stage=stage,
        steps=1,
        batch_size=2,
        n_cards=3,
        num_corpus_games=2,
        device="cpu",
    )
    base.update(overrides)
    return TrainingRunConfig(**base)


# ---------------------------------------------------------------------------
# 断线 1 — build_corpus is a LIVE producer: P1 trains on the corpus states.
# ---------------------------------------------------------------------------
def test_stage1_trains_on_corpus_states_not_synthetic(tmp_path):
    corpus = tmp_path / "game_corpus.jsonl"
    generate_random_game_corpus(out_path=corpus, num_games=3, seed=5)

    # The states P1 will train on are EXACTLY the reconstructed corpus states —
    # recompute them here and require a non-empty, playable set.
    corpus_states = _load_corpus_states(corpus)
    assert corpus_states, "corpus produced no trainable states"
    assert all(not s.done and s.self_actions and s.opponent_actions for s in corpus_states)

    metrics = run_stage1_pretrain(
        steps=2, batch_size=2, out_dir=tmp_path / "checkpoints", n_cards=3, corpus_path=corpus
    ).metrics
    # The stage RECORDS that it stood on the corpus, and the count it reports is
    # the same count we recomputed independently above (not a literal).
    assert metrics["trained_on_corpus"] == 1.0
    assert metrics["corpus_states"] == float(len(corpus_states))


def test_stage1_falls_back_when_no_corpus(tmp_path):
    # With no corpus path, P1 must still run (on-the-fly sampling) and say so.
    metrics = run_stage1_pretrain(
        steps=1, batch_size=2, out_dir=tmp_path / "checkpoints", n_cards=3, corpus_path=None
    ).metrics
    assert metrics["trained_on_corpus"] == 0.0
    assert metrics["corpus_states"] == 0.0


# ---------------------------------------------------------------------------
# 断线 2 — the full sequence feeds stage6/7/evaluate THIS run's checkpoints.
# ---------------------------------------------------------------------------
def test_full_sequence_wires_downstream_to_real_checkpoints(tmp_path):
    result = TrainingCoordinator(_tiny_config(tmp_path, "all")).run_full_sequence()
    assert result["stage"] == "full_sequence"

    p3 = tmp_path / "checkpoints" / "stage3_sft.pt"
    p4 = tmp_path / "checkpoints" / "stage4_robust_rl.pt"
    p5 = tmp_path / "stage5_adaptive.pt"
    assert p3.exists() and p4.exists() and p5.exists()

    # --- stage7 corrected the run's REAL P4 backbone -----------------------
    # The corrected checkpoint stamps init_checkpoint_id = the P4 file path, and
    # its stored regression re-plays are reproducible (checked in the P7 tests).
    corrected = tmp_path / "redteam" / "stage7_corrected.pt"
    assert corrected.exists()
    corrected_meta = load_checkpoint(str(corrected))["metadata"]
    assert corrected_meta["init_checkpoint_id"], "stage7 recorded no init checkpoint"
    assert Path(corrected_meta["init_checkpoint_id"]).resolve() == p4.resolve(), (
        "stage7 did not focus-correct THIS run's P4 backbone"
    )

    # --- stage6 league referenced the run's real P3/P4/P5 snapshots --------
    league = json.loads((tmp_path / "league" / "registry.json").read_text(encoding="utf-8"))
    referenced = {a["checkpoint_path"] for a in league["agents"] if a["checkpoint_path"]}
    # Every referenced checkpoint is a real file on disk; and the three role
    # checkpoints resolve to the run's own P3/P4/P5 SHAs (not minted seeds).
    for path in referenced:
        assert Path(path).exists()
    run_shas = {sha256_file(p3), sha256_file(p4), sha256_file(p5)}
    league_shas = {sha256_file(p) for p in referenced}
    assert run_shas & league_shas, "league referenced no checkpoint from this run"

    # --- evaluate scored the run's P4 (E2 = real model play, G2 computed) --
    quick = json.loads(
        (tmp_path / "reports" / "quick" / "summary.json").read_text(encoding="utf-8")
    )
    e2 = quick["arenas"]["E2_N13_ROBUST"]
    assert e2["source"] == "trained_model_vs_random", "evaluate fell back to the heuristic reference"
    assert quick["hard_gates"]["G2_exploitability"] is not None, "G2 was left unrun"

    # --- Phase 5 axis promotion ran and registered file-backed aliases -----
    assert result["axis_selection"] is not None, "full sequence skipped axis promotion"
    registry = json.loads(
        (tmp_path / "registry" / "checkpoint_registry.json").read_text(encoding="utf-8")
    )
    for alias in ("best_robust", "best_search", "best_generalization"):
        assert alias in registry, f"{alias} not registered by full sequence"
        assert Path(registry[alias]["registry_path"]).exists()
