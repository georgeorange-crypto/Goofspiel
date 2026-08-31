"""Phase 2.2b — P3 (Robust Strategic SFT) must CONSUME a multi-source dataset.

Before this phase P2 wrote a `teacher_dataset.jsonl` that no one read, and P3's
four "SFT source" metrics were all the same `exact_anchor_count`.  These tests
assert the fix at the level of *behaviour*, not metric labels:

  1. The four robust teacher sources produce genuinely DIFFERENT sample counts
     and DIFFERENT policies where they overlap (re-executed here, not read).
  2. None of the sources carries opponent behaviour (Q_R ⊥ Q_A firewall).
  3. P3's loss actually RESPONDS to the dataset content — swapping the stored
     teacher policies changes the first-step loss — proving P3 trains on the
     file rather than re-deriving targets and ignoring it.

Per the project testing principle these re-run the underlying computation
(rebuild the dataset, retrain a step) instead of trusting an emitted field.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - machine env guard
    pytest.skip(f"torch cannot be imported: {exc}", allow_module_level=True)

import numpy as np

from goofspiel.game import GameState, transition
from goofspiel.training.data import JsonlStore
from goofspiel.training.state_coverage import sample_reachable_states
from goofspiel.training.teacher_dataset import (
    ROBUST_TEACHER_SOURCES,
    build_teacher_dataset,
    cfr_label,
    exact_label,
    search_label,
)


def test_four_sources_have_distinct_counts_and_no_opponent_signal(tmp_path):
    states = sample_reachable_states(48, n=5, step=0, seed=1)
    store = JsonlStore(tmp_path / "teacher_dataset.jsonl")
    counts = build_teacher_dataset(states, store)

    # All four robust sources are present.
    assert set(counts) == set(ROBUST_TEACHER_SOURCES)
    # They are not four aliases of one number: at N=5 the applicability gates
    # (EXACT small-N/policy-map, SEARCH budget, PSEUDO confidence) make the
    # coverages genuinely differ.
    assert len(set(counts.values())) >= 3, counts

    # No opponent-behaviour source leaked in (Q_R ⊥ Q_A).
    sources = {row["teacher_source"] for row in store.iter_dicts()}
    assert sources <= set(ROBUST_TEACHER_SOURCES)
    assert not any("OPP" in s.upper() or "OPPONENT" in s.upper() for s in sources)


def test_sources_disagree_where_continuation_matters():
    """CFR (immediate) and EXACT (full recursion) must give different policies on
    a state where the future rounds actually change the right play.

    Re-executes both labelers on a reachable N=5, 3-cards-left state and asserts
    the exact full-game policy differs from the myopic immediate one — the whole
    point of having distinct sources."""
    s0 = GameState.initial(5, current_prize=1)
    s1 = transition(s0, 1, 1, next_prize=5).state   # tie -> carry, prize 5
    s2 = transition(s1, 2, 3, next_prize=4).state    # opp wins -> 3 cards, prize 4

    cfr = cfr_label(s2)
    exact = exact_label(s2)
    assert cfr is not None and exact is not None
    cfr_pol = np.asarray(cfr.teacher_policy)
    exact_pol = np.asarray(exact.teacher_policy)
    # The immediate teacher piles onto the highest card; the exact teacher spreads
    # mass because it values keeping cards for later rounds. They must differ.
    assert not np.allclose(cfr_pol, exact_pol, atol=1e-2), (cfr_pol, exact_pol)


def test_p3_loss_responds_to_dataset_content(tmp_path):
    """P3 must train on the dataset: replacing stored teacher policies with a
    different target changes P3's measured first-step loss.

    This is the discriminating fact — if P3 ignored the file (re-deriving targets
    internally, the pre-Phase-2.2b behaviour) the loss would be identical for both
    datasets. We build two datasets that differ ONLY in the stored teacher
    policies and assert P3's loss differs.
    """
    from goofspiel.training.stages import run_stage3_sft

    states = sample_reachable_states(16, n=5, step=0, seed=1)

    # Dataset A: the real multi-source teacher policies.
    data_a = tmp_path / "a" / "teacher_dataset.jsonl"
    store_a = JsonlStore(data_a)
    build_teacher_dataset(states, store_a)

    # Dataset B: same states/rows but the teacher policy is deliberately corrupted
    # to a reversed one-hot, so the KL target is very different. If P3 consumes the
    # file, its loss must react.
    data_b = tmp_path / "b" / "teacher_dataset.jsonl"
    data_b.parent.mkdir(parents=True, exist_ok=True)
    rows = list(JsonlStore(data_a).iter_dicts())
    with data_b.open("w", encoding="utf-8") as fh:
        for row in rows:
            pol = row.get("teacher_policy") or []
            # Move all mass to the *lowest* legal card index instead of the taught one.
            legal_idx = [i for i, v in enumerate(pol)]
            corrupted = [0.0] * len(pol)
            # Put mass on the first index that is a legal action in the state.
            n = row["state"]["n"]
            self_mask = row["state"]["self_mask"]
            legal = [c - 1 for c in range(1, n + 1) if self_mask & (1 << (c - 1))]
            corrupted[min(legal)] = 1.0
            row = dict(row)
            row["teacher_policy"] = corrupted
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    m_a = run_stage3_sft(steps=1, batch_size=16, out_dir=tmp_path / "cka", n_cards=5, teacher_dataset_path=data_a)
    m_b = run_stage3_sft(steps=1, batch_size=16, out_dir=tmp_path / "ckb", n_cards=5, teacher_dataset_path=data_b)

    assert m_a.metrics["teacher_dataset_consumed"] == 1.0
    assert m_b.metrics["teacher_dataset_consumed"] == 1.0
    # The two datasets differ only in the taught policy; a P3 that truly consumes
    # the file yields different losses. Identical losses would mean the file is
    # ignored (the bug this phase fixes).
    assert m_a.metrics["loss_last"] != m_b.metrics["loss_last"], (
        m_a.metrics["loss_last"],
        m_b.metrics["loss_last"],
    )


def test_p3_reports_four_source_counts_from_the_file(tmp_path):
    """P3's four `sft_source_*` counts must equal the file's per-source rows —
    recomputed here by re-reading the dataset, not by trusting P3's metric."""
    from goofspiel.training.stages import run_stage3_sft

    states = sample_reachable_states(24, n=5, step=0, seed=1)
    data = tmp_path / "teacher_dataset.jsonl"
    store = JsonlStore(data)
    build_teacher_dataset(states, store)

    # Ground truth counts by re-reading the file.
    truth = {src: 0 for src in ROBUST_TEACHER_SOURCES}
    for row in JsonlStore(data).iter_dicts():
        truth[row["teacher_source"]] += 1

    metrics = run_stage3_sft(steps=2, batch_size=8, out_dir=tmp_path / "ck", n_cards=5, teacher_dataset_path=data).metrics
    for src in ROBUST_TEACHER_SOURCES:
        assert metrics[f"sft_source_{src.lower()}_samples"] == float(truth[src])
