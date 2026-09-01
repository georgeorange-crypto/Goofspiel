"""Shared pytest fixtures for the Goofspiel test suite.

Consolidates the ``_tiny_config`` builder and the torch-import skip guard that
had been copy-pasted across several training tests.  New tests should depend on
these fixtures; the older tests keep their local copies (migrating them is a
pure refactor with regression risk and no behavioural gain).
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def torch_or_skip():
    """Import torch once per session, skipping the whole test if unavailable.

    Some CI machines have no working torch build; a test that needs it should
    ``request.getfixturevalue`` this or take it as an argument rather than
    repeating the try/except at module import.
    """
    try:
        import torch
    except OSError as exc:  # pragma: no cover - machine environment guard
        pytest.skip(f"torch cannot be imported in this environment: {exc}")
    return torch


@pytest.fixture
def tiny_config():
    """Factory for a minimal ``TrainingRunConfig`` into a temp artifact dir.

    Usage::

        def test_something(tmp_path, tiny_config):
            cfg = tiny_config(tmp_path, stage="stage1_pretrain", steps=2)

    Defaults are the smallest values that still exercise every code path
    (1 step, batch 2, 3 cards, 1 corpus game), so an end-to-end run stays fast.
    ``n_cards`` defaults to 3 — the smallest board on which the honest
    exploitability sweep and the league still have legal structure to play.
    """
    from goofspiel.training import TrainingRunConfig

    def _make(tmp_path, *, stage: str = "stage0_verify", **overrides):
        base = dict(
            artifact_dir=str(tmp_path),
            stage=stage,
            steps=1,
            batch_size=2,
            n_cards=3,
            num_corpus_games=1,
            device="cpu",
        )
        base.update(overrides)
        return TrainingRunConfig(**base)

    return _make
