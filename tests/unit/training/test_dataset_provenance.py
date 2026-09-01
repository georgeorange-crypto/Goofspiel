"""Priority ④ — dataset provenance is CONTENT-addressed, not path-addressed.

Before this, a chained checkpoint recorded its teacher dataset as a path string
(`teacher_dataset_ids=[str(path)]`).  That answers "which file" but not "which
data": edit the dataset in place and the lineage is unchanged — the audit cannot
see that the model was trained on different bytes under the same name.

The fix stamps a structured `datasets` list into the checkpoint metadata, each
entry carrying the file's byte-level `sha256`, the `num_samples` consumed, and a
`role`.  These tests verify it by RE-EXECUTING the fact, never by trusting the
stored field in isolation:

  1. The stored sha256 EQUALS an independently-recomputed sha256 of the dataset
     file on disk (P1 corpus and P3 teacher dataset), and num_samples matches an
     independent count.
  2. The hash actually TRACKS CONTENT: change the dataset bytes, run the stage
     again, and the newly-stored sha256 differs — the property a path string
     could never provide.
"""

from __future__ import annotations

import json

import pytest

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - machine environment guard
    pytest.skip(f"torch cannot be imported in this environment: {exc}", allow_module_level=True)

from goofspiel.training.checkpoint import load_checkpoint, sha256_file


def _dataset_entry(ckpt_path, role: str) -> dict:
    meta = load_checkpoint(ckpt_path)["metadata"]
    entries = [d for d in meta.get("datasets", []) if d["role"] == role]
    assert len(entries) == 1, f"expected exactly one {role} provenance entry, got {entries}"
    return entries[0]


def test_stage1_records_corpus_sha256_matching_disk(tmp_path):
    """P1's stored corpus sha256 must equal a re-hash of the corpus file."""
    from goofspiel.training.corpus import generate_random_game_corpus
    from goofspiel.training.stages import _load_corpus_states, run_stage1_pretrain

    corpus = tmp_path / "data" / "game_corpus.jsonl"
    generate_random_game_corpus(out_path=corpus, num_games=3, seed=1)

    p1 = run_stage1_pretrain(
        steps=1, batch_size=4, out_dir=tmp_path / "ck", n_cards=3, corpus_path=corpus
    )
    entry = _dataset_entry(p1.checkpoint, "game_corpus")

    # (1) sha256 equals an INDEPENDENT re-hash of the same file.
    assert entry["sha256"] == sha256_file(corpus)
    # (2) num_samples equals an INDEPENDENT recount of trainable states.
    assert entry["num_samples"] == len(_load_corpus_states(corpus))
    assert entry["path"] == str(corpus)


def test_stage3_records_teacher_dataset_sha256_matching_disk(tmp_path):
    """P3's stored teacher-dataset sha256 must equal a re-hash of that file."""
    from goofspiel.training.stages import (
        _load_teacher_dataset,
        run_stage1_pretrain,
        run_stage2_semi_supervised,
        run_stage3_sft,
    )

    # P2 writes the teacher dataset P3 consumes.
    run_stage2_semi_supervised(steps=2, out_dir=tmp_path / "data", n_cards=3)
    teacher = tmp_path / "data" / "teacher_dataset.jsonl"
    assert teacher.exists(), "stage2 did not write the teacher dataset"

    p1 = run_stage1_pretrain(steps=1, batch_size=4, out_dir=tmp_path / "ck", n_cards=3)
    p3 = run_stage3_sft(
        steps=1, batch_size=4, out_dir=tmp_path / "ck", n_cards=3,
        init_from_checkpoint=p1.checkpoint, teacher_dataset_path=teacher,
    )
    entry = _dataset_entry(p3.checkpoint, "teacher_dataset")
    assert entry["sha256"] == sha256_file(teacher)
    assert entry["num_samples"] == len(_load_teacher_dataset(teacher))


def test_changed_dataset_yields_different_sha256(tmp_path):
    """The whole point: identical PATH, changed BYTES → different stored sha256.

    A path string cannot distinguish these two runs; the content hash must.
    """
    from goofspiel.training.corpus import generate_random_game_corpus
    from goofspiel.training.stages import run_stage1_pretrain

    corpus = tmp_path / "data" / "game_corpus.jsonl"
    generate_random_game_corpus(out_path=corpus, num_games=3, seed=1)
    p1a = run_stage1_pretrain(
        steps=1, batch_size=4, out_dir=tmp_path / "a", n_cards=3, corpus_path=corpus
    )
    sha_before = _dataset_entry(p1a.checkpoint, "game_corpus")["sha256"]

    # Mutate the dataset in place (same path), then re-run P1.
    generate_random_game_corpus(out_path=corpus, num_games=7, seed=99)
    p1b = run_stage1_pretrain(
        steps=1, batch_size=4, out_dir=tmp_path / "b", n_cards=3, corpus_path=corpus
    )
    sha_after = _dataset_entry(p1b.checkpoint, "game_corpus")["sha256"]

    assert sha_before != sha_after, (
        "corpus content changed but stored sha256 did not — provenance is not content-addressed"
    )
    # And each recorded sha256 matches the file state that produced it (the
    # second one; the first file is gone, but the hash is self-consistent now).
    assert sha_after == sha256_file(corpus)
