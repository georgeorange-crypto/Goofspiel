from __future__ import annotations

from goofspiel.training.distributed import current_runtime, derive_rank_seed


def test_current_runtime_reads_torchrun_environment(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")
    runtime = current_runtime()
    assert runtime.rank == 3
    assert runtime.local_rank == 1
    assert runtime.world_size == 8
    assert runtime.is_distributed
    assert not runtime.is_rank0


def test_rank_seed_derivation_decorrelates_and_replays():
    seeds_a = [derive_rank_seed(1234, rank) for rank in range(8)]
    seeds_b = [derive_rank_seed(1234, rank) for rank in range(8)]
    assert seeds_a == seeds_b
    assert len(set(seeds_a)) == len(seeds_a)
    assert seeds_a[0] == 1234
    assert seeds_a[1] != seeds_a[0]
