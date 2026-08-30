from __future__ import annotations

from goofspiel.training.distributed import current_runtime


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
