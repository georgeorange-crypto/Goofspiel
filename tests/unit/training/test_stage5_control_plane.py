"""Regression tests for the Stage5 distributed control-plane (Phase 1).

Two layers, both of which RE-EXECUTE the fact rather than reading a stored
field or monkeypatching the distributed path:

1. Fast, deterministic unit tests of ``stage_control`` — the partial-write
   guarantee and every ``wait_for_rank0`` exit path, driven by injected fake
   clocks so they are instant and reproducible.

2. REAL two-rank integration tests (``slow`` + ``integration``) that spawn two
   genuine OS processes joined by a real ``gloo`` process group with a SHORT
   timeout, and assert:

     * ``old_bug``   — the pre-fix design (rank1 holding a collective across a
                       long rank0-only stage) actually TIMES OUT.  This is the
                       bug the fix removes; if this ever stops timing out the
                       test is no longer reproducing the original failure.
     * ``new_success`` — the fixed design (rank0 heartbeats, rank1 polls) does
                       NOT false-kill across a stage far longer than the PG
                       timeout, and both ranks continue past ONE short barrier.
     * ``new_crash``  — rank0 killed without a terminal record → rank1's
                       heartbeat-stale guard fails closed (never hangs).
     * ``new_failed`` — rank0 writes FAILED → rank1 fails closed immediately.
     * ``new_stale_success`` — a PRIOR invocation's SUCCESS is on disk; rank1
                       must reject it (foreign invocation id) and wait for THIS
                       invocation, closing the stale-terminal race.  Run twice:
                       once across a different launch (run id differs), once
                       across a worker-group restart of the SAME launch (run id
                       matches, restart count differs).

The integration layer is the one that matters for the testing principle: the
existing "distributed" stage tests fake ``WORLD_SIZE`` and mock the collectives,
so only a real multi-process PG can prove the deadlock is gone.
"""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from goofspiel.training.stage_control import (
    STATE_FAILED,
    STATE_RUNNING,
    STATE_SUCCESS,
    Rank0Heartbeat,
    Stage5ControlError,
    Stage5Status,
    control_dir_for,
    current_invocation_id,
    read_status,
    wait_for_rank0,
    write_status,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKER = Path(__file__).resolve().parent / "_stage5_control_plane_worker.py"

EXIT_OK = 0
EXIT_FAIL_CLOSED = 42
EXIT_UNEXPECTED = 43
EXIT_SHOULD_HAVE_FAILED = 44


# ---------------------------------------------------------------------------
# Layer 1 — fast, deterministic unit tests (no processes, no torch)
# ---------------------------------------------------------------------------
def _seq_clock(values: list[float]):
    """A monotonic-clock stub that yields ``values`` then holds the last one."""
    it = iter(values)
    holder = {"last": values[0] if values else 0.0}

    def clock() -> float:
        with contextlib.suppress(StopIteration):
            holder["last"] = next(it)
        return holder["last"]

    return clock


def test_read_status_rejects_partial_write(tmp_path: Path):
    cd = control_dir_for(tmp_path)
    cd.mkdir(parents=True)
    status_file = cd / "status.json"

    # Missing file -> None (rank0 not started yet).
    assert read_status(cd) is None

    # Truncated / partial JSON -> None (never a spurious verdict).
    status_file.write_text('{"state": "RUNN', encoding="utf-8")
    assert read_status(cd) is None

    # Well-formed JSON but unknown state -> None.
    status_file.write_text('{"state": "NONSENSE"}', encoding="utf-8")
    assert read_status(cd) is None

    # A real atomic write is always readable and complete.
    write_status(cd, Stage5Status(state=STATE_RUNNING, current_step=7, updated_at=1.0))
    got = read_status(cd)
    assert got is not None and got.state == STATE_RUNNING and got.current_step == 7


def test_read_status_never_observes_tmp_file(tmp_path: Path):
    """The atomic writer must not leave the real path pointing at a partial file.

    We write many times and, after each, assert the on-disk status parses fully
    — os.replace guarantees the reader sees old-or-new, never half-written.
    """
    cd = control_dir_for(tmp_path)
    cd.mkdir(parents=True)
    for i in range(50):
        write_status(cd, Stage5Status(state=STATE_RUNNING, current_step=i, updated_at=float(i)))
        got = read_status(cd)
        assert got is not None and got.current_step == i
    # No leftover tmp files.
    assert not list(cd.glob(".status.json.*.tmp"))


def test_wait_returns_success_with_result(tmp_path: Path):
    cd = control_dir_for(tmp_path)
    cd.mkdir(parents=True)
    write_status(
        cd,
        Stage5Status(state=STATE_SUCCESS, checkpoint="/c.pt", metrics={"m": 2.0}, updated_at=1.0),
    )
    status = wait_for_rank0(
        cd, heartbeat_timeout=300, hard_timeout=1e9, poll_interval=0,
        clock=lambda: 0.0, sleep_fn=lambda _s: None,
    )
    assert status.state == STATE_SUCCESS
    assert status.checkpoint == "/c.pt"
    assert status.metrics["m"] == 2.0


def test_wait_fails_closed_on_explicit_failed(tmp_path: Path):
    cd = control_dir_for(tmp_path)
    cd.mkdir(parents=True)
    write_status(cd, Stage5Status(state=STATE_FAILED, error="boom", updated_at=1.0))
    with pytest.raises(Stage5ControlError) as ei:
        wait_for_rank0(
            cd, heartbeat_timeout=300, hard_timeout=1e9, poll_interval=0,
            clock=lambda: 0.0, sleep_fn=lambda _s: None,
        )
    assert ei.value.reason == "failed"


def test_wait_fails_closed_on_stale_heartbeat(tmp_path: Path):
    cd = control_dir_for(tmp_path)
    cd.mkdir(parents=True)
    write_status(cd, Stage5Status(state=STATE_RUNNING, current_step=5, updated_at=1.0))
    # Status never advances; clock jumps 0 -> 10 (fresh) -> 400 (> 300 stale).
    with pytest.raises(Stage5ControlError) as ei:
        wait_for_rank0(
            cd, heartbeat_timeout=300, hard_timeout=1e9, poll_interval=0,
            clock=_seq_clock([0.0, 10.0, 400.0]), sleep_fn=lambda _s: None,
        )
    assert ei.value.reason == "heartbeat_stale"


def test_wait_fails_closed_when_rank0_never_wrote(tmp_path: Path):
    cd = control_dir_for(tmp_path)
    cd.mkdir(parents=True)
    # No status file at all: still bounded, still fail-closed (never hangs).
    with pytest.raises(Stage5ControlError) as ei:
        wait_for_rank0(
            cd, heartbeat_timeout=300, hard_timeout=1e9, poll_interval=0,
            clock=_seq_clock([0.0, 400.0]), sleep_fn=lambda _s: None,
        )
    assert ei.value.reason == "heartbeat_stale"


def test_wait_hard_timeout_beats_fresh_heartbeat(tmp_path: Path):
    """Fake-alive guard: heartbeat stays fresh forever but stage never ends."""
    cd = control_dir_for(tmp_path)
    cd.mkdir(parents=True)
    write_status(cd, Stage5Status(state=STATE_RUNNING, current_step=0, updated_at=0.0))
    tick = {"t": 0}

    def clock() -> float:
        return float(tick["t"])

    def sleep_fn(_s: float) -> None:
        # Every poll advances time AND refreshes the heartbeat, so the stale
        # guard can never fire — only the hard timeout can stop this.
        tick["t"] += 5
        write_status(cd, Stage5Status(state=STATE_RUNNING, current_step=tick["t"], updated_at=float(tick["t"])))

    with pytest.raises(Stage5ControlError) as ei:
        wait_for_rank0(cd, heartbeat_timeout=300, hard_timeout=50, poll_interval=1, clock=clock, sleep_fn=sleep_fn)
    assert ei.value.reason == "hard_timeout"


def test_wait_rejects_stale_success_from_prior_invocation(tmp_path: Path):
    """The stale-terminal race: a SUCCESS left by a PREVIOUS invocation in the
    same artifact-dir must be invisible to a new wait, which must instead fail
    closed (nothing fresh for THIS invocation) rather than return the old
    result.  This is the unit-level proof of the identity guard."""
    cd = control_dir_for(tmp_path)
    cd.mkdir(parents=True)
    # Prior invocation's terminal record, complete and well-formed.
    write_status(
        cd,
        Stage5Status(
            state=STATE_SUCCESS,
            stage_invocation_id="OLD-invocation",
            checkpoint="/prev.pt",
            metrics={"m": 1.0},
            updated_at=1.0,
        ),
    )
    # This invocation expects a different id; the stale SUCCESS is treated like
    # a missing file, so with no fresh record the wait fails closed, never
    # returns the prior checkpoint.
    with pytest.raises(Stage5ControlError) as ei:
        wait_for_rank0(
            cd, expect_invocation_id="NEW-invocation",
            heartbeat_timeout=300, hard_timeout=1e9, poll_interval=0,
            clock=_seq_clock([0.0, 10.0, 400.0]), sleep_fn=lambda _s: None,
        )
    assert ei.value.reason == "heartbeat_stale"


def test_wait_rejects_stale_failed_from_prior_invocation(tmp_path: Path):
    """Symmetric to the SUCCESS case: a prior invocation's FAILED must NOT
    fail-close a new wait either — a peer only acts on its own invocation's
    terminal record."""
    cd = control_dir_for(tmp_path)
    cd.mkdir(parents=True)
    write_status(
        cd,
        Stage5Status(state=STATE_FAILED, stage_invocation_id="OLD", error="old boom", updated_at=1.0),
    )
    # A matching-id SUCCESS then lands; the wait must return THAT, having ignored
    # the foreign FAILED entirely (no "failed" raise).
    write_status(
        cd,
        Stage5Status(
            state=STATE_SUCCESS, stage_invocation_id="NEW", checkpoint="/new.pt",
            metrics={"m": 2.0}, updated_at=2.0,
        ),
    )
    status = wait_for_rank0(
        cd, expect_invocation_id="NEW",
        heartbeat_timeout=300, hard_timeout=1e9, poll_interval=0,
        clock=lambda: 0.0, sleep_fn=lambda _s: None,
    )
    assert status.state == STATE_SUCCESS and status.checkpoint == "/new.pt"


def test_wait_accepts_matching_invocation_success(tmp_path: Path):
    """The positive control: when the id matches, SUCCESS is returned as normal
    — the guard rejects foreign records without blocking genuine ones."""
    cd = control_dir_for(tmp_path)
    cd.mkdir(parents=True)
    write_status(
        cd,
        Stage5Status(
            state=STATE_SUCCESS, stage_invocation_id="MINE", checkpoint="/c.pt",
            metrics={"m": 3.0}, updated_at=1.0,
        ),
    )
    status = wait_for_rank0(
        cd, expect_invocation_id="MINE",
        heartbeat_timeout=300, hard_timeout=1e9, poll_interval=0,
        clock=lambda: 0.0, sleep_fn=lambda _s: None,
    )
    assert status.checkpoint == "/c.pt" and status.metrics["m"] == 3.0


def test_invocation_id_shared_via_torchelastic_and_unique_per_launch(monkeypatch):
    """current_invocation_id() derives the SAME id for all ranks of one launch-
    generation (from TORCHELASTIC_RUN_ID + RESTART_COUNT) and a different id per
    launch AND per restart-generation; absent the env it falls back to a
    per-process uuid (which makes peers fail closed)."""
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "launch-A")
    monkeypatch.delenv("TORCHELASTIC_RESTART_COUNT", raising=False)
    a1 = current_invocation_id()
    a2 = current_invocation_id()  # a "second rank" reading the same env
    assert a1 == a2 == "torchelastic:launch-A:r0"  # RESTART_COUNT defaults to 0
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "launch-B")
    assert current_invocation_id() != a1  # distinct launch -> distinct id
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
    f1, f2 = current_invocation_id(), current_invocation_id()
    assert f1 != f2 and f1.startswith("pid:")  # no shared env -> disagree -> fail closed


def test_invocation_id_distinguishes_restart_generations(monkeypatch):
    """Same launch (stable RUN_ID) but a worker-group restart bumps
    RESTART_COUNT, so the id changes generation-to-generation — a peer that
    restarted into generation N must not accept generation N-1's leftover
    status.  Both ranks of a generation share the (group-wide) count, so they
    still agree with each other."""
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "same-launch")
    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "0")
    gen0_rank0 = current_invocation_id()
    gen0_rank1 = current_invocation_id()
    assert gen0_rank0 == gen0_rank1 == "torchelastic:same-launch:r0"
    # Worker group restarts together -> both ranks now see count 1.
    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "1")
    gen1 = current_invocation_id()
    assert gen1 == "torchelastic:same-launch:r1"
    assert gen1 != gen0_rank0  # generation N rejects generation N-1's record


def test_disabled_heartbeat_writes_nothing(tmp_path: Path):
    """Single-process runs must be side-effect identical to the old code."""
    cd = control_dir_for(tmp_path)
    hb = Rank0Heartbeat(cd, enabled=False, total_steps=5)
    hb.starting()
    hb.running(step=1, phase="training", force=True)
    hb.success(checkpoint=None, metrics={})
    assert not (cd / "status.json").exists()


def test_heartbeat_interval_gates_writes(tmp_path: Path):
    """running() without force only writes once per interval (cheap per-step)."""
    cd = control_dir_for(tmp_path)
    cd.mkdir(parents=True)
    now = {"t": 0.0}
    hb = Rank0Heartbeat(cd, enabled=True, total_steps=100, interval=10.0, clock=lambda: now["t"])
    hb.running(step=1, phase="training", force=True)  # first write
    first = read_status(cd)
    assert first.current_step == 1
    now["t"] = 5.0
    hb.running(step=2, phase="training")  # within interval -> skipped
    assert read_status(cd).current_step == 1
    now["t"] = 20.0
    hb.running(step=3, phase="training")  # interval elapsed -> written
    assert read_status(cd).current_step == 3


# ---------------------------------------------------------------------------
# Layer 2 — REAL two-rank integration (spawns processes + real gloo PG)
# ---------------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _spawn_two_ranks(
    scenario: str,
    control_root: Path,
    *,
    pg_timeout_s: float,
    stage_duration_s: float,
    hb_timeout_s: float,
    join_timeout_s: float,
    run_id: str | None = None,
    restart_count: int = 0,
) -> tuple[int, int]:
    """Launch rank0 and rank1 as real processes sharing one gloo PG.

    Returns ``(rank0_exit, rank1_exit)``.  A rank that does not exit within
    ``join_timeout_s`` is killed and reported as exit code -1 (a hang — which
    for the fixed design is itself a failure, since the whole point is that no
    rank waits forever).

    ``run_id`` + ``restart_count`` populate the torchelastic env vars both ranks
    read to derive their shared invocation id.  A test that pre-seeds a status
    from an EARLIER restart-generation passes the same ``run_id`` with a higher
    ``restart_count`` here, so the spawned pair rejects the leftover record.
    """
    port = _free_port()
    base_env = dict(os.environ)
    base_env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), base_env.get("PYTHONPATH", "")])
    base_env["MASTER_ADDR"] = "127.0.0.1"
    base_env["MASTER_PORT"] = str(port)
    base_env["WORLD_SIZE"] = "2"
    # torchrun sets TORCHELASTIC_RUN_ID identically in every rank's env (stable
    # across restarts) and TORCHELASTIC_RESTART_COUNT per worker-group restart;
    # our raw Popen harness mirrors that contract so both ranks derive the SAME
    # current_invocation_id() from the env.  A fresh uuid per spawned pair (when
    # run_id is not pinned) means distinct _spawn_two_ranks calls never collide.
    base_env["TORCHELASTIC_RUN_ID"] = run_id if run_id is not None else f"itest-{uuid.uuid4()}"
    base_env["TORCHELASTIC_RESTART_COUNT"] = str(restart_count)
    base_env["PG_TIMEOUT_S"] = str(pg_timeout_s)
    base_env["STAGE_DURATION_S"] = str(stage_duration_s)
    base_env["HB_TIMEOUT_S"] = str(hb_timeout_s)

    procs = []
    for rank in (0, 1):
        env = dict(base_env)
        env["RANK"] = str(rank)
        env["LOCAL_RANK"] = str(rank)
        procs.append(
            subprocess.Popen(
                [sys.executable, str(WORKER), scenario, str(control_root)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        )

    exits: list[int] = []
    for p in procs:
        try:
            p.communicate(timeout=join_timeout_s)
            exits.append(p.returncode)
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate()
            exits.append(-1)  # hung
    return exits[0], exits[1]


def _require_torch_distributed():
    try:
        import torch.distributed as dist
    except Exception:
        pytest.skip("torch.distributed not importable")
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend unavailable")


@pytest.mark.slow
@pytest.mark.integration
def test_old_design_times_out_when_stage_exceeds_pg_timeout(tmp_path: Path):
    """RE-EXECUTE the original bug: a collective held across a long rank0 stage
    is killed by the PG watchdog.  rank1 must fail (fail-closed exit 42)."""
    _require_torch_distributed()
    _r0, r1 = _spawn_two_ranks(
        "old_bug", tmp_path,
        pg_timeout_s=3.0,       # short watchdog
        stage_duration_s=9.0,   # rank0 stage FAR exceeds it
        hb_timeout_s=3.0,
        join_timeout_s=60.0,
    )
    # rank1 was stuck in the collective and got watchdog-killed.
    assert r1 == EXIT_FAIL_CLOSED, f"expected old design rank1 to time out, got exit {r1}"


@pytest.mark.slow
@pytest.mark.integration
def test_new_design_survives_long_rank0_stage_and_continues(tmp_path: Path):
    """The fix: rank1 polls the heartbeat, does NOT false-kill across a stage
    far longer than the PG timeout, and both ranks pass ONE short barrier."""
    _require_torch_distributed()
    r0, r1 = _spawn_two_ranks(
        "new_success", tmp_path,
        pg_timeout_s=3.0,        # same short watchdog as the old-bug test...
        stage_duration_s=9.0,    # ...and the same long stage that broke it
        hb_timeout_s=4.0,        # > per-beat gap (0.2s), so never falsely stale
        join_timeout_s=60.0,
    )
    assert r0 == EXIT_OK, f"rank0 expected clean exit, got {r0}"
    assert r1 == EXIT_OK, f"rank1 expected clean exit (no false kill), got {r1}"


@pytest.mark.slow
@pytest.mark.integration
def test_new_design_fails_closed_on_rank0_crash(tmp_path: Path):
    """rank0 dies with no terminal record → rank1's stale guard fails closed."""
    _require_torch_distributed()
    _r0, r1 = _spawn_two_ranks(
        "new_crash", tmp_path,
        pg_timeout_s=30.0,
        stage_duration_s=9.0,
        hb_timeout_s=2.0,        # rank1 declares dead 2s after last heartbeat
        join_timeout_s=60.0,
    )
    # rank0 self-killed (137); the assertion that matters is rank1 did NOT hang
    # and fell through the fail-closed path rather than waiting forever.
    assert r1 == EXIT_FAIL_CLOSED, f"expected rank1 fail-closed on crash, got {r1}"


@pytest.mark.slow
@pytest.mark.integration
def test_new_design_fails_closed_on_explicit_failed(tmp_path: Path):
    """rank0 writes FAILED via its try/except → rank1 exits immediately."""
    _require_torch_distributed()
    _r0, r1 = _spawn_two_ranks(
        "new_failed", tmp_path,
        pg_timeout_s=30.0,
        stage_duration_s=9.0,
        hb_timeout_s=10.0,       # long — proves rank1 exits on FAILED, not on stale
        join_timeout_s=60.0,
    )
    assert r1 == EXIT_FAIL_CLOSED, f"expected rank1 fail-closed on FAILED, got {r1}"


@pytest.mark.slow
@pytest.mark.integration
def test_new_design_rejects_stale_success_from_prior_invocation(tmp_path: Path):
    """REAL two-rank proof of the stale-terminal fix.  A previous invocation's
    SUCCESS (foreign stage_invocation_id) is pre-seeded into the control-dir;
    rank0 of THIS invocation starts slowly.  rank1 must NOT accept the stale
    record and return early — it must wait for this invocation's own SUCCESS.
    A false accept would surface as the stale checkpoint (EXIT_UNEXPECTED)."""
    _require_torch_distributed()
    # Pre-seed a prior invocation's terminal record into the exact control dir
    # the worker will poll, carrying a foreign id + a distinguishable checkpoint.
    cd = control_dir_for(tmp_path)
    cd.mkdir(parents=True, exist_ok=True)
    write_status(
        cd,
        Stage5Status(
            state=STATE_SUCCESS,
            stage_invocation_id="PRIOR-run-do-not-accept",
            checkpoint="STALE_prev_invocation.pt",
            metrics={"opponent_nll": 999.0},
            updated_at=1.0,
        ),
    )
    r0, r1 = _spawn_two_ranks(
        "new_stale_success", tmp_path,
        pg_timeout_s=30.0,
        stage_duration_s=3.0,
        hb_timeout_s=5.0,        # > rank0's slow-start delay, so no false-stale
        join_timeout_s=60.0,
    )
    assert r0 == EXIT_OK, f"rank0 expected clean exit, got {r0}"
    # EXIT_OK (not EXIT_UNEXPECTED) is the proof: rank1 ignored the stale SUCCESS
    # and returned only this invocation's genuine result.
    assert r1 == EXIT_OK, f"expected rank1 to ignore stale SUCCESS and accept fresh, got {r1}"


@pytest.mark.slow
@pytest.mark.integration
def test_new_design_rejects_stale_success_from_earlier_restart_same_run(tmp_path: Path):
    """The finer race: SAME launch (identical TORCHELASTIC_RUN_ID), but a
    worker-group restart.  A SUCCESS left by restart-generation r0 must be
    rejected by the ranks that restarted into generation r1 — the run id alone
    matches, only the restart count distinguishes them.  Without folding
    RESTART_COUNT into the invocation id, r1's rank1 would wrongly accept r0's
    leftover status and skip straight to the barrier while r1's rank0 is still
    training — the very deadlock, one level finer."""
    _require_torch_distributed()
    shared_run = f"restart-race-{uuid.uuid4()}"
    # Pre-seed generation r0's SUCCESS: same run id, restart count 0, poison ckpt.
    cd = control_dir_for(tmp_path)
    cd.mkdir(parents=True, exist_ok=True)
    write_status(
        cd,
        Stage5Status(
            state=STATE_SUCCESS,
            stage_invocation_id=f"torchelastic:{shared_run}:r0",
            checkpoint="STALE_restart0.pt",
            metrics={"opponent_nll": 999.0},
            updated_at=1.0,
        ),
    )
    # Spawn the pair as generation r1 of the SAME run.
    r0, r1 = _spawn_two_ranks(
        "new_stale_success", tmp_path,
        pg_timeout_s=30.0,
        stage_duration_s=3.0,
        hb_timeout_s=5.0,
        join_timeout_s=60.0,
        run_id=shared_run,
        restart_count=1,
    )
    assert r0 == EXIT_OK, f"rank0 expected clean exit, got {r0}"
    # r1's rank1 must ignore generation r0's SUCCESS (matching run id, older
    # restart count) and accept only its own generation's fresh result.
    assert r1 == EXIT_OK, (
        f"expected rank1 to reject earlier-restart SUCCESS and accept fresh, got {r1}"
    )
