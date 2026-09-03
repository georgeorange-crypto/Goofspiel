"""File-based control-plane for long, rank0-only training stages (Stage5).

Why this exists
---------------
Stage5 (the adaptive/opponent branch) runs ONLY on rank0 and can take hours.
The previous implementation had every non-rank0 rank block inside an NCCL
collective (``broadcast_object`` + ``barrier``) for the *entire* duration of
rank0's Stage5.  NCCL's watchdog kills any collective that does not complete
within its timeout (default 600s), so a Stage5 longer than that timeout
deadlocked the whole job — a duration-dependent, pre-existing bug.

The fix is NOT a bigger NCCL timeout (a future, larger Stage5 would still blow
past any fixed timeout).  Instead, non-rank0 ranks must not hold a collective
across rank0's long stage at all.  rank0 publishes a heartbeat/status file;
non-rank0 ranks poll it for *liveness and the terminal result*, and only after
rank0 reaches a terminal state do all ranks do ONE short collective to realign
the NCCL sequence number before continuing.

Design principle (locked 2026-09-03): liveness detection, not wall-clock
estimation.  Stage5's true duration varies hugely by workload / device /
budget, so non-rank0 must judge "is rank0 still ALIVE and progressing?" — not
"how long should this theoretically take?".  Two independent guards:

  * heartbeat-stale timeout ("how long with no fresh heartbeat before we
    declare rank0 dead") — guards rank0 crash / hang.
  * absolute hard timeout ("fake-alive" guard) — guards the case where rank0
    keeps heartbeating but its logic never terminates.

NCCL's own timeout stays purely as low-level fault protection; it does NOT
carry Stage5's long-task liveness.

All writes use tmp + fsync + os.replace so a reader never observes a partially
written status file (``os.replace`` is atomic on POSIX and Windows).  The
reader is additionally defensive: any unreadable / truncated / malformed
status is treated as "no new information this tick", never as a spurious
SUCCESS or FAILED.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Terminal + transient states rank0 can publish.
# ---------------------------------------------------------------------------
STATE_STARTING = "STARTING"
STATE_RUNNING = "RUNNING"
STATE_SUCCESS = "SUCCESS"
STATE_FAILED = "FAILED"
_TERMINAL_STATES = frozenset({STATE_SUCCESS, STATE_FAILED})

STATUS_FILENAME = "status.json"

# Conservative defaults (all overridable).  heartbeat-stale ~5min, hard ~48h.
DEFAULT_HEARTBEAT_TIMEOUT_S = 300.0
DEFAULT_HARD_TIMEOUT_S = 48.0 * 3600.0
DEFAULT_HEARTBEAT_INTERVAL_S = 15.0
DEFAULT_POLL_INTERVAL_S = 2.0


class Stage5ControlError(RuntimeError):
    """Raised on a non-rank0 rank when rank0 is declared dead or failed.

    ``reason`` is one of: ``"failed"`` (rank0 published an explicit FAILED
    state), ``"heartbeat_stale"`` (no fresh heartbeat within the heartbeat
    timeout — rank0 crashed or hung), or ``"hard_timeout"`` (rank0 kept
    heartbeating but never reached a terminal state).  All three fail closed:
    the peer stops waiting instead of blocking forever.
    """

    def __init__(self, message: str, *, reason: str, status: Stage5Status | None = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.status = status


@dataclass
class Stage5Status:
    """The heartbeat/status record rank0 publishes for its peers.

    ``updated_at`` is rank0's wall-clock ``time.time()`` at write; the poller
    does NOT trust cross-rank clock sync for its liveness decision — it tracks,
    on its own monotonic clock, when it last observed the record *advance*.
    ``rank0_pid`` is diagnostic only (meaningless across nodes), never the
    liveness signal.  On the terminal SUCCESS record, ``checkpoint`` and
    ``metrics`` carry the stage result so peers can return it without touching
    a collective.
    """

    state: str
    run_id: str = ""
    stage_invocation_id: str = ""
    commit: str | None = None
    rank0_pid: int | None = None
    current_step: int = 0
    total_steps: int = 0
    phase: str = ""
    updated_at: float = 0.0
    parent_checkpoint_sha256: str | None = None
    error: str | None = None
    checkpoint: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.state in _TERMINAL_STATES


def control_dir_for(out_dir: str | os.PathLike[str]) -> Path:
    """Directory holding the Stage5 control-plane files for a run."""
    return Path(out_dir) / "stage5_control"


def current_invocation_id() -> str:
    """A token shared by all ranks of THIS launch-generation, unique across them.

    Prevents the stale-terminal race: the same artifact-dir can host several
    Stage5 invocations over its life (crash-resume, salvage, an explicit
    re-run), each leaving a ``status.json`` behind.  Without an identity a fast
    non-rank0 rank could read a *previous* invocation's ``SUCCESS`` (or
    ``FAILED``) and act on it while this invocation's rank0 has not even written
    ``STARTING`` yet — re-opening the very deadlock the control-plane closes.

    All ranks must derive the SAME id from their common launch, without a
    collective and without trusting cross-rank clocks.  torchrun gives us
    exactly that:

      * ``TORCHELASTIC_RUN_ID`` is set identically in every rank's environment
        and equals the rendezvous id, which torchrun auto-assigns as a fresh
        ``uuid4`` per launch when ``--rdzv-id`` is not passed (it is not, in
        ``torchrun_command``).  It distinguishes one *launch* from another.
      * ``TORCHELASTIC_RESTART_COUNT`` distinguishes one *restart-generation*
        from the next WITHIN a launch.  torchelastic keeps ``RUN_ID`` stable
        across a worker-group restart and only bumps this counter, and it
        restarts the WHOLE group together — so after a restart both rank0 and
        the peers come back with the same incremented count.  Folding it in
        means a peer that restarted into generation N rejects the pre-restart
        generation N-1 rank0's leftover status (same run id, older count),
        instead of wrongly accepting it.

    So every rank of one restart-generation agrees on the id, while a different
    launch OR a later restart of the same launch gets a different one — for
    free, from the env, no collective.

    If ``RUN_ID`` is absent (not launched under torchrun), we fall back to a
    per-PROCESS uuid.  That deliberately makes peers disagree, so a peer treats
    any pre-existing status as foreign and FAILS CLOSED rather than trusting a
    record it cannot prove belongs to this invocation.  A run that is genuinely
    single-process never reaches the poller (the heartbeat is disabled and no
    peer waits), so the fallback only ever hardens, never harms.
    """
    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "").strip()
    if run_id:
        # RESTART_COUNT is per worker-group restart; default 0 if unset (e.g. a
        # launcher that sets RUN_ID but not the counter).  A non-integer value
        # is passed through verbatim rather than crashing the id derivation.
        restart = os.environ.get("TORCHELASTIC_RESTART_COUNT", "0").strip() or "0"
        return f"torchelastic:{run_id}:r{restart}"
    return f"pid:{uuid.uuid4()}"


def _status_path(control_dir: str | os.PathLike[str]) -> Path:
    return Path(control_dir) / STATUS_FILENAME


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (tmp + fsync + os.replace).

    Mirrors ``checkpoint._write_text_atomic`` but kept local so the control
    plane is self-contained and independently testable.  ``os.replace`` is
    atomic on POSIX and Windows, so a concurrent reader sees either the old
    file or the fully written new file — never a partial one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def write_status(control_dir: str | os.PathLike[str], status: Stage5Status) -> Path:
    """Atomically publish ``status`` to ``<control_dir>/status.json``."""
    path = _status_path(control_dir)
    _atomic_write_text(path, json.dumps(asdict(status), ensure_ascii=False, indent=2))
    return path


def read_status(control_dir: str | os.PathLike[str]) -> Stage5Status | None:
    """Read the current status, or ``None`` if absent / unreadable / malformed.

    A missing file (rank0 not started yet), a truncated/partial file, or JSON
    that does not carry a recognised ``state`` all return ``None`` — the poller
    treats that as "no new information", never as a terminal verdict.  This is
    the guarantee the partial-write regression test pins.
    """
    path = _status_path(control_dir)
    try:
        raw = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    state = data.get("state")
    if state not in (STATE_STARTING, STATE_RUNNING, STATE_SUCCESS, STATE_FAILED):
        return None
    known = Stage5Status.__dataclass_fields__.keys()
    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    kwargs: dict[str, Any] = {k: data[k] for k in known if k in data and k != "metrics"}
    kwargs["metrics"] = {str(k): float(v) for k, v in metrics.items()}
    return Stage5Status(**kwargs)


class Rank0Heartbeat:
    """rank0's writer: publishes STARTING/RUNNING(+heartbeat)/SUCCESS/FAILED.

    ``running`` is interval-gated by a monotonic clock so calling it every
    training step is cheap: it only writes when ``interval`` has elapsed since
    the last write (or ``force=True`` for sub-phase transitions).  The interval
    is wall-clock based, so the heartbeat cadence is identical whether a step
    takes 0.5s (GPU) or 25s (CPU).
    """

    def __init__(
        self,
        control_dir: str | os.PathLike[str],
        *,
        enabled: bool = True,
        run_id: str = "",
        invocation_id: str | None = None,
        commit: str | None = None,
        total_steps: int = 0,
        parent_checkpoint_sha256: str | None = None,
        interval: float = DEFAULT_HEARTBEAT_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        pid: int | None = None,
    ) -> None:
        self.control_dir = Path(control_dir)
        # When disabled (single-process runs), every method is a no-op and no
        # status file is ever created, so a non-distributed Stage5 is byte- and
        # side-effect-identical to the pre-control-plane implementation.
        self.enabled = bool(enabled)
        self.run_id = run_id
        self.invocation_id = invocation_id if invocation_id is not None else current_invocation_id()
        self.commit = commit
        self.total_steps = int(total_steps)
        self.parent_checkpoint_sha256 = parent_checkpoint_sha256
        self.interval = float(interval)
        self._clock = clock
        self._wall = wall_clock
        self._pid = int(pid if pid is not None else os.getpid())
        self._last_beat: float | None = None

    def _write(self, status: Stage5Status) -> None:
        if not self.enabled:
            return
        status.run_id = self.run_id
        status.stage_invocation_id = self.invocation_id
        status.commit = self.commit
        status.rank0_pid = self._pid
        status.total_steps = self.total_steps
        status.parent_checkpoint_sha256 = self.parent_checkpoint_sha256
        status.updated_at = float(self._wall())
        write_status(self.control_dir, status)
        self._last_beat = self._clock()

    def starting(self, *, phase: str = "startup") -> None:
        self._write(Stage5Status(state=STATE_STARTING, current_step=0, phase=phase))

    def running(self, *, step: int, phase: str, force: bool = False) -> None:
        """Publish a RUNNING heartbeat, but only if the interval has elapsed.

        Pass ``force=True`` at sub-phase boundaries (session generation, tensor
        build, checkpoint write) so a long non-training phase still refreshes
        the heartbeat even though no training step advanced.
        """
        if not force and self._last_beat is not None and (self._clock() - self._last_beat) < self.interval:
            return
        self._write(Stage5Status(state=STATE_RUNNING, current_step=int(step), phase=phase))

    def success(self, *, checkpoint: str | None, metrics: dict[str, float]) -> None:
        self._write(
            Stage5Status(
                state=STATE_SUCCESS,
                current_step=self.total_steps,
                phase="done",
                checkpoint=checkpoint,
                metrics={str(k): float(v) for k, v in metrics.items()},
            )
        )

    def fail(self, *, error: str, step: int = 0, phase: str = "") -> None:
        self._write(Stage5Status(state=STATE_FAILED, current_step=int(step), phase=phase, error=str(error)))


def wait_for_rank0(
    control_dir: str | os.PathLike[str],
    *,
    expect_invocation_id: str | None = None,
    heartbeat_timeout: float = DEFAULT_HEARTBEAT_TIMEOUT_S,
    hard_timeout: float = DEFAULT_HARD_TIMEOUT_S,
    poll_interval: float = DEFAULT_POLL_INTERVAL_S,
    clock: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Stage5Status:
    """Poll the control-plane until rank0 reaches a terminal state.

    Returns the SUCCESS status (carrying ``checkpoint`` + ``metrics``).  Raises
    :class:`Stage5ControlError` — fail-closed, never blocks forever — when:

      * rank0 publishes FAILED (``reason="failed"``);
      * no fresh heartbeat is observed within ``heartbeat_timeout`` seconds,
        measured on the poller's own monotonic clock since it last saw the
        record advance — this also covers "rank0 never wrote anything"
        (``reason="heartbeat_stale"``);
      * total wait exceeds ``hard_timeout`` even while heartbeats stay fresh —
        the fake-alive guard (``reason="hard_timeout"``).

    Liveness is judged by *observed progress* (a change in ``updated_at`` /
    ``current_step`` / ``state``), not by comparing rank0's wall clock to ours,
    so it is robust to cross-rank clock skew.

    ``expect_invocation_id`` closes the stale-terminal race.  The same
    artifact-dir can hold a ``status.json`` left by a PREVIOUS Stage5 invocation
    (crash-resume, salvage, re-run).  When ``expect_invocation_id`` is given,
    any status whose ``stage_invocation_id`` differs is treated EXACTLY like a
    missing file — invisible to the SUCCESS check, the FAILED check, and the
    liveness clock alike — so a peer can never accept a prior invocation's
    terminal record, and instead waits for (or fails closed against) THIS
    invocation's rank0.  ``None`` disables the check (used only by low-level
    unit tests that write and read within one invocation).
    """
    started = clock()
    last_change_at = started
    seen_updated_at: float | None = None
    seen_state: str | None = None
    seen_step: int | None = None

    while True:
        status = read_status(control_dir)
        # A status from another invocation is not ours to act on: drop it to
        # None so it counts as "no record for this invocation yet" — never a
        # stale SUCCESS we wrongly trust nor a stale FAILED we wrongly die on.
        if (
            status is not None
            and expect_invocation_id is not None
            and status.stage_invocation_id != expect_invocation_id
        ):
            status = None
        now = clock()
        if status is not None:
            progressed = (
                status.updated_at != seen_updated_at
                or status.state != seen_state
                or status.current_step != seen_step
            )
            if progressed:
                last_change_at = now
                seen_updated_at = status.updated_at
                seen_state = status.state
                seen_step = status.current_step
            if status.state == STATE_SUCCESS:
                return status
            if status.state == STATE_FAILED:
                raise Stage5ControlError(
                    f"stage5 rank0 reported FAILED: {status.error!r}",
                    reason="failed",
                    status=status,
                )

        if now - last_change_at > heartbeat_timeout:
            raise Stage5ControlError(
                "stage5 rank0 heartbeat stale for "
                f"{now - last_change_at:.1f}s > {heartbeat_timeout}s "
                f"(last observed state={seen_state}, step={seen_step}); failing closed",
                reason="heartbeat_stale",
                status=status,
            )
        if now - started > hard_timeout:
            raise Stage5ControlError(
                f"stage5 exceeded hard timeout {hard_timeout}s while heartbeating "
                "(fake-alive guard); failing closed",
                reason="hard_timeout",
                status=status,
            )
        sleep_fn(poll_interval)
