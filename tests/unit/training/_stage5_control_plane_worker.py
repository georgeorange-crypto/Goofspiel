"""Worker entrypoint spawned as REAL OS processes by the Stage5 control-plane
regression test (``test_stage5_control_plane.py``).

This is NOT a test module — pytest must not collect it.  It is executed as
``python -m ... _stage5_control_plane_worker`` under a real ``gloo`` process
group so the regression test exercises the actual distributed code path
(per the testing principle: RE-EXECUTE the fact, never monkeypatch the PG).

Scenarios (selected by argv[1]):

  old_bug        Reproduce the pre-fix deadlock: rank1 holds an NCCL/gloo
                 collective (broadcast_object) across a rank0-only stage that
                 runs longer than the process-group timeout.  With a short PG
                 timeout, rank1 is expected to RAISE (watchdog kill) — proving
                 the old design is duration-fragile.

  new_success    The fixed design: rank0 heartbeats through a long stage while
                 rank1 polls the control-plane (NO collective held).  rank1
                 must NOT false-kill; after rank0's SUCCESS both ranks do ONE
                 short barrier and exit 0 together.

  new_crash      rank0 dies WITHOUT writing a terminal status (simulated crash,
                 no FAILED record).  rank1's heartbeat-stale guard must fire and
                 it must exit with the sentinel code, never hang.

  new_failed     rank0 hits an exception and its try/except writes FAILED.
                 rank1 must read FAILED and fail closed immediately.

  new_stale_success
                 A PRIOR invocation's SUCCESS status.json is already on disk
                 (pre-seeded by the parent with a foreign stage_invocation_id).
                 rank0 of THIS invocation starts slowly, so rank1 spends a real
                 polling window seeing only the stale record.  rank1 must IGNORE
                 it (foreign id) and return only once THIS invocation reaches
                 SUCCESS — proving a re-run into a used artifact-dir cannot
                 accept a previous run's terminal record.

Exit codes let the parent assert precisely: 0 = clean, 42 = fail-closed
(Stage5ControlError), 43 = unexpected exception, 44 = rank1 wrongly succeeded
when it should have failed.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import os
import sys
import time
from pathlib import Path

EXIT_OK = 0
EXIT_FAIL_CLOSED = 42
EXIT_UNEXPECTED = 43
EXIT_SHOULD_HAVE_FAILED = 44


def _init_pg(timeout_s: float):
    import torch.distributed as dist

    # gloo is CPU-capable, device-agnostic, and (unlike the default nccl path)
    # honours an explicit short timeout on all platforms — exactly what we need
    # to reproduce "a collective held longer than the PG timeout gets killed".
    dist.init_process_group(
        backend="gloo",
        timeout=_dt.timedelta(seconds=timeout_s),
    )
    return dist


def _run() -> int:
    scenario = sys.argv[1]
    control_root = Path(sys.argv[2])
    rank = int(os.environ["RANK"])
    pg_timeout = float(os.environ.get("PG_TIMEOUT_S", "5"))
    stage_duration = float(os.environ.get("STAGE_DURATION_S", "12"))
    hb_timeout = float(os.environ.get("HB_TIMEOUT_S", "3"))

    from goofspiel.training.stage_control import (
        Rank0Heartbeat,
        Stage5ControlError,
        control_dir_for,
        current_invocation_id,
        wait_for_rank0,
    )

    control_dir = control_dir_for(control_root)
    control_dir.mkdir(parents=True, exist_ok=True)
    # All ranks of this launch share TORCHELASTIC_RUN_ID (set by the parent to
    # mirror torchrun), so both ranks derive the SAME invocation id from the env
    # — rank0 stamps it on every status write, rank1 only accepts a status that
    # carries it.  A pre-seeded status from a "prior invocation" carries a
    # different id and must be ignored.
    invocation_id = current_invocation_id()

    # ---- OLD design: rank1 blocks in a collective for the whole rank0 stage ----
    if scenario == "old_bug":
        from goofspiel.training.distributed import broadcast_object

        dist = _init_pg(pg_timeout)
        try:
            if rank == 0:
                # rank0-only long stage — longer than the PG timeout.
                time.sleep(stage_duration)
                broadcast_object({"checkpoint": "x", "metrics": {}}, src=0)
                dist.barrier()
                return EXIT_OK
            else:
                # rank1 enters the collective at stage START and is stuck in it
                # for the whole rank0 stage -> watchdog kills it once the stage
                # exceeds the PG timeout.
                broadcast_object(None, src=0)
                dist.barrier()
                return EXIT_SHOULD_HAVE_FAILED  # if we get here, no timeout fired
        except Exception:
            return EXIT_FAIL_CLOSED
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    # ---- NEW design: heartbeat control-plane, no collective held across stage ----
    dist = _init_pg(pg_timeout)
    try:
        if rank == 0:
            hb = Rank0Heartbeat(
                control_dir, enabled=True, run_id="reg", invocation_id=invocation_id,
                total_steps=100, interval=0.2
            )
            if scenario == "new_stale_success":
                # A prior invocation's SUCCESS is already on disk (pre-seeded by
                # the parent).  Start SLOWLY so rank1 spends a real polling
                # window with ONLY that stale record visible — it must reject it
                # (foreign invocation id) rather than return it.  The delay is
                # kept below hb_timeout so rank1 does not false-declare stale.
                time.sleep(min(2.0, hb_timeout * 0.4))
            hb.starting()
            if scenario == "new_crash":
                # Heartbeat a few times, then die HARD without a terminal record.
                deadline = time.monotonic() + hb_timeout * 0.5
                step = 0
                while time.monotonic() < deadline:
                    step += 1
                    hb.running(step=step, phase="training", force=True)
                    time.sleep(0.2)
                os._exit(137)  # simulate SIGKILL: no FAILED written, PG left dirty
            if scenario == "new_failed":
                try:
                    hb.running(step=1, phase="training", force=True)
                    raise RuntimeError("intentional rank0 failure")
                except BaseException as exc:
                    hb.fail(error=repr(exc))
                    # rank0 exits nonzero; the point is rank1 reads FAILED fast.
                    return EXIT_UNEXPECTED
            # new_success: heartbeat through a stage LONGER than pg_timeout and
            # longer than hb_timeout-per-beat, proving liveness beats duration.
            start = time.monotonic()
            step = 0
            while time.monotonic() - start < stage_duration:
                step += 1
                hb.running(step=step, phase="training", force=True)
                time.sleep(0.2)
            hb.success(checkpoint="stage5_adaptive.pt", metrics={"opponent_nll": 1.0})
            dist.barrier()  # the ONE short realigning collective
            return EXIT_OK
        else:
            # rank1 NEVER holds a collective during the stage; it polls.
            try:
                status = wait_for_rank0(
                    control_dir,
                    expect_invocation_id=invocation_id,
                    heartbeat_timeout=hb_timeout,
                    hard_timeout=stage_duration * 10,
                    poll_interval=0.1,
                )
            except Stage5ControlError:
                return EXIT_FAIL_CLOSED
            # Accepting the pre-seeded prior-invocation SUCCESS would surface
            # here as the stale checkpoint name -> EXIT_UNEXPECTED.  Only THIS
            # invocation's genuine SUCCESS carries "stage5_adaptive.pt".
            if status.checkpoint != "stage5_adaptive.pt":
                return EXIT_UNEXPECTED
            if status.stage_invocation_id != invocation_id:
                return EXIT_UNEXPECTED
            dist.barrier()  # matches rank0's single post-stage barrier
            return EXIT_OK
    except Stage5ControlError:
        return EXIT_FAIL_CLOSED
    except Exception:
        return EXIT_UNEXPECTED
    finally:
        if dist.is_initialized():
            # The crash scenario leaves the PG dirty; suppress teardown errors.
            with contextlib.suppress(Exception):
                dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(_run())
