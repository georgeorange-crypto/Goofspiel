"""F1–F4 measurement drivers (guarded; NOT executed during harness build).

These are the entry points that actually invoke the Arena runners and time them:

  * F1 :func:`verify_exact_sha`             — sanity that we are on the exact SHA.
  * F2 :func:`run_stage6_sweep`             — Stage6 league workload sweep.
  * F3 :func:`run_stage7_component_timing`  — Stage7 red-team, 7-component split.
  * F4 :func:`run_evaluate_family_timing`   — evaluate, split by agent family.

Every driver goes through :func:`preflight` first (§4): it refuses to run unless
the worktree is clean *and* sits on the expected committed Arena integration SHA,
and unless the Arena runner is importable.  This is why the module imports no
Arena code at load time — the analysis core stays torch-free and these drivers
import lazily only when a measurement is actually requested.

Nothing here runs during the harness build.  F1–F6 begin only from the clean,
committed Arena integration SHA (Arena rebased/cherry-picked onto the H100-gated
final Phase2 SHA), per the governing directive.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from .provenance import RunProvenance, capture, git_commit, git_is_dirty
from .records import (
    EvaluateBreakdown,
    Stage6Measurement,
    Stage7Breakdown,
    WorkloadPoint,
)


class CalibrationPreflightError(RuntimeError):
    """Raised when the §4 provenance/environment guard fails."""


def _default_runner_probe() -> None:
    """Import the Arena runner lazily; raise ImportError if absent.

    Kept as the default ``runner_probe`` so :func:`preflight` can be unit-tested
    with an injected probe (no Arena checkout required for the guard's tests).
    """
    # Imported here, not at module top, so this file loads torch-free and on a
    # base that has no Arena code (e.g. the calibration/F build base).
    import importlib

    importlib.import_module("goofspiel.arena")


@dataclass(frozen=True)
class Preflight:
    """Result of a passed preflight — carries the provenance to stamp records."""

    provenance: RunProvenance
    expected_sha: str


def preflight(
    repo: str | Path,
    expected_sha: str,
    *,
    device: str,
    world_size: int,
    seed: int,
    n_cards: int,
    checkpoint_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    runner_probe: Callable[[], None] = _default_runner_probe,
    allow_dirty: bool = False,
) -> Preflight:
    """§4 guard: refuse to measure unless clean + on the exact Arena SHA.

    * ``expected_sha`` must be the full 40-char committed Arena integration SHA.
    * The worktree HEAD must equal it, and (unless ``allow_dirty``) be clean.
    * The Arena runner must import (``runner_probe``).

    A formal calibration measurement must satisfy all three; ``allow_dirty`` exists
    only for local DEV probing and produces a record tagged DEV / NON-BINDING.
    """
    head = git_commit(repo)
    if head != expected_sha:
        raise CalibrationPreflightError(
            f"HEAD {head} != expected Arena integration SHA {expected_sha}; "
            "F measures only from the exact committed Arena SHA"
        )
    dirty = git_is_dirty(repo)
    if dirty and not allow_dirty:
        raise CalibrationPreflightError(
            "worktree is dirty; a formal calibration measurement requires a clean "
            "tree (dirty=false). Pass allow_dirty=True only for a DEV probe."
        )
    try:
        runner_probe()
    except Exception as exc:  # noqa: BLE001 - surface any import failure as preflight
        raise CalibrationPreflightError(
            f"Arena runner is not importable on this checkout: {exc!r}. "
            "F runs only against the integrated Arena SHA."
        ) from exc

    prov = capture(
        repo,
        device=device,
        world_size=world_size,
        seed=seed,
        n_cards=n_cards,
        checkpoint_path=checkpoint_path,
        config=config,
    )
    return Preflight(provenance=prov, expected_sha=expected_sha)


def verify_exact_sha(repo: str | Path, expected_sha: str) -> None:
    """F1: assert the checkout is exactly ``expected_sha`` (no measurement)."""
    head = git_commit(repo)
    if head != expected_sha:
        raise CalibrationPreflightError(
            f"F1 SHA sanity failed: HEAD {head} != {expected_sha}"
        )


@contextmanager
def _timed() -> Iterator[Callable[[], float]]:
    """Context manager yielding a callable that returns elapsed wall seconds."""
    start = time.perf_counter()
    done: dict[str, float] = {}

    def elapsed() -> float:
        return done.get("t", time.perf_counter() - start)

    try:
        yield elapsed
    finally:
        done["t"] = time.perf_counter() - start


# --------------------------------------------------------------------------- #
# The F2/F3/F4 drivers accept an injected ``run_fn`` that performs the actual
# Arena call and returns the raw counts.  This keeps the driver logic (timing,
# provenance stamping, record assembly) unit-testable with a stub, while the real
# invocation is supplied only at execution time from the integrated Arena SHA.
# None of these are called during the harness build.
# --------------------------------------------------------------------------- #
def run_stage6_sweep(
    preflight_result: Preflight,
    workload_points: Sequence[dict[str, Any]],
    run_fn: Callable[[dict[str, Any]], Stage6Measurement],
) -> list[Stage6Measurement]:
    """F2: execute the Stage6 league at each workload point, return measurements.

    ``run_fn`` is the Arena bridge (supplied at execution time); it receives one
    workload spec and returns a fully-populated :class:`Stage6Measurement`.  This
    function does not early-stop and runs every point (no sequential stopping).
    """
    if len(workload_points) < 2:
        raise CalibrationPreflightError(
            "F2 requires a sweep of >=2 workload points (the cost model needs a "
            "slope); got %d" % len(workload_points)
        )
    out: list[Stage6Measurement] = []
    for spec in workload_points:
        out.append(run_fn(spec))
    return out


def run_stage7_component_timing(
    preflight_result: Preflight,
    run_fn: Callable[[], Stage7Breakdown],
) -> Stage7Breakdown:
    """F3: run the Stage7 red-team once, returning the 7-component breakdown."""
    breakdown = run_fn()
    return breakdown


def run_evaluate_family_timing(
    preflight_result: Preflight,
    run_fn: Callable[[], EvaluateBreakdown],
) -> EvaluateBreakdown:
    """F4: run evaluate once, returning the per-family timing breakdown."""
    return run_fn()


def stage6_to_points(
    measurements: Sequence[Stage6Measurement], *, n_cards: int
) -> list[WorkloadPoint]:
    """Adapt Stage6 measurements into fit inputs for F5."""
    return [m.as_workload_point(n_cards=n_cards) for m in measurements]


__all__ = [
    "CalibrationPreflightError",
    "Preflight",
    "preflight",
    "verify_exact_sha",
    "run_stage6_sweep",
    "run_stage7_component_timing",
    "run_evaluate_family_timing",
    "stage6_to_points",
]
