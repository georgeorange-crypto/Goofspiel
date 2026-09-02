"""Dual-channel training run logger.

One object, two synchronised channels:

  ① human-readable  — a named ``logging.Logger("goofspiel.training")`` with a
     console ``StreamHandler`` and a ``FileHandler(artifact_dir/run.log)``.
  ② structured      — a ``JsonlEventSink(artifact_dir/events/run.jsonl)`` that
     receives one ``BaseEvent`` per logged action.

Every semantic method writes BOTH a human line and a JSONL event, so the console
tail and the machine-parseable event stream never drift out of sync.

Design constraints:
  * **rank0-only** — constructed with ``is_rank0``; on non-rank0 ranks every
    method is a no-op, so multi-GPU runs neither spam the console nor race on the
    log files.
  * **host-safe** — uses a *named* logger with explicit handlers and
    ``propagate=False``; it never touches the root logger or ``basicConfig``.
  * **idempotent** — reconstructing a logger for the same ``artifact_dir`` clears
    its handlers first, so re-entry never doubles every line.
  * **optional everywhere** — stages accept ``logger: TrainingLogger | None`` and
    default to ``None``; a stage run standalone simply logs nothing.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .events import BaseEvent, Severity, emit_exception_event
from .jsonl_sink import JsonlEventSink
from .system_metrics import collect_system_metrics

_LOGGER_NAME = "goofspiel.training"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(stage)-18s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


class _StageFilter(logging.Filter):
    """Guarantees every record carries a ``stage`` attribute for the formatter."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        if not hasattr(record, "stage"):
            record.stage = "-"
        return True


class TrainingLogger:
    """Dual-channel (human + JSONL) logger for a single training run."""

    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        is_rank0: bool = True,
        run_id: str = "local",
        console: bool = True,
    ) -> None:
        self.is_rank0 = bool(is_rank0)
        self.run_id = str(run_id)
        self.artifact_dir = Path(artifact_dir)
        self.run_log_path = self.artifact_dir / "run.log"
        self.event_log_path = self.artifact_dir / "events" / "run.jsonl"
        self._stage_started_at: dict[str, float] = {}
        self._run_started_at: float | None = None

        # Non-rank0 ranks stay completely silent: no handlers, no sink, no files.
        if not self.is_rank0:
            self._logger: logging.Logger | None = None
            self._sink: JsonlEventSink | None = None
            return

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.run_log_path.unlink(missing_ok=True)
        self.event_log_path.unlink(missing_ok=True)
        self._sink = JsonlEventSink(self.event_log_path)

        logger = logging.getLogger(_LOGGER_NAME)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False  # never leak into the host's root logger
        # Idempotent: drop any handlers a prior construction installed so we do
        # not emit each line twice on re-entry.
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

        formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
        stage_filter = _StageFilter()

        file_handler = logging.FileHandler(self.run_log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(stage_filter)
        logger.addHandler(file_handler)

        if console:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(formatter)
            stream_handler.addFilter(stage_filter)
            logger.addHandler(stream_handler)

        self._logger = logger

    # ------------------------------------------------------------------ core --
    def _emit(
        self,
        event_type: str,
        *,
        severity: str = Severity.INFO.value,
        step: int | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        if self._sink is None:
            return
        self._sink.emit(
            BaseEvent(
                event_type=event_type,
                severity=severity,
                run_id=self.run_id,
                step=step,
                payload=dict(payload or {}),
            )
        )

    def _log(self, level: int, stage: str, msg: str) -> None:
        if self._logger is None:
            return
        self._logger.log(level, msg, extra={"stage": stage})

    # --------------------------------------------------------------- run-level --
    def run_start(self, config: Mapping[str, Any]) -> None:
        self._run_started_at = time.perf_counter()
        cfg = _jsonable(config)
        stage = str(cfg.get("stage", "?"))
        self._log(
            logging.INFO,
            "run",
            f"RUN START stage={stage} artifact_dir={self.artifact_dir}",
        )
        self._emit("RUN_START", payload={"config": cfg})

    def run_end(self, summary: Mapping[str, Any]) -> None:
        elapsed = None
        if self._run_started_at is not None:
            elapsed = round(time.perf_counter() - self._run_started_at, 3)
        summ = _jsonable(summary)
        self._log(
            logging.INFO,
            "run",
            f"RUN END ok={summ.get('ok')} elapsed_s={elapsed} "
            f"stages={summ.get('stages_run')}",
        )
        self._emit("RUN_END", payload={"summary": summ, "elapsed_s": elapsed})

    # ------------------------------------------------------------- stage-level --
    def stage_start(
        self,
        stage: str,
        *,
        init_from: str | None = None,
        inherited: str | None = None,
    ) -> None:
        self._stage_started_at[stage] = time.perf_counter()
        detail = ""
        if inherited:
            detail = f" theta<-{inherited}"
            if init_from:
                detail += f" ({init_from})"
        self._log(logging.INFO, stage, f"STAGE START{detail}")
        self._emit(
            "STAGE_START",
            payload={"stage": stage, "init_from": init_from, "inherited_from": inherited},
        )

    def stage_end(
        self,
        stage: str,
        *,
        ok: bool,
        metrics: Mapping[str, Any] | None = None,
        checkpoint: str | None = None,
        elapsed_s: float | None = None,
    ) -> None:
        if elapsed_s is None and stage in self._stage_started_at:
            elapsed_s = round(time.perf_counter() - self._stage_started_at[stage], 3)
        self._log(
            logging.INFO,
            stage,
            f"STAGE END ok={ok} elapsed_s={elapsed_s} checkpoint={checkpoint}",
        )
        self._emit(
            "STAGE_END",
            severity=Severity.INFO.value if ok else Severity.ERROR.value,
            payload={
                "stage": stage,
                "ok": bool(ok),
                "metrics": _jsonable(metrics or {}),
                "checkpoint": checkpoint,
                "elapsed_s": elapsed_s,
            },
        )

    # --------------------------------------------------------------- wiring ----
    def theta_wired(
        self,
        child: str,
        parent: str,
        *,
        init_ckpt: str | None = None,
        produced_ckpt: str | None = None,
    ) -> None:
        self._log(
            logging.INFO,
            child,
            f"THETA WIRED {parent} -> {child}  init={init_ckpt}",
        )
        self._emit(
            "THETA_WIRED",
            payload={
                "child": child,
                "parent": parent,
                "init_checkpoint": init_ckpt,
                "produced_checkpoint": produced_ckpt,
            },
        )

    def step_metrics(
        self,
        stage: str,
        step: int,
        total: int | None,
        metrics: Mapping[str, Any],
    ) -> None:
        m = _jsonable(metrics)
        pretty = " ".join(f"{k}={_fmt(v)}" for k, v in m.items())
        total_str = f"/{total}" if total is not None else ""
        self._log(logging.INFO, stage, f"step {step}{total_str}  {pretty}")
        self._emit(
            "STEP_METRICS",
            step=step,
            payload={"stage": stage, "step": step, "total": total, "metrics": m},
        )

    def checkpoint_saved(
        self,
        stage: str,
        path: str | Path,
        *,
        global_step: int | None = None,
        sha256: str | None = None,
    ) -> None:
        self._log(
            logging.INFO,
            stage,
            f"CHECKPOINT saved step={global_step} → {path}",
        )
        self._emit(
            "CHECKPOINT_SAVED",
            step=global_step,
            payload={
                "stage": stage,
                "path": str(path),
                "global_step": global_step,
                "sha256": sha256,
            },
        )

    def resume_stage(
        self,
        stage: str,
        *,
        completed_step: int,
        next_step: int,
        total_steps: int,
        checkpoint: str | Path,
    ) -> None:
        display_stage = "Stage4" if stage == "stage4_robust_rl" else stage
        msg = (
            f"resume {display_stage}: checkpoint completed_step={completed_step} "
            f"-> continuing at step={next_step} / {total_steps}"
        )
        self._log(logging.INFO, stage, msg)
        self._emit(
            "STAGE_RESUME",
            step=next_step,
            payload={
                "stage": stage,
                "completed_step": int(completed_step),
                "next_step": int(next_step),
                "total_steps": int(total_steps),
                "checkpoint": str(checkpoint),
            },
        )

    # -------------------------------------------------------------- lineage ----
    def lineage_verdict(
        self,
        consistent: bool,
        *,
        inconsistencies: Sequence[Any] | None = None,
        order: Sequence[str] | None = None,
    ) -> None:
        inc = list(inconsistencies or [])
        self._log(
            logging.INFO if consistent else logging.ERROR,
            "lineage",
            f"LINEAGE consistent={consistent} inconsistencies={len(inc)}",
        )
        self._emit(
            "LINEAGE_VERDICT",
            severity=Severity.INFO.value if consistent else Severity.ERROR.value,
            payload={
                "consistent": bool(consistent),
                "inconsistencies": _jsonable(inc),
                "order": list(order or []),
            },
        )

    # -------------------------------------------------------------- system -----
    def system_metrics(self, label: str = "system") -> None:
        try:
            metrics = collect_system_metrics()
        except Exception as exc:  # never let telemetry crash a run
            self.warn(label, f"system metrics unavailable: {exc}")
            return
        m = _jsonable(metrics)
        self._log(logging.DEBUG, label, f"SYSTEM {m}")
        self._emit("SYSTEM_METRICS", payload={"label": label, "metrics": m})

    # ------------------------------------------------------------- diagnostics -
    def warn(self, stage: str, msg: str) -> None:
        self._log(logging.WARNING, stage, msg)
        self._emit("WARNING", severity=Severity.WARNING.value, payload={"stage": stage, "message": msg})

    def error(self, stage: str, msg: str, exc: BaseException | None = None) -> None:
        self._log(logging.ERROR, stage, msg)
        if exc is not None and self._sink is not None:
            event = emit_exception_event(exc)
            event.run_id = self.run_id
            event.payload.setdefault("stage", stage)
            event.payload.setdefault("message", msg)
            self._sink.emit(event)
        else:
            self._emit("ERROR", severity=Severity.ERROR.value, payload={"stage": stage, "message": msg})

    # ------------------------------------------------------------- lifecycle ---
    def close(self) -> None:
        if self._logger is None:
            return
        for handler in list(self._logger.handlers):
            self._logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    @property
    def event_count(self) -> int:
        return self._sink.count() if self._sink is not None else 0


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _jsonable(obj: Any) -> Any:
    """Best-effort coercion to JSON-serialisable primitives for the event payload."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    # dataclasses / namedtuples / arbitrary objects → stringify defensively.
    for attr in ("_asdict", "__dict__"):
        holder = getattr(obj, attr, None)
        if callable(holder):
            try:
                return _jsonable(holder())
            except Exception:
                break
        if isinstance(holder, dict):
            return _jsonable(holder)
    return str(obj)
