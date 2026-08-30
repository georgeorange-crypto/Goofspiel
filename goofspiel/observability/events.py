"""Canonical structured event schemas."""

from __future__ import annotations

import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class BaseEvent:
    event_type: str
    severity: str = Severity.INFO.value
    run_id: str = "local"
    step: int | None = None
    state_hash: str | None = None
    model_version: str | None = None
    game_id: str | None = None
    session_id: str | None = None
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)


def emit_exception_event(
    exc: BaseException,
    *,
    event_type: str = "EXCEPTION",
    severity: str = Severity.ERROR.value,
    state_hash: str | None = None,
    model_version: str | None = None,
) -> BaseEvent:
    return BaseEvent(
        event_type=event_type,
        severity=severity,
        state_hash=state_hash,
        model_version=model_version,
        payload={"exception_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
    )
