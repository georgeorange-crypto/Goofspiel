"""Structured events, metrics, and artifact logging."""

from .events import BaseEvent, Severity, emit_exception_event
from .jsonl_sink import JsonlEventSink
from .metric_aggregator import MetricAggregator
from .system_metrics import collect_system_metrics

__all__ = [
    "BaseEvent",
    "JsonlEventSink",
    "MetricAggregator",
    "Severity",
    "collect_system_metrics",
    "emit_exception_event",
]
