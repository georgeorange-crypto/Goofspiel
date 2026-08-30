from __future__ import annotations

from goofspiel.observability import BaseEvent, JsonlEventSink, MetricAggregator, collect_system_metrics


def test_jsonl_event_sink_writes_structured_event(tmp_path):
    sink = JsonlEventSink(tmp_path / "events" / "learner.jsonl")
    sink.emit(BaseEvent(event_type="SEARCH_COMPLETED", state_hash="abc", model_version="m1"))
    assert sink.count() == 1


def test_metric_aggregator_keeps_low_frequency_summaries():
    agg = MetricAggregator()
    agg.add("robust/q_loss", 2.0)
    agg.add("robust/q_loss", 4.0)
    [summary] = agg.summaries()
    assert summary.name == "robust/q_loss"
    assert summary.mean == 3.0


def test_system_metrics_probe_is_nonfatal():
    metrics = collect_system_metrics()
    assert isinstance(metrics, dict)
