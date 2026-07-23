from __future__ import annotations

from functions.telemetry import (
    log_query_result,
    read_events,
    summarize_events,
)


def test_telemetry_records_only_latency_and_iterations(tmp_path, monkeypatch):
    monkeypatch.setenv("INSIGHT_STORAGE_DIR", str(tmp_path))

    log_query_result(mode="agent", iterations=2, latency_ms=45.6789)

    events = read_events()
    assert set(events[0]) == {"timestamp", "mode", "iterations", "latency_ms"}
    assert summarize_events() == {
        "queries": 1,
        "p50_latency_ms": 45.679,
        "p95_latency_ms": 45.679,
        "mean_iterations": 2.0,
    }
