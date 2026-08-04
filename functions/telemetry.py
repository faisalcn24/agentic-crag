from __future__ import annotations

import json
import os
import statistics
import threading
from datetime import UTC, datetime
from pathlib import Path


_write_lock = threading.Lock()


def telemetry_file() -> Path:
    storage = Path(
        os.getenv("AGENTIC_CRAG_STORAGE_DIR", Path.home() / ".agentic_crag_data")
    )
    return storage.expanduser() / "metrics" / "queries.jsonl"


def log_query_result(*, mode: str, iterations: int, latency_ms: float) -> None:
    path = telemetry_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "mode": mode,
        "iterations": iterations,
        "latency_ms": round(latency_ms, 3),
    }
    with _write_lock, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def read_events() -> list[dict]:
    path = telemetry_file()
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def summarize_events() -> dict:
    events = read_events()
    latencies = sorted(float(event["latency_ms"]) for event in events)
    iterations = [int(event["iterations"]) for event in events]
    return {
        "queries": len(events),
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "mean_iterations": statistics.fmean(iterations) if iterations else None,
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return values[int((len(values) - 1) * fraction)]


if __name__ == "__main__":
    print(json.dumps(summarize_events(), indent=2))
