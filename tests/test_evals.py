from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import evals.run_evals as runner
from evals.run_evals import _install_ragas_vertexai_compatibility_alias


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(name: str) -> list[dict]:
    path = ROOT / "evals" / "datasets" / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_golden_dataset_has_fixed_stratification():
    rows = read_jsonl("golden.jsonl")

    assert len(rows) == 60
    assert Counter(row["stratum"] for row in rows) == {
        "single-hop": 20,
        "multi-hop": 15,
        "spreadsheet lookup": 15,
        "unanswerable": 10,
    }


def test_adversarial_dataset_has_twenty_cases():
    rows = read_jsonl("adversarial.jsonl")

    assert len(rows) == 20
    assert Counter(row["case_type"] for row in rows) == {
        "false premise": 6,
        "out-of-scope": 6,
        "prompt injection": 8,
    }


def test_pinned_evaluation_framework_metric_imports():
    _install_ragas_vertexai_compatibility_alias()

    from deepeval.metrics import GEval, HallucinationMetric
    from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness

    assert all((GEval, HallucinationMetric, ContextPrecision, ContextRecall, Faithfulness))


def test_single_case_uses_current_telemetry_api(tmp_path, monkeypatch):
    monkeypatch.setenv("INSIGHT_STORAGE_DIR", str(tmp_path))
    monkeypatch.setattr(runner, "setup_llm", lambda: None)
    monkeypatch.setattr(runner, "ask_index_with_sources", lambda *_args: {"answer": "answer", "sources": []})
    case = {
        "id": "smoke",
        "stratum": "single-hop",
        "question": "question",
        "expected_answer": "answer",
        "expected_source": [],
        "answer_should_exist": True,
    }

    assert runner.run_golden_case(object(), case, "single")["generation_error"] is None


def test_unjudged_false_premise_answer_does_not_auto_pass(monkeypatch):
    monkeypatch.setattr(runner, "run_agent", lambda *_args: {
        "answer": "The false premise is accepted.",
        "agent": {"termination_reason": "answered"},
    })
    result = runner.run_adversarial_case(object(), {
        "expected_behavior": "refuse_or_correct",
        "question": "false premise",
        "injected_context": "",
    })
    summary = runner.summarize([], [result], judged=False)

    assert summary["adversarial"] == {
        "cases": 1,
        "scored_cases": 0,
        "behavior_pass_rate": None,
    }
