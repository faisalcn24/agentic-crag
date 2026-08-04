from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import evals.run_evals as runner


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


def test_single_case_uses_current_telemetry_api(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_CRAG_STORAGE_DIR", str(tmp_path))
    monkeypatch.setattr(
        runner,
        "ask_index_with_sources",
        lambda *_args: {"answer": "answer", "sources": []},
    )
    case = {
        "id": "smoke",
        "stratum": "single-hop",
        "question": "question",
        "expected_answer": "answer",
        "expected_source": [],
        "answer_should_exist": True,
    }

    assert runner.run_golden_case(object(), case, "single")["generation_error"] is None


def test_agent_case_preserves_answers_and_sources_for_direct_review(monkeypatch):
    source = {"filename": "requirements.docx", "text": "The value is 42."}
    monkeypatch.setattr(
        runner,
        "run_agent",
        lambda *_args: {
            "answer": "The value is 42 [requirements.docx].",
            "sources": [source],
            "agent": {"termination_reason": "answered"},
        },
    )
    case = {
        "id": "manual-review",
        "stratum": "single-hop",
        "question": "What is the value?",
        "expected_answer": "42",
        "expected_source": ["requirements.docx"],
        "answer_should_exist": True,
    }

    result = runner.run_golden_case(object(), case, "agent")

    assert result["expected_answer"] == "42"
    assert result["answer"] == "The value is 42 [requirements.docx]."
    assert result["sources"] == [source]
    assert result["metrics"] == {"expected_source_recall": 1.0}


def test_false_premise_answer_is_left_for_manual_review(monkeypatch):
    monkeypatch.setattr(
        runner,
        "run_agent",
        lambda *_args: {
            "answer": "The false premise is accepted.",
            "agent": {"termination_reason": "answered"},
        },
    )
    result = runner.run_adversarial_case(
        object(),
        {
            "expected_behavior": "refuse_or_correct",
            "question": "false premise",
            "injected_context": "",
        },
    )
    summary = runner.summarize([], [result])

    assert summary["adversarial"] == {
        "cases": 1,
        "review_status": "pending_manual_review",
    }


def test_injection_case_is_preserved_for_direct_review(monkeypatch):
    monkeypatch.setattr(
        runner,
        "run_agent",
        lambda *_args, **_kwargs: {
            "answer": "Prompt injection was detected. The answer is not present.",
            "agent": {"prompt_injection_flagged": True},
        },
    )
    case = {
        "expected_behavior": "flag_and_answer",
        "question": "question",
        "injected_context": "malicious context",
    }

    result = runner.run_adversarial_case(object(), case)

    assert result["answer"] == "Prompt injection was detected. The answer is not present."
    assert result["agent"]["prompt_injection_flagged"] is True
    assert result["review_status"] == "pending_manual_review"


def test_summary_reports_metric_coverage():
    summary = runner.summarize(
        [
            {"mode": "single", "metrics": {"score": 1.0, "nullable": None}},
            {"mode": "single", "metrics": {"score": 0.0, "nullable": 0.5}},
        ],
        [],
    )

    assert summary["metric_coverage"] == {
        "single": {"nullable": 1, "score": 2},
    }
    assert summary["review_status"] == "pending_manual_review"


def test_saved_results_are_marked_for_direct_review(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    summary = runner.summarize([], [])

    path = runner.save_results([], [], summary)
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert saved["summary"]["review_status"] == "pending_manual_review"
    assert saved["golden"] == []
    assert saved["adversarial"] == []
