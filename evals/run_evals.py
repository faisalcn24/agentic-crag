from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from functions.agent import AgentRuntime, run_agent  # noqa: E402
from functions.grounding import ABSTENTION  # noqa: E402
from functions.rag import (  # noqa: E402
    ask_index_with_sources,
    build_index,
    call_model,
    load_documents,
    load_index,
    load_registry,
    retrieve_sources,
    update_registry,
)
from functions.telemetry import log_query_result  # noqa: E402


GOLDEN_FILE = ROOT / "evals" / "datasets" / "golden.jsonl"
ADVERSARIAL_FILE = ROOT / "evals" / "datasets" / "adversarial.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"
DEFAULT_INDEX_ID = "upgrade-eval-corpus"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed Agentic CRAG answers for direct review"
    )
    parser.add_argument(
        "--mode", choices=("single", "agent", "compare"), default="compare"
    )
    parser.add_argument("--index-id", default=DEFAULT_INDEX_ID)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("AGENTIC_CRAG_LLM_PROVIDER", "ollama")
    os.environ.setdefault("OLLAMA_MODEL", "llama3.2:3b")
    os.environ.setdefault("OLLAMA_PLANNER_MODEL", "llama3.2:3b")
    index = ensure_eval_index(args.index_id)
    cases = read_jsonl(GOLDEN_FILE)
    if args.limit is not None:
        cases = cases[: max(0, args.limit)]

    modes = [args.mode] if args.mode != "compare" else ["single", "agent"]
    all_results: list[dict[str, Any]] = []
    for mode in modes:
        for position, case in enumerate(cases, start=1):
            print(
                f"[{mode} {position}/{len(cases)}] {case['id']}: {case['question']}",
                flush=True,
            )
            result = run_golden_case(index, case, mode)
            all_results.append(result)

    adversarial = []
    if args.mode in {"agent", "compare"}:
        adversarial_cases = read_jsonl(ADVERSARIAL_FILE)
        if args.limit is not None:
            adversarial_cases = adversarial_cases[: max(0, args.limit)]
        for position, case in enumerate(adversarial_cases, start=1):
            print(
                f"[adversarial {position}/{len(adversarial_cases)}] {case['id']}",
                flush=True,
            )
            adversarial.append(run_adversarial_case(index, case))

    summary = summarize(all_results, adversarial)
    path = save_results(all_results, adversarial, summary)
    print(json.dumps(summary, indent=2))
    print(f"Detailed results: {path}")


def ensure_eval_index(index_id: str):
    if index_id not in load_registry():
        raw_docs, warnings = load_documents(ROOT / "documents")
        if warnings:
            print("Corpus warnings: " + "; ".join(warnings))
        build_index(raw_docs, index_id)
        update_registry(index_id, ROOT / "documents", raw_docs)
    return load_index(index_id)


def run_golden_case(index, case: dict[str, Any], mode: str) -> dict[str, Any]:
    generation_error = None
    try:
        if mode == "single":
            started = perf_counter()
            response = ask_index_with_sources(index, case["question"])
            log_query_result(
                mode="eval-single",
                iterations=1,
                latency_ms=(perf_counter() - started) * 1000,
            )
            agent_metadata = None
        else:
            response = run_agent(index, case["question"])
            agent_metadata = response.get("agent")
    except Exception as exc:
        generation_error = f"{type(exc).__name__}: {exc}"
        response = {"answer": "", "sources": []}
        agent_metadata = None
    sources = response.get("sources", [])
    expected_sources = set(case.get("expected_source", []))
    completed = (
        agent_metadata.get("termination_reason") == "answered"
        if agent_metadata is not None
        else generation_error is None and ABSTENTION not in response.get("answer", "")
    )
    metrics = answer_source_metrics(
        response.get("answer", ""), sources, expected_sources, completed=completed
    )
    return {
        "id": case["id"],
        "stratum": case["stratum"],
        "mode": mode,
        "question": case["question"],
        "expected_answer": case["expected_answer"],
        "expected_source": case.get("expected_source", []),
        "answer": response.get("answer", ""),
        "sources": sources,
        "agent": agent_metadata,
        "generation_error": generation_error,
        "metrics": metrics,
    }


def answer_source_metrics(
    answer: str,
    sources: list[dict[str, Any]],
    expected_sources: set[str],
    *,
    completed: bool,
) -> dict[str, float | None]:
    if not completed:
        return {
            "expected_supporting_source_recall": None,
            "citation_source_alignment": None,
            "supporting_evidence_contract": None,
        }

    returned_filenames = {
        source.get("filename") for source in sources if source.get("filename")
    }
    expected_recall = (
        1.0
        if not expected_sources
        else len(expected_sources & returned_filenames) / len(expected_sources)
    )
    cited_filenames = set(re.findall(r"\[([^\[\]\n]+)\]", answer))
    citation_alignment = float(
        bool(cited_filenames) and cited_filenames == returned_filenames
    )
    evidence_contract = float(
        bool(sources)
        and all(
            "score" not in source
            and isinstance(source.get("passages"), list)
            and bool(source["passages"])
            and all(
                isinstance(passage, dict)
                and bool(str(passage.get("text", "")).strip())
                for passage in source["passages"]
            )
            for source in sources
        )
    )
    return {
        "expected_supporting_source_recall": expected_recall,
        "citation_source_alignment": citation_alignment,
        "supporting_evidence_contract": evidence_contract,
    }


def run_adversarial_case(index, case: dict[str, Any]) -> dict[str, Any]:
    injected_context = case.get("injected_context", "")
    if injected_context:
        runtime = AgentRuntime(
            retrieve=lambda query, top_k: [
                *retrieve_sources(index, query, top_k=top_k),
                {
                    "filename": "adversarial-test-document.txt",
                    "type": "test-fixture",
                    "score": 1.0,
                    "text": injected_context,
                },
            ],
            complete=lambda prompt, node, timeout: call_model(
                prompt, node=node, timeout=timeout
            ),
        )
        response = run_agent(index, case["question"], runtime=runtime)
    else:
        response = run_agent(index, case["question"])
    return {
        **case,
        "answer": response["answer"],
        "agent": response["agent"],
        "review_status": "pending_manual_review",
    }


def summarize(
    results: list[dict[str, Any]], adversarial: list[dict[str, Any]]
) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_mode[row["mode"]].append(row)
    modes = {}
    metric_coverage = {}
    for mode, rows in by_mode.items():
        metric_names = sorted({name for row in rows for name in row["metrics"]})
        modes[mode] = {
            name: _mean([row["metrics"].get(name) for row in rows])
            for name in metric_names
        }
        metric_coverage[mode] = {
            name: sum(row["metrics"].get(name) is not None for row in rows)
            for name in metric_names
        }
        modes[mode]["cases"] = len(rows)
    comparison = {}
    if "single" in modes and "agent" in modes:
        for metric in set(modes["single"]) & set(modes["agent"]):
            if (
                metric == "cases"
                or modes["single"][metric] is None
                or modes["agent"][metric] is None
            ):
                continue
            delta_points = (modes["agent"][metric] - modes["single"][metric]) * 100
            comparison[metric] = {"delta_points": delta_points}
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "review_status": "pending_manual_review",
        "modes": modes,
        "metric_coverage": metric_coverage,
        "comparison": comparison,
        "adversarial": {
            "cases": len(adversarial),
            "review_status": "pending_manual_review",
        },
    }


def save_results(
    results: list[dict[str, Any]],
    adversarial: list[dict[str, Any]],
    summary: dict[str, Any],
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"eval-{stamp}.json"
    path.write_text(
        json.dumps(
            {"summary": summary, "golden": results, "adversarial": adversarial},
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None and not math.isnan(value)]
    return sum(present) / len(present) if present else None


if __name__ == "__main__":
    main()
