from __future__ import annotations
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from time import monotonic, perf_counter
from typing import Any, Callable
from .grounding import (
    ABSTENTION,
    _conversational_claims,
    _structured_field_spans,
    classify_answer_intent,
    consolidate_repeated_citations,
    extract_absence_evidence,
    extract_collection_overview,
    extract_grounded_passages,
    extract_grounded_sentence,
    extract_negative_answer,
    extract_source_overview,
    extract_spreadsheet_lookup,
    extract_structured_answer,
    find_grounded_evidence,
    ground_conversational_answer,
    ground_generated_answer,
    is_collection_overview_question,
    overview_covers_core_fields,
)
from .rag import (
    call_model,
    conversation_history_text,
    conversation_plan_prompt,
    estimate_tokens,
    overview_answer_prompt,
    parse_json_object,
    parse_overview_answer,
    referential_overview_prompt,
    retrieve_sources,
    supporting_source_groups,
)
from .telemetry import log_query_result
from .multihop import (
    DecompositionPlan,
    is_multi_hop_question,
    plan_decomposition,
    requires_cross_leg_join,
    _clean_sources,
    _corrective_query_prompt,
    _deterministic_join_queries,
    _distinctive_join_terms,
    _evidence_coverage_prompt,
    _fallback_corrective_queries,
    _focus_tokens,
    _literal_source_quotes,
    _neutralize_prompt_injection,
    _ordered_questions,
    _quote_matches_subquestion,
    _reported_filename_matches,
    _safe_sources,
    _strong_prose_coverage,
    _strong_row_coverage,
    _subquery_labels_match,
    _subquestion_evidence_text,
    _validate_corrective_queries,
    contains_prompt_injection,
)

AgentState = dict[str, Any]


@dataclass(frozen=True)
class AgentConfig:
    timeout_seconds: float = 30.0
    token_limit: int = 12000
    retrieve_top_k: int = 5
    corrective_pass_enabled: bool = True
    multi_hop_enabled: bool = True
    answer_reserve_seconds: float = 8.0

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            timeout_seconds=min(
                max(float(os.getenv("AGENTIC_CRAG_AGENT_TIMEOUT_SECONDS", "30")), 1.0),
                30.0,
            ),
            token_limit=int(os.getenv("AGENTIC_CRAG_AGENT_TOKEN_LIMIT", "12000")),
        )


@dataclass
class AgentRuntime:
    retrieve: Callable[[str, int], list[dict[str, Any]]]
    complete: Callable[[str, str, float], tuple[str, int]]


def _without_trailing_abstention(answer: str) -> str:
    position = answer.casefold().rfind(ABSTENTION.casefold())
    if position <= 0 or answer[position + len(ABSTENTION) :].strip():
        return answer
    substantive = answer[:position].strip()
    return substantive or answer


def _source_key(source: dict[str, Any]) -> tuple[str, str]:
    return (source.get("filename", "unknown"), source.get("text", ""))


def _merge_sources(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(
        {_source_key(source): source for group in groups for source in group}.values()
    )


def _retrieve_queries(
    runtime: AgentRuntime, queries: list[str], top_k: int
) -> list[list[dict[str, Any]]]:
    return [runtime.retrieve(query, top_k) for query in queries]


def budget_fallback(
    sources: list[dict[str, Any]], reason: str, injection_flagged: bool = False
) -> str:
    warning = "Prompt injection was detected and ignored. " if injection_flagged else ""
    if not sources:
        return f"{warning}Low confidence: {ABSTENTION} Agent budget ended ({reason})."
    best = sources[0]
    filename = best.get("filename", "unknown")
    excerpt = " ".join(best.get("text", "").split())[:320].rstrip()
    return f"{warning}Low confidence: the agent budget ended before the answer completed ({reason}). Best-effort evidence: {excerpt} [{filename}]. The answer may be incomplete."


def _fallback_follow_up_query(history: list[dict[str, str]], question: str) -> str:
    return f"{next((turn.get('content', '').strip() for turn in reversed(history) if turn.get('role') == 'user' and turn.get('content', '').strip()), '')} {question}".strip()


def _follow_up_query_lacks_context(
    planned_query: str, history: list[dict[str, str]], question: str
) -> bool:
    if not re.search(
        "\\b(?:it|its|this|that|these|those|they|them|their|there)\\b",
        question,
        re.IGNORECASE,
    ):
        return False
    question_terms, planned_terms = (
        _focus_tokens(question),
        _focus_tokens(planned_query),
    )
    return not planned_terms & (
        _focus_tokens(conversation_history_text(history)) - question_terms
    )


def _remaining_seconds(state: AgentState, config: AgentConfig) -> float:
    return config.timeout_seconds - (monotonic() - state["started_at"])


def _evidence_text(sources: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"SOURCE: {source.get('filename', 'unknown')}\nSOURCE TYPE: {source.get('type', 'unknown')}\n{source.get('text', '')}"
        for source in sources
    )


def _coverage_inputs(
    results: list[dict[str, Any]], *, join_results: list[dict[str, Any]] | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    needs_model_review, missing = [], []
    for item in results:
        question, sources = (
            item.get("question", ""),
            _safe_sources(item.get("sources", [])),
        )
        if not sources:
            missing.append(question)
        elif re.search(r"\bincidents?\b", question, re.IGNORECASE) and not any(
            re.search(r"\bIncident_ID\s*:", source.get("text", ""), re.IGNORECASE)
            for source in sources
        ):
            missing.append(question)
        elif evidence := _strong_row_coverage(
            question,
            sources,
            join_terms=_distinctive_join_terms(join_results or [], question),
        ):
            item["verified_evidence"] = evidence
            item["coverage_method"] = "deterministic"
        elif evidence := _strong_prose_coverage(question, sources):
            item["verified_evidence"] = evidence
            item["coverage_method"] = "deterministic"
        else:
            needs_model_review.append(item)
    return (needs_model_review, missing)


def _review_coverage(
    context: Any,
    state: AgentState,
    results: list[dict[str, Any]],
    *,
    join_results: list[dict[str, Any]] | None = None,
    revalidation: bool = False,
) -> tuple[int, list[str], str | None]:
    review, missing = _coverage_inputs(results, join_results=join_results)
    if (
        review
        and not join_results
        and state.get("dependent_join")
        and any(item.get("verified_evidence") for item in results)
    ):
        join_results = results
        review, additional_missing = _coverage_inputs(review, join_results=join_results)
        missing.extend(additional_missing)
    if not review:
        return (0, missing, None)
    prompt = _evidence_coverage_prompt(
        review, revalidation=revalidation, verified_results=join_results
    )
    text, tokens, failure = context.complete_node(
        state,
        prompt,
        "evidence_coverage",
        1024,
        time_reserve=1.0 if revalidation else 0.0,
    )
    if failure:
        return (
            tokens,
            [*missing, *(item.get("question", "") for item in review)],
            failure,
        )
    missing.extend(
        _validate_evidence_coverage(
            parse_json_object(text or ""), review, join_results=join_results
        )[1]
    )
    return (tokens, missing, None)


def _validate_evidence_coverage(
    data: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    join_results: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[str]]:
    records = data.get("coverage")
    if not isinstance(records, list):
        return ([], [item.get("question", "") for item in results])
    covered, missing = [], []
    for item in results:
        question = item.get("question", "")
        join_terms = _distinctive_join_terms(join_results or [], question)
        matches = [
            record
            for record in records
            if isinstance(record, dict)
            and _subquery_labels_match(record.get("subquery"), question)
        ]
        if len(matches) != 1 or matches[0].get("covered") is not True:
            missing.append(question)
            continue
        filename, quote, quote_matches = (
            matches[0].get("filename"),
            matches[0].get("quote"),
            [],
        )
        if isinstance(filename, str) and isinstance(quote, str) and quote.strip():
            quote_matches = [
                (source.get("filename", "unknown"), exact_quote)
                for source in _safe_sources(item.get("sources", []))
                if _reported_filename_matches(
                    filename, source.get("filename", "unknown")
                )
                for exact_quote in _literal_source_quotes(source.get("text", ""), quote)
                if (not join_terms or len(_focus_tokens(exact_quote) & join_terms) >= 2)
                and _quote_matches_subquestion(question, exact_quote)
            ]
        quote_matches = list(dict.fromkeys(quote_matches))
        if quote_matches and len({match[0] for match in quote_matches}) == 1:
            actual_filename, exact_quote = max(
                quote_matches,
                key=lambda match: (
                    len(_focus_tokens(question) & _focus_tokens(match[1])),
                    -len(match[1]),
                ),
            )
            covered.append(question)
            item["verified_evidence"] = {
                "filename": actual_filename,
                "quote": exact_quote,
            }
            item["coverage_method"] = "model_verified"
        else:
            missing.append(question)
    return (covered, missing)


def _cited_history_filenames(history: list[dict[str, Any]]) -> list[str]:
    turn = next(
        (turn for turn in reversed(history) if turn.get("role") == "assistant"), {}
    )
    available = set(turn.get("source_filenames", []))
    return list(
        dict.fromkeys(
            filename
            for filename in re.findall("\\[([^\\[\\]\\n]+)\\]", turn.get("content", ""))
            if filename in available
        )
    )


def _multi_source_history_overview(
    history: list[dict[str, Any]], sources: list[dict[str, Any]]
) -> str | None:
    prior_question = next(
        (
            str(turn.get("content", "")).strip()
            for turn in reversed(history)
            if turn.get("role") == "user" and turn.get("content")
        ),
        "",
    )
    parts = [
        part.strip().rstrip("?")
        for part in re.split("(?<=[.!?])\\s+", prior_question)
        if part.strip()
    ]
    if not parts:
        return None
    topics = []
    for part in parts[:3]:
        cleaned = re.sub("^then\\s+", "", part, flags=re.IGNORECASE)
        comparison, statement = (
            re.match("^(compare|contrast)\\s+(.+)$", cleaned, re.IGNORECASE),
            re.match("^state\\s+(.+)$", cleaned, re.IGNORECASE),
        )
        topic = (
            comparison.group(2)
            if comparison
            else statement.group(1)
            if statement
            else cleaned
        )
        topics.append(topic[:1].upper() + topic[1:].rstrip(".!?") + ".")
    citations = " ".join(
        (f"[{source.get('filename', 'unknown')}]" for source in sources)
    )
    topics[-1] = f"{topics[-1].rstrip('.')} {citations}."
    return "This covers:\n" + "\n".join((f"- {topic}" for topic in topics))


def _structured_multi_part_answer(answer: str, results: list[dict[str, Any]]) -> str:
    if answer == ABSTENTION:
        return answer
    if re.search(r"(?m)^\s*##\s+", answer):
        return _ensure_structured_section_citations(answer, results)
    claims = [
        re.sub(
            r"^(?:[-*]\s+|(?:while|whereas)\s+)", "", claim.strip(), flags=re.IGNORECASE
        )
        for claim in re.split(r"(?<=[.!?])\s+|\n+|,\s+(?=(?:while|whereas)\b)", answer)
        if claim.strip() and claim.strip().casefold() != "here's the breakdown:"
    ]
    if len(claims) < 2:
        return answer
    records = [
        (
            heading,
            evidence.get("quote", ""),
            evidence.get("filename"),
            _answer_identifiers(f"{question} {evidence.get('quote', '')}"),
            _focus_tokens(evidence.get("quote", "")),
        )
        for item in results
        for question in [item.get("question", "")]
        for evidence in [item.get("verified_evidence", {})]
        for heading in [
            _answer_section_heading(question) or _question_section_heading(question)
        ]
    ]
    headings = list(dict.fromkeys(record[0] for record in records))
    grouped = {heading: [] for heading in headings}
    for index, claim in enumerate(claims):
        semantic = _answer_section_heading(claim)
        claim_text = " ".join(
            re.sub(r"\[[^\[\]\r\n]+\]", "", claim).casefold().split()
        ).strip(" .")
        filenames, identifiers, terms = (
            set(re.findall(r"\[([^\[\]\r\n]+)\]", claim)),
            _answer_identifiers(claim),
            _focus_tokens(claim),
        )
        scores = [
            (
                100
                * bool(claim_text and claim_text in " ".join(quote.casefold().split()))
                + 50 * bool(filename in filenames)
                + 20 * bool(identifiers and identifiers <= record_ids)
                + len(terms & record_terms),
                heading,
            )
            for heading, quote, filename, record_ids, record_terms in records
        ]
        best = max((score for score, _ in scores), default=0)
        matches = list(
            dict.fromkeys(heading for score, heading in scores if score == best)
        )
        heading = (
            semantic
            if semantic in grouped
            else matches[0]
            if best and len(matches) == 1
            else headings[min(index, len(headings) - 1)]
        )
        filenames = {
            filename
            for record_heading, _, filename, _, _ in records
            if record_heading == heading and filename
        }
        grouped[heading].append(_claim_with_group_citation(claim, filenames))
    sections = [
        f"## {heading}\n" + "\n".join(f"- {claim}" for claim in grouped[heading])
        for heading in headings
        if grouped[heading]
    ]
    return _ensure_structured_section_citations(
        "Here's the breakdown:\n\n" + "\n\n".join(sections), results
    )


def _ensure_structured_section_citations(
    answer: str, results: list[dict[str, Any]]
) -> str:
    filenames: dict[str, set[str]] = {}
    for item in results:
        heading = _answer_section_heading(
            item.get("question", "")
        ) or _question_section_heading(item.get("question", ""))
        if filename := item.get("verified_evidence", {}).get("filename"):
            filenames.setdefault(heading, set()).add(filename)

    def cite(match: re.Match[str]) -> str:
        section, heading = match.group(0), match.group(1).strip()
        if (
            re.search(r"\[[^\[\]\r\n]+\]", section)
            or len(filenames.get(heading, set())) != 1
        ):
            return section
        trailing, lines = (
            section[len(section.rstrip()) :],
            section.rstrip().splitlines(),
        )
        index = next(
            (
                index
                for index in range(len(lines) - 1, 0, -1)
                if re.match(r"^[-*]\s+", lines[index])
            ),
            None,
        )
        if index is not None:
            prefix, claim = re.match(r"^([-*]\s+)(.+)$", lines[index]).groups()
            lines[index] = prefix + _claim_with_group_citation(
                claim, filenames[heading]
            )
        return "\n".join(lines) + trailing

    return re.sub(r"(?ms)^##\s+([^\r\n]+)\r?\n.*?(?=^##\s+|\Z)", cite, answer)


def _question_section_heading(question: str) -> str:
    heading = re.sub(
        r"^(?:(?:what|which)\s+(?:is|are|was|were)|what\s+do(?:es)?\s+the\s+documents?\s+say\s+about)\s+",
        "",
        question.strip().rstrip(".?!"),
        flags=re.IGNORECASE,
    )
    return heading[:1].upper() + heading[1:] if heading else "Details"


def _claim_with_group_citation(claim: str, filenames: set[str]) -> str:
    if re.search("\\[[^\\[\\]\\r\\n]+\\]", claim) or len(filenames) != 1:
        return claim
    punctuation = claim[-1] if claim.endswith((".", "!", "?")) else "."
    body = claim[:-1] if claim.endswith(punctuation) else claim
    return f"{body} [{next(iter(filenames))}]{punctuation}"


def _answer_section_heading(text: str) -> str | None:
    for pattern, heading in (
        (r"\b(?:private|remain private|localhost-only)\b", "Private access"),
        (r"\b(?:public|publicly|exposed)\b", "Public access"),
        (r"\b(?:storage|paths?|locations?)\b", "Storage locations"),
    ):
        if re.search(pattern, text, re.IGNORECASE):
            return heading
    return None


def _identifier_scoped_answer(answer: str, results: list[dict[str, Any]]) -> str:
    target = re.compile(
        r"\b(?:ports?|paths?|storage locations?|file locations?|bind addresses?|ip addresses?|identifiers?|ids?)\b",
        re.IGNORECASE,
    )
    if any(not target.search(item.get("question", "")) for item in results):
        return answer
    if not (
        expected := [
            _answer_identifiers(item.get("verified_evidence", {}).get("quote", ""))
            for item in results
        ]
    ) or any(not identifiers for identifiers in expected):
        return answer
    claims = _conversational_claims(answer)
    matches = [
        {
            index
            for index, identifiers in enumerate(expected)
            if (claim_ids := _answer_identifiers(claim)) and claim_ids <= identifiers
        }
        for claim in claims
    ]
    selected = [
        claim for claim, obligations in zip(claims, matches, strict=True) if obligations
    ]
    covered = (
        set().union(*(obligations for obligations in matches)) if matches else set()
    )
    return (
        " ".join(dict.fromkeys(selected)) if len(covered) == len(expected) else answer
    )


def _answer_identifiers(text: str) -> set[str]:
    return {
        value.casefold().rstrip(".,;:!?")
        for value in re.findall(
            "(?:/[A-Za-z0-9_.-]+)+(?:/[A-Za-z0-9_.-]+)*|(?<![A-Za-z0-9_-])\\.[A-Za-z][A-Za-z0-9_.-]*|\\b\\d+(?:\\.\\d+)+(?:/\\d+)?\\b|\\b\\d{2,}\\b",
            text,
        )
        if value != "127.0.0.1"
    }


def _verified_evidence_answer(results: list[dict[str, Any]]) -> str:
    grouped_quotes: dict[str, list[str]] = {}
    for item in results:
        evidence = item.get("verified_evidence", {})
        filename, quote = evidence.get("filename"), evidence.get("quote")
        if not filename or not quote:
            continue
        quotes = grouped_quotes.setdefault(filename, [])
        normalized_quote = " ".join(quote.casefold().split())
        if any(
            normalized_quote in " ".join(existing.casefold().split())
            for existing in quotes
        ):
            continue
        quotes[:] = [
            existing
            for existing in quotes
            if " ".join(existing.casefold().split()) not in normalized_quote
        ]
        if quote not in quotes:
            quotes.append(quote)
    return (
        "\n\n".join(
            (
                _claim_with_group_citation(" ".join(quotes), {filename})
                for filename, quotes in grouped_quotes.items()
            )
        )
        if grouped_quotes
        else ABSTENTION
    )


def _covers_verified_evidence(answer: str, results: list[dict[str, Any]]) -> bool:
    normalized_answer, verified_items = (
        " ".join(answer.casefold().split()),
        [item for item in results if item.get("verified_evidence", {}).get("quote")],
    )
    if not verified_items:
        return False
    answer_source = [{"filename": "answer", "text": answer, "type": "txt"}]
    return all(
        (
            (
                _verified_quote_is_covered(
                    normalized_answer, item["verified_evidence"]["quote"]
                )
                if " | " in item["verified_evidence"]["quote"]
                else bool(
                    find_grounded_evidence(question, answer_source, allow_complex=True)
                )
            )
            if (question := item.get("question", "").strip())
            else _verified_quote_is_covered(
                normalized_answer, item["verified_evidence"]["quote"]
            )
            for item in verified_items
        )
    )


def _flagged_verified_evidence_answer(
    question: str, sources: list[dict[str, Any]]
) -> str | None:
    evidence = find_grounded_evidence(question, sources, allow_complex=True)
    if (
        not evidence
        or len(evidence.text) > 1000
        or re.search(
            r"\b(?:reviewer can ask|answer should|expected answer)\b",
            evidence.text,
            re.IGNORECASE,
        )
    ):
        return None
    source = next(
        (
            source
            for source in sources
            if source.get("filename", "unknown") == evidence.filename
        ),
        {},
    )
    if source.get("type") == "xlsx" and not re.search(
        r"\b(?:caus\w*|incident)\b", question, re.IGNORECASE
    ):
        return None
    return f"{evidence.text} [{evidence.filename}]"


def _contextualize_flagged_lookup(question: str, answer: str) -> str:
    if "\n" in answer or ":" in re.sub(r"\[[^\]]+\]", "", answer):
        return answer
    subject = re.sub(
        r"^(?:report|state|give|show)\s+(?:the\s+)?",
        "",
        question.strip().rstrip("?."),
        flags=re.IGNORECASE,
    )
    return f"{subject[:1].upper() + subject[1:]}: {answer}" if subject else answer


def _verified_quote_is_covered(normalized_answer: str, quote: str) -> bool:
    if " ".join(quote.casefold().split()) in normalized_answer:
        return True
    fields = _structured_field_spans(quote)
    return bool(
        fields
        and fields[0].endswith((".", "!", "?"))
        and " ".join(fields[0].casefold().split()) in normalized_answer
    )


def _answer_update(
    answer: str,
    confidence: str = "low",
    termination_reason: str = "answered",
    **updates,
) -> dict:
    return {
        "answer": answer,
        "confidence": confidence,
        "termination_reason": termination_reason,
        **updates,
    }


def _budget_answer(state: AgentState, sources: list[dict], reason: str) -> dict:
    text = budget_fallback(
        sources, reason, injection_flagged=state.get("injection_flagged", False)
    )
    return _answer_update(text, termination_reason=reason)


def answer(state: AgentState, context: Any) -> dict:
    safe_sources, previous_answer = _safe_sources(state.get("sources", [])), ""
    overview = state.get("answer_intent") == "overview"
    flagged = state.get("injection_flagged", False)
    confidence = state.get("confidence", "low")
    if overview:
        referenced_filenames = (
            _cited_history_filenames(state.get("history", []))
            if not is_collection_overview_question(state["question"])
            else []
        )
        first_sources = {
            source.get("filename", "unknown"): source
            for source in reversed(safe_sources)
        }
        referenced_sources = [
            first_sources[name]
            for name in referenced_filenames
            if name in first_sources
        ]
        safe_sources = list(
            {
                source.get("filename", "unknown"): source
                for source in referenced_sources or safe_sources
            }.values()
        )[:6]
        if referenced_sources:
            previous_answer = next(
                (
                    str(turn.get("content", "")).strip()
                    for turn in reversed(state.get("history", []))
                    if turn.get("role") == "assistant" and turn.get("content")
                ),
                "",
            )
    if reason := state.get("termination_reason"):
        return _budget_answer(state, safe_sources, reason)
    if not safe_sources:
        warning = "Prompt injection was detected and ignored. " if flagged else ""
        return _answer_update(warning + ABSTENTION, termination_reason="abstained")
    question = state["question"]
    prefix = (
        "Warning: a prompt-injection pattern was detected in retrieved text and ignored.\n\n"
        if flagged
        else ""
    )
    if absence_answer := extract_absence_evidence(question, safe_sources):
        return _answer_update(prefix + absence_answer, "high", "abstained")
    if flagged and (
        verified_answer := _flagged_verified_evidence_answer(question, safe_sources)
    ):
        return _answer_update(prefix + verified_answer, "high", sources=safe_sources)

    results = state.get("subquestion_results")
    if results:
        deterministic = all(
            item.get("coverage_method") == "deterministic"
            and item.get("verified_evidence")
            for item in results
        )
        if (
            deterministic
            or _remaining_seconds(state, context.config)
            <= context.config.answer_reserve_seconds
        ):
            verified_answer = _structured_multi_part_answer(
                _verified_evidence_answer(results), results
            )
            return _answer_update(
                prefix + verified_answer, "high" if deterministic else confidence
            )
        prompt = f"Answer the original multi-part question in two or three connected sentences. Cover every supported part, make the requested comparison or relationship explicit, and do not invent missing links. Do not mention decomposition.\n<evidence>\n{_subquestion_evidence_text(state['subquestion_results'])}\n</evidence>\nQuestion: {state['question']}"
        node = "synthesis"
    else:
        spreadsheet_answer = extract_spreadsheet_lookup(question, safe_sources)
        if spreadsheet_answer and flagged:
            spreadsheet_answer = _contextualize_flagged_lookup(
                question, spreadsheet_answer
            )
        extractors = (
            (extract_collection_overview,)
            if overview
            else (
                extract_structured_answer,
                extract_negative_answer,
                extract_grounded_passages,
                extract_grounded_sentence,
            )
        )
        direct_answer = spreadsheet_answer or next(
            (
                text
                for extract in extractors
                if (text := extract(question, safe_sources))
            ),
            None,
        )
        if direct_answer:
            sources = (
                safe_sources
                if overview and not spreadsheet_answer
                else state.get("sources", [])
            )
            return _answer_update(
                prefix + direct_answer,
                confidence if spreadsheet_answer else "high",
                sources=sources,
            )
        if previous_answer and len(safe_sources) > 1:
            history_overview = _multi_source_history_overview(
                state.get("history", []), safe_sources
            )
            if history_overview:
                return _answer_update(
                    prefix + history_overview, "high", sources=safe_sources
                )
        resolved_context = state.get("current_query") or question
        if overview:
            prompt = (
                referential_overview_prompt(question, previous_answer, safe_sources)
                if previous_answer
                else overview_answer_prompt(question, resolved_context, safe_sources)
            )
        else:
            prompt = f"Answer the question directly. Correct a false premise. Give exactly the requested number of items; otherwise use one factual sentence. Resolved context only identifies references.\nQuestion: {state['question']}\nResolved context: {resolved_context}\n<evidence>\n{_evidence_text(safe_sources)}\n</evidence>"
        node = "overview" if overview else "synthesis"

    text, tokens, failure = context.complete_node(state, prompt, node, 1024)
    if failure:
        return _budget_answer(state, safe_sources, failure)
    if results:
        generated_answer = str(
            parse_json_object(text or "").get("answer") or text or ""
        )
    else:
        generated_answer = (
            parse_overview_answer(text)
            if overview
            else str(parse_json_object(text).get("answer") or text).strip()
            or ABSTENTION
        )
    if flagged:
        generated_answer = _without_trailing_abstention(generated_answer)
    updates = {"total_tokens": state["total_tokens"] + tokens}
    if results:
        generated_answer = _identifier_scoped_answer(
            generated_answer.strip() or ABSTENTION, results
        )
        generated_answer = ground_conversational_answer(
            question, generated_answer, safe_sources
        )
        if not _covers_verified_evidence(generated_answer, results):
            generated_answer = _verified_evidence_answer(results)
        generated_answer = _structured_multi_part_answer(generated_answer, results)
        reason = "answered"
    else:
        if not re.search(
            r"\b(?:why|reasons?|explain|compare|contrast|list|steps?|both|all|and)\b",
            question,
            re.IGNORECASE,
        ):
            generated_answer = generated_answer.split("\n\n", 1)[0].strip()
        if overview or flagged:
            generated_answer = ground_conversational_answer(
                question,
                generated_answer,
                safe_sources,
                minimum_coverage=0.6 if flagged else None,
            )
        else:
            generated_answer = ground_generated_answer(
                question, generated_answer, safe_sources
            )
        if previous_answer and not all(
            f"[{source.get('filename', 'unknown')}]" in generated_answer
            for source in safe_sources
        ):
            generated_answer = previous_answer
        if overview and (
            generated_answer == ABSTENTION
            or not overview_covers_core_fields(generated_answer, safe_sources)
        ):
            generated_answer = extract_source_overview(safe_sources) or ABSTENTION
        updates["sources"] = safe_sources
        reason = "abstained" if generated_answer == ABSTENTION else "answered"
        if reason == "abstained":
            confidence = "low"
    return _answer_update(prefix + generated_answer, confidence, reason, **updates)


def abstain(state: AgentState) -> dict:
    warning = (
        "Prompt injection was detected and ignored. "
        if state.get("injection_flagged")
        else ""
    )
    text = warning + ABSTENTION
    if missing := state.get("missing_subquestions", []):
        text += " Missing evidence for: " + "; ".join(missing) + "."
    return _answer_update(
        text,
        termination_reason=state.get("termination_reason", "abstained"),
        sources=state.get("sources", []),
    )


@dataclass
class AgentSession:
    config: AgentConfig
    runtime: AgentRuntime
    state: AgentState
    retrieval_plan: DecompositionPlan | None = None

    def complete_node(
        self,
        state: AgentState,
        prompt: str,
        node: str,
        output_reserve: int,
        *,
        time_reserve: float = 0.0,
    ):
        if (
            state.get("total_tokens", 0) + estimate_tokens(prompt) + output_reserve
            > self.config.token_limit
        ):
            return (None, 0, "token_limit")
        available_seconds = _remaining_seconds(state, self.config) - time_reserve
        if available_seconds <= 0.1:
            return (None, 0, "wall_clock_limit")
        try:
            text, tokens = self.runtime.complete(prompt, node, available_seconds)
            return (
                (None, tokens, "wall_clock_limit")
                if _remaining_seconds(state, self.config) <= max(0.1, time_reserve)
                else (text, tokens, None)
            )
        except Exception:
            return (
                None,
                0,
                "wall_clock_limit"
                if _remaining_seconds(state, self.config) <= 0.1
                else "model_error",
            )

    def stop(self, reason: str, missing: list[str] | None = None) -> None:
        self.state.update(action="abstain", confidence="low", termination_reason=reason)
        if missing is not None:
            self.state["missing_subquestions"] = missing

    def plan(self) -> bool:
        state = self.state
        self.retrieval_plan = None
        if (
            state["total_tokens"] >= self.config.token_limit
            or _remaining_seconds(state, self.config) <= 0
        ):
            state["termination_reason"] = (
                "token_limit"
                if state["total_tokens"] >= self.config.token_limit
                else "wall_clock_limit"
            )
            return False
        elif not state["history"]:
            if self.config.multi_hop_enabled and is_multi_hop_question(
                state["question"]
            ):
                self.retrieval_plan = plan_decomposition(state["question"])
            return self.retrieval_plan is not None
        else:
            text, tokens, failure = self.complete_node(
                state,
                conversation_plan_prompt(state["question"], state["history"]),
                "planner",
                256,
            )
            state["total_tokens"] += tokens
            if failure in {"token_limit", "wall_clock_limit"}:
                state["termination_reason"] = failure
                return False
            data = parse_json_object(text or "") if not failure else {}
            query = str(data.get("query", "")).strip()
            intent = str(data.get("intent", "")).strip().casefold()
            if (
                failure
                or not query
                or len(query) > 500
                or _follow_up_query_lacks_context(
                    query, state["history"], state["question"]
                )
            ):
                query = _fallback_follow_up_query(state["history"], state["question"])
            state.update(
                current_query=query,
                answer_intent=intent
                if intent in {"answer", "overview"}
                else classify_answer_intent(state["question"]),
            )
            return False

    def multi_retrieve(self) -> None:
        state = self.state
        plan = self.retrieval_plan or plan_decomposition(state["question"])
        if plan is None:
            state["termination_reason"] = "routing_error"
            return
        subquestions, shared = list(plan.questions), plan.shared_retrieval
        dependent = requires_cross_leg_join(state["question"])
        state.update(
            subquestions=subquestions, shared_retrieval=shared, dependent_join=dependent
        )
        subquestions = subquestions[:4]
        first_evidence = None
        if shared:
            source_groups = [
                self.runtime.retrieve(state["question"], self.config.retrieve_top_k)
            ] * len(subquestions)
            queries = [state["question"]]
        elif dependent and len(subquestions) == 2:
            first_sources = self.runtime.retrieve(
                subquestions[0], self.config.retrieve_top_k
            )
            safe_first_sources = _safe_sources(first_sources)
            first_evidence = _strong_row_coverage(
                subquestions[0], safe_first_sources
            ) or _strong_prose_coverage(subquestions[0], safe_first_sources)
            first_result = {"question": subquestions[0], "sources": first_sources}
            if first_evidence:
                first_result.update(
                    verified_evidence=first_evidence, coverage_method="deterministic"
                )
                second_query = _deterministic_join_queries(
                    [subquestions[1]], [first_result]
                )[subquestions[1]]
            else:
                second_query = subquestions[1]
            source_groups = [
                first_sources,
                self.runtime.retrieve(second_query, self.config.retrieve_top_k),
            ]
            queries = [subquestions[0], second_query]
        else:
            source_groups = _retrieve_queries(
                self.runtime, subquestions, self.config.retrieve_top_k
            )
            queries = subquestions
        source_groups = [self._sanitize_sources(group) for group in source_groups]
        sources = _merge_sources(*source_groups)
        results = [
            {"question": question, "sources": group}
            for question, group in zip(subquestions, source_groups, strict=True)
        ]
        if dependent and len(subquestions) == 2 and first_evidence:
            results[0].update(
                verified_evidence=first_evidence, coverage_method="deterministic"
            )
        state.update(
            sources=sources,
            subquestion_results=results,
            iterations=state["iterations"] + len(queries),
            query_history=[*state["query_history"], *queries],
            confidence="medium" if sources else "low",
        )

    def _sanitize_sources(self, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sources, flagged = _clean_sources(sources)
        self.state["injection_flagged"] |= flagged
        return sources

    def retrieve(self) -> None:
        state, top_k = self.state, self.config.retrieve_top_k
        if is_collection_overview_question(state["question"]):
            top_k = max(top_k, 10)
        query = state["current_query"]
        sources = self.runtime.retrieve(query, top_k)
        sources = self._sanitize_sources(sources)
        state.update(
            sources=sources,
            iterations=state["iterations"] + 1,
            query_history=[*state["query_history"], query],
            confidence="medium" if sources else "low",
        )

    def verify(self) -> None:
        state, results = self.state, self.state["subquestion_results"]
        questions = [item["question"] for item in results]
        tokens, missing, failure = _review_coverage(self, state, results)
        state["total_tokens"] += tokens
        missing = _ordered_questions(questions, missing)
        if failure:
            self.stop("coverage_check_failed", missing)
        elif not missing:
            state["missing_subquestions"] = []
        elif not self.config.corrective_pass_enabled:
            self.stop("correction_disabled", missing)
        elif (
            _remaining_seconds(state, self.config)
            <= self.config.answer_reserve_seconds + 1.0
        ):
            state["corrective_pass_triggered"] = True
            self.stop("insufficient_correction_budget", missing)
        else:
            self.correct(missing)

    def correct(self, missing: list[str]) -> None:
        state, results, dependent = (
            self.state,
            self.state["subquestion_results"],
            self.state.get("dependent_join", False),
        )
        state["corrective_pass_triggered"] = True
        queries = (
            _deterministic_join_queries(missing, results)
            if dependent and any(item.get("verified_evidence") for item in results)
            else None
        )
        if queries:
            state["corrective_fallback_used"] = True
        else:
            text, tokens, failure = self.complete_node(
                state,
                _corrective_query_prompt(results, missing, use_verified_join=dependent),
                "corrective_queries",
                512,
                time_reserve=self.config.answer_reserve_seconds + 1.0,
            )
            state["total_tokens"] += tokens
            queries = _validate_corrective_queries(
                parse_json_object(text or ""),
                missing,
                results,
                allow_verified_join=dependent,
            )
            if failure:
                self.stop("corrective_planning_failed", missing)
                return
            state["corrective_fallback_used"] = queries is None
            queries = queries or _fallback_corrective_queries(
                missing, results=results, use_verified_join=dependent
            )
        updated = [
            {**item, "sources": list(item.get("sources", []))} for item in results
        ]
        items = [item for item in updated if item["question"] in queries]
        query_list = [queries[item["question"]] for item in items]
        groups = _retrieve_queries(self.runtime, query_list, self.config.retrieve_top_k)
        revalidate, unchanged = [], []
        for item, query, sources in zip(items, query_list, groups, strict=True):
            state["query_history"].append(query)
            sources = self._sanitize_sources(sources)
            known = {_source_key(source) for source in item["sources"]}
            new = [source for source in sources if _source_key(source) not in known]
            if new:
                item["sources"].extend(new)
                revalidate.append(item)
            else:
                unchanged.append(item["question"])
        state["sources"] = _merge_sources(state["sources"], *groups)
        state.update(
            subquestion_results=updated,
            iterations=state["iterations"] + len(queries),
            corrective_pass_used=True,
            corrective_new_evidence=bool(revalidate),
            corrective_recovered=False,
        )
        if not revalidate:
            self.stop("correction_no_new_evidence", missing)
            return
        if _remaining_seconds(state, self.config) <= 1.0:
            self.stop("correction_budget_exhausted", missing)
            return
        tokens, still_missing, failure = _review_coverage(
            self,
            state,
            revalidate,
            join_results=updated if dependent else None,
            revalidation=True,
        )
        state["total_tokens"] += tokens
        still_missing = _ordered_questions(missing, [*still_missing, *unchanged])
        if failure or still_missing:
            self.stop("revalidation_failed", still_missing or missing)
        else:
            state.update(
                missing_subquestions=[], confidence="medium", corrective_recovered=True
            )

    def run(self) -> AgentState:
        multi_hop = self.plan()
        if not self.state.get("termination_reason"):
            self.multi_retrieve() if multi_hop else self.retrieve()
        if self.state.get("subquestion_results"):
            self.verify()
        result = (
            abstain(self.state)
            if self.state.get("action") == "abstain"
            else answer(self.state, self)
        )
        self.state.update(result)
        return self.state


def run_agent(
    index,
    question: str,
    history: list[dict[str, str]] | None = None,
    config: AgentConfig | None = None,
    runtime: AgentRuntime | None = None,
    *,
    telemetry_mode: str = "agent",
) -> dict[str, Any]:
    config, runtime = (
        config or AgentConfig.from_env(),
        runtime
        or AgentRuntime(
            retrieve=lambda query, top_k: retrieve_sources(index, query, top_k=top_k),
            complete=lambda prompt, node, timeout: call_model(
                prompt, node=node, timeout=timeout
            ),
        ),
    )
    injection_flagged, started = contains_prompt_injection(question), perf_counter()
    safe_question = (
        _neutralize_prompt_injection(question).strip()
        if injection_flagged
        else question.strip()
    )
    initial_state: AgentState = {
        "question": safe_question,
        "current_query": safe_question,
        "history": history or [],
        "answer_intent": classify_answer_intent(safe_question),
        "query_history": [],
        "sources": [],
        "iterations": 0,
        "total_tokens": 0,
        "started_at": monotonic(),
        "confidence": "low",
        "injection_flagged": injection_flagged,
    }
    executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="agentic-crag-agent"
    )
    try:
        state = executor.submit(
            AgentSession(config, runtime, initial_state).run
        ).result(timeout=max(0.01, config.timeout_seconds - 0.05))
    except Exception as exc:
        reason = (
            "wall_clock_limit"
            if isinstance(exc, FutureTimeoutError)
            else f"error:{type(exc).__name__}"
        )
        state = dict(initial_state)
        state.update(
            _budget_answer(
                initial_state, _safe_sources(initial_state["sources"]), reason
            )
        )
        state["termination_reason"] = (
            "wall_clock_limit" if isinstance(exc, FutureTimeoutError) else "error"
        )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    latency_ms = (perf_counter() - started) * 1000
    log_query_result(
        mode=telemetry_mode,
        iterations=state.get("iterations", 0),
        latency_ms=latency_ms,
    )
    answer, sources = (
        consolidate_repeated_citations(state.get("answer", ABSTENTION)),
        state.get("sources", []),
    )
    sources = (
        supporting_source_groups(
            answer, sources, verified_evidence=state.get("subquestion_results", [])
        )
        if state.get("termination_reason") == "answered" and ABSTENTION not in answer
        else sources
    )
    return {
        "answer": answer,
        "sources": sources,
        "agent": {
            "iterations": state.get("iterations", 0),
            "confidence": state.get("confidence", "low"),
            "termination_reason": state.get("termination_reason", "unknown"),
            "query_history": state.get("query_history", []),
            "token_usage": state.get("total_tokens", 0),
            "prompt_injection_flagged": state.get("injection_flagged", False),
            "corrective_pass_used": state.get("corrective_pass_used", False),
            "corrective_pass_triggered": state.get("corrective_pass_triggered", False),
            "corrective_fallback_used": state.get("corrective_fallback_used", False),
            "corrective_new_evidence": state.get("corrective_new_evidence", False),
            "corrective_recovered": state.get("corrective_recovered", False),
            "missing_subquestions": state.get("missing_subquestions", []),
            "latency_ms": round(latency_ms, 3),
        },
    }
