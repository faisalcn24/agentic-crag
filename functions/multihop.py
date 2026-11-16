from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from .grounding import (
    _normalize,
    find_grounded_evidence,
    _named_anchors,
    _ordered_terms,
    _contains_all,
    _evidence_sentences,
)


DECOMPOSITION_SCHEMA = {
    "type": "object",
    "properties": {
        "subquestions": {
            "type": "array",
            "items": {"type": "string", "minLength": 5, "maxLength": 300},
            "minItems": 2,
            "maxItems": 4,
            "uniqueItems": True,
        }
    },
    "required": ["subquestions"],
    "additionalProperties": False,
}

_IDENTIFIER = re.compile(r"\b[A-Z]{1,8}-\d{2,6}\b", re.IGNORECASE)
_ANCHOR = re.compile(r"\b[A-Z]{1,8}-\d{2,6}\b|\b\d+\.\d+\b", re.IGNORECASE)
_MULTI_HOP = re.compile(
    r"\b(?:compare|contrast|connect|relate|distinguish|differentiate)\b"
    r"|\b(?:across|between)\b.+\b(?:documents?|sources?|versions?|files?|corpus|collection)\b"
    r"|\b(?:documents?|sources?|versions?|files?|corpus|collection)\b.+\b(?:across|between)\b"
    r"|\b(?:change|difference|progression|relationship)\b.+\b(?:between|from|across)\b"
    r"|\b(?:first|second|another|other)\s+(?:document|source)\b.+"
    r"\b(?:first|second|another|other)\s+(?:document|source)\b"
    r"|\bhow do\b.+\band\b.+\b(?:address|align|compare|contrast|differ|divide|relate|support|work)\b"
    r"|\bhow does\b.+\b(?:align|connect|relate|support)\b"
    r"|\b(?:two|three|multiple)\s+(?:document-based|source-backed|independent)\b"
    r"|\b(?:plus|together with|combined with|beyond)\b"
    r"|,\s*and\s+(?:what|which|who|where|when|why)\b",
    re.IGNORECASE | re.DOTALL,
)


def is_multi_hop_question(question: str) -> bool:
    return len(set(map(str.casefold, _IDENTIFIER.findall(question)))) >= 2 or bool(
        _MULTI_HOP.search(question)
    )


def requires_cross_leg_join(question: str) -> bool:
    return bool(
        re.search(
            r"\b(?:connect|relate|support|align|match(?:es|ed|ing)?|same)\b",
            question,
            re.IGNORECASE,
        )
    )


def decomposition_prompt(question: str) -> str:
    return f"Split this into two to four standalone retrieval questions, one required fact or calculation each. Preserve exact identifiers and qualifiers; do not invent or answer.\nQuestion: {question}"


def sentence_decomposition(question: str) -> list[str] | None:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", question.strip())
        if sentence.strip()
        and not re.match(
            r"^(?:cite|include|provide|show)\b.*\b(?:citation|citations|source|sources)\b",
            sentence.strip(),
            re.IGNORECASE,
        )
    ]
    if not 2 <= len(sentences) <= 3:
        return None
    parts = [
        _as_question(clause)
        for sentence in sentences
        for clause in re.split(
            r"\s+and\s+(?=(?:what|which|who|where|when|why|how)\b)",
            re.sub(r"^then\s+", "", sentence, flags=re.IGNORECASE),
            flags=re.IGNORECASE,
        )
    ]
    return parts if 2 <= len(parts) <= 3 else None


def validate_decomposition(data: Any, original_question: str) -> list[str] | None:
    values = data.get("subquestions") if isinstance(data, dict) else None
    if not isinstance(values, list):
        return None
    questions = [
        " ".join(value.split()).strip() for value in values if isinstance(value, str)
    ]
    distinct = {q.casefold() for q in questions}
    valid = (
        len(questions) == len(values)
        and 2 <= len(questions) <= 4
        and len(distinct) == len(questions)
    )
    if not valid or not all(
        5 <= len(q) <= 300 and q.endswith("?") and "{" not in q and "}" not in q
        for q in questions
    ):
        return None
    if distinct == {original_question.casefold()}:
        return None
    anchors, combined = _anchors(original_question), " ".join(questions).casefold()
    if any(anchor not in combined for anchor in anchors):
        return None
    if len(anchors) >= 2:
        separated = [
            q.casefold()
            for q in questions
            if not all(a in q.casefold() for a in anchors)
        ]
        if any(not any(anchor in q for q in separated) for anchor in anchors):
            return None
    return questions


def fallback_decomposition(question: str) -> list[str]:
    identifiers = list(dict.fromkeys(_IDENTIFIER.findall(question)))
    if len(identifiers) >= 2:
        return [
            f"What does {identifier} state that is relevant to the question?"
            for identifier in identifiers[:4]
        ]

    versions = _version_range(question)
    if versions:
        topic = re.search(
            r"\bhow did\s+(.+?)\s+change\s+from\s+version",
            question,
            re.IGNORECASE,
        )
        return [
            (
                f"What {topic.group(1).strip()} was documented for version {version}?"
                if topic
                else f"What changed in version {version}?"
            )
            for version in versions
        ]

    comparison = re.search(
        r"\b(?:compare|contrast)\s+(.+?)\s+(?:and|with|to)\s+(.+?)[?.]?$",
        question,
        re.IGNORECASE,
    )
    if comparison:
        return [
            f"What do the documents say about {subject.strip()}?"
            for subject in comparison.groups()
        ]

    parts = re.split(
        r",\s*and\s+(?=what|how|why|which)",
        question,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    if len(parts) == 2:
        first, second = map(_as_question, parts)
        topic = re.search(r"^Why might (.+?) be difficult\?$", first, re.IGNORECASE)
        if topic and re.match(r"^What improvement\b", second, re.IGNORECASE):
            second = _as_question(f"{second.rstrip('?')} for {topic.group(1)}")
        return [first, second]

    clean = question.strip().rstrip("?")
    return [
        f"What evidence supports the first required part of this question: {clean}?",
        f"What evidence supports the second required part of this question: {clean}?",
    ]


def deterministic_decomposition(question: str) -> list[str] | None:
    if len(
        set(map(str.casefold, _IDENTIFIER.findall(question)))
    ) >= 2 or _version_range(question):
        return fallback_decomposition(question)

    match = re.fullmatch(
        r"\s*(?:(?:compare|contrast)\s+)?how(?:\s+do)?\s+(?P<subjects>.+?)\s+"
        r"(?:address|align|compare|contrast|differ|divide|relate|support)\s+"
        r"(?P<topic>.+?)[?.]?\s*",
        question,
        re.IGNORECASE,
    )
    if match:
        subjects = [
            re.sub(r"^and\s+", "", value.strip(" ,"), flags=re.IGNORECASE)
            for value in re.split(r"\s*,\s*|\s+and\s+", match.group("subjects"))
            if value.strip(" ,")
        ]
        if 2 <= len(subjects) <= 4:
            return [
                f"What does {subject} say about {match.group('topic').strip()}?"
                for subject in subjects
            ]

    match = re.fullmatch(
        r"\s*how\s+does\s+(?P<subject>.+?)\s+"
        r"(?:align|connect|relate|support)\s+(?P<object>.+?)[?.]?\s*",
        question,
        re.IGNORECASE,
    )
    if match:
        target = match.group("object").strip()
        relation = (
            "measure" if re.search(r"\bmetrics?\b", target, re.IGNORECASE) else "state"
        )
        return [
            f"What does {match.group('subject').strip()} state?",
            f"What does {target} {relation}?",
        ]

    match = re.fullmatch(
        r"\s*connect\s+(?P<left>.+?)\s+with\s+(?P<right>.+?)[?.]?\s*",
        question,
        re.IGNORECASE,
    )
    if match:
        return [
            f"What does {match.group('left').strip()} state?",
            f"What evidence describes {match.group('right').strip()}?",
        ]

    named_comparison = re.fullmatch(
        r"\s*(?i:compare|contrast)\s+(?P<prefix>.*?)"
        r"(?P<left>[A-Z][A-Za-z0-9._/-]*)\s+and\s+"
        r"(?P<right>[A-Z][A-Za-z0-9._/-]*)\s+(?P<topic>.+?)[?.]?\s*",
        question,
    )
    if named_comparison:
        subject_prefix = re.sub(
            r"^the\s+",
            "",
            named_comparison.group("prefix").strip(),
            flags=re.IGNORECASE,
        )
        topic = named_comparison.group("topic").strip()
        return [
            _as_question(
                f"What {subject_prefix} {topic} does {subject} use".replace("  ", " ")
            )
            for subject in (
                named_comparison.group("left"),
                named_comparison.group("right"),
            )
        ]

    if re.search(r"\b(?:compare|contrast)\b", question, re.IGNORECASE):
        comparison = fallback_decomposition(question)
        if not comparison[0].startswith(
            "What evidence supports the first required part"
        ):
            return comparison

    parts = re.split(
        r",\s*and\s+(?=what|how|why|which)",
        question,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    if len(parts) == 2:
        return fallback_decomposition(question)
    return None


@dataclass(frozen=True)
class DecompositionPlan:
    questions: tuple[str, ...]
    shared_retrieval: bool


def plan_decomposition(question: str) -> DecompositionPlan | None:
    """Resolve and validate the retrieval obligations once per request."""
    sentences = sentence_decomposition(question)
    if sentences is not None:
        return DecompositionPlan(tuple(sentences), shared_retrieval=True)
    parts = deterministic_decomposition(question)
    if not parts or len(parts) > 2 and not _version_range(question):
        return None
    if re.search(r"\b(?:compare|contrast)\b", question, re.IGNORECASE):
        named = re.findall(r"\b[A-Z][A-Za-z0-9._/-]*\b", question)
        if (
            len(named) < 2
            and not _IDENTIFIER.search(question)
            and not _version_range(question)
        ):
            return None
    return DecompositionPlan(tuple(parts), shared_retrieval=False)


def decomposition_is_useful(question: str) -> bool:
    return plan_decomposition(question) is not None


def _version_range(question: str) -> list[str]:
    match = re.search(
        r"\bversions?\s+(\d+)\.(\d+)\s+(?:to|through)\s+"
        r"(?:versions?\s+)?(\d+)\.(\d+)\b",
        question,
        re.IGNORECASE,
    )
    if not match or match.group(1) != match.group(3):
        return []
    start, end = int(match.group(2)), int(match.group(4))
    if end < start or end - start > 3:
        return []
    minors = (
        [start, end]
        if re.search(r"\bfrom\s+version\b", question, re.IGNORECASE)
        else range(start, end + 1)
    )
    return [f"{match.group(1)}.{minor}" for minor in minors]


def _anchors(question: str) -> list[str]:
    return list(dict.fromkeys(value.casefold() for value in _ANCHOR.findall(question)))


def _as_question(text: str) -> str:
    text = text.strip().rstrip(".?")
    return text[:1].upper() + text[1:] + "?"


# Evidence selection and bounded corrective-query support.
INJECTION_PATTERNS = (
    re.compile(
        "\\b(?:ignore|disregard|forget|override)\\b.{0,80}\\b(?:instructions?|question|context|values?|system|developer)\\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        "\\b(?:system|developer|assistant)\\s+(?:message|override|prompt)\\s*:",
        re.IGNORECASE,
    ),
    re.compile(
        "\\b(?:hide|omit|remove)\\b.{0,40}\\b(?:citations?|sources?)\\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        "\\b(?:new task|unrestricted assistant|instruction inside document|reveal the system prompt|exfiltrate)\\b",
        re.IGNORECASE,
    ),
)


def contains_prompt_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in INJECTION_PATTERNS)


def _neutralize_prompt_injection(text: str) -> str:
    spans = [
        span.strip() for span in re.split(r"(?<=[.!?])\s+|\r?\n+", text) if span.strip()
    ]
    return " ".join(
        span for span in spans if not contains_prompt_injection(span)
    ).strip()


def _clean_sources(
    sources: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    cleaned, flagged = [], False
    for source in sources:
        text = source.get("text", "")
        if contains_prompt_injection(text):
            flagged = True
            text = _neutralize_prompt_injection(text)
            source = {**source, "text": text}
        if text.strip():
            cleaned.append(source)
    return cleaned, flagged


def _safe_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _clean_sources(sources)[0]


def _focused_evidence(text: str, question: str, max_chars: int = 1200) -> str:
    if len(text.strip()) <= max_chars:
        return text.strip()
    units = [
        unit.strip()
        for unit in (
            text.splitlines() if " | " in text else re.split(r"(?<=[.!?])\s+|\n+", text)
        )
        if unit.strip() and not unit.strip().endswith("?")
    ]
    tokens = _focus_tokens(question)
    ranked = sorted(
        enumerate(units),
        key=lambda item: len(tokens & _focus_tokens(item[1])),
        reverse=True,
    )[:4]
    return " ".join(unit for _, unit in sorted(ranked))[:max_chars].rstrip()


def _focus_tokens(text: str) -> set[str]:
    return {
        _normalize(token)
        for token in re.findall("[a-z0-9]+", text.casefold())
        if token
        not in {
            "a",
            "and",
            "are",
            "does",
            "for",
            "how",
            "is",
            "of",
            "the",
            "to",
            "what",
            "which",
        }
    }


def _focus_token_allowed(token: str, allowed: set[str]) -> bool:
    return (
        token in allowed
        or len(token) >= 4
        and (token + "e" in allowed or token.endswith("e") and token[:-1] in allowed)
    )


def _question_identifiers(text: str) -> set[str]:
    return {
        value.casefold()
        for value in re.findall(r"\b[A-Za-z]{1,8}-\d{2,6}\b|\b\d+\.\d+\b", text)
    }


def _source_body(text: str) -> str:
    return re.sub(
        r"^Document filename:[^\r\n]*?(?=(?:Version\s+\d+\.\d+\s+){2}|Executive Summary|Deployment Shape|Evaluation Goals|\r?\n\r?\n|\Z)",
        "",
        text.strip(),
        count=1,
        flags=re.IGNORECASE,
    )


def _subquestion_evidence_text(results: list[dict[str, Any]]) -> str:
    groups = []
    for item in results:
        question, sources = (
            item.get("question", ""),
            _safe_sources(item.get("sources", [])),
        )
        ranked = sorted(
            (
                (
                    len(
                        _focus_tokens(question) & _focus_tokens(source.get("text", ""))
                    ),
                    position,
                    source,
                )
                for position, source in enumerate(sources)
            ),
            key=lambda item: (-item[0], item[1]),
        )
        cutoff = ranked[min(1, len(ranked) - 1)][0] if ranked else 0
        selected = [source for score, _, source in ranked if score >= cutoff]
        evidence = "\n\n".join(
            f"SOURCE: {source.get('filename', 'unknown')}\n{_focused_evidence(source.get('text', ''), question)}"
            for source in selected
        )
        groups.append(
            f"SUBQUESTION: {question}\n"
            + (evidence or "(no supporting evidence retrieved)")
        )
    return "\n\n".join(groups)


def _evidence_coverage_prompt(
    results: list[dict[str, Any]],
    *,
    revalidation: bool = False,
    verified_results: list[dict[str, Any]] | None = None,
) -> str:
    stage, join_instruction = (
        ("corrective" if revalidation else "initial"),
        (
            " Verified first-pass quotes may supply join terms, but each returned quote must come from that subquestion's retrieval."
            if verified_results
            else ""
        ),
    )
    verified_context = (
        f"\n<verified_first_pass>\n{_verified_join_evidence_text(verified_results or [])}\n</verified_first_pass>"
        if verified_results
        else ""
    )
    return f'Review {stage} retrieval. For each SUBQUESTION, mark covered only if one named source directly answers it; copy one verbatim quote. Otherwise use false and nulls.{join_instruction} Return one record per subquestion.\n<output>{{"coverage":[{{"subquery":"exact subquestion","covered":true,"filename":"exact filename","quote":"verbatim quote"}}]}}</output>\n<retrieval_results>\n{_subquestion_evidence_text(results)}\n</retrieval_results>{verified_context}'


def _strong_row_coverage(
    question: str, sources: list[dict[str, Any]], *, join_terms: set[str] | None = None
) -> dict[str, str] | None:
    tokens = _focus_tokens(question) - {
        "about",
        "describe",
        "document",
        "evidence",
        "exact",
        "say",
        "state",
    }
    if len(tokens) < 2:
        return None
    identifiers, candidates = _question_identifiers(question), []
    for source in sources:
        text, filename = source.get("text", ""), source.get("filename", "unknown")
        if re.search(r"\bincidents?\b", question, re.IGNORECASE) and not re.search(
            r"\bIncident_ID\s*:", text, re.IGNORECASE
        ):
            continue
        if re.search(r"\brisks?\b", question, re.IGNORECASE) and not re.search(
            r"\bRisk_ID\s*:", text, re.IGNORECASE
        ):
            continue
        source_tokens = _focus_tokens(text)
        source_identifiers = _question_identifiers(text)
        overlap, join_overlap = (
            len(tokens & source_tokens) / len(tokens),
            len((join_terms or set()) & source_tokens),
        )
        rows = [
            line.strip() for line in text.splitlines() if " | " in line and ": " in line
        ]
        identifier_match = not identifiers or identifiers <= source_identifiers
        enough_context = (
            overlap >= 0.6
            or overlap >= 0.25
            and (bool(identifiers) or join_overlap >= 2)
        )
        if (
            source.get("type") == "xlsx"
            and identifier_match
            and len(rows) == 1
            and enough_context
        ):
            candidates.append((overlap + min(join_overlap, 5) / 10, filename, rows[0]))
    candidates.sort(reverse=True)
    if not candidates or len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return None
    _, filename, quote = candidates[0]
    return {"filename": filename, "quote": quote}


def _strong_prose_coverage(
    question: str, sources: list[dict[str, Any]]
) -> dict[str, str] | None:
    question_tokens = _focus_tokens(question) - {
        "about",
        "be",
        "describe",
        "difficult",
        "document",
        "evidence",
        "improvement",
        "might",
        "repeatedly",
        "say",
        "state",
        "suggested",
        "why",
        "should",
    }
    if len(question_tokens) < 2:
        return None
    required_identifiers = _question_identifiers(question)
    candidates: list[tuple[bool, float, int, str, str]] = []
    for position, source in enumerate(sources):
        if source.get("type") == "xlsx":
            continue
        body = _source_body(source.get("text", ""))
        if required_identifiers and not required_identifiers <= _question_identifiers(
            body
        ):
            continue
        evidence = find_grounded_evidence(
            question, [{**source, "text": body}], allow_complex=True
        )
        units = [
            unit.strip()
            for unit in re.split(r"(?<=[.!?])\s+|\n+", body)
            if unit.strip() and not unit.strip().endswith("?")
        ]
        lexical = [
            unit
            for unit in units
            if len(question_tokens & _focus_tokens(unit)) >= 2
            and len(question_tokens & _focus_tokens(unit)) / len(question_tokens) >= 0.4
        ]
        choices = lexical or ([evidence.text] if evidence else [])
        if not choices:
            continue
        quote = max(
            choices,
            key=lambda unit: (
                len(question_tokens & _focus_tokens(unit)) / len(question_tokens),
                -len(_focus_tokens(unit)),
            ),
        )
        quote = _expand_anchor_context(body, quote, question)
        if not _quote_matches_subquestion(question, quote):
            continue
        overlap = len(question_tokens & _focus_tokens(quote)) / len(question_tokens)
        explicit_identifier = bool(
            required_identifiers
            and required_identifiers <= _question_identifiers(quote)
        )
        candidates.append(
            (
                explicit_identifier,
                overlap,
                position,
                source.get("filename", "unknown"),
                quote,
            )
        )
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3], item[4]))
    if not candidates:
        return None
    _, _, _, filename, quote = candidates[0]
    return {"filename": filename, "quote": quote}


def _expand_anchor_context(source_text: str, evidence: str, question: str) -> str:
    identifiers = _question_identifiers(question)
    if not identifiers:
        return evidence
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", source_text)
        if sentence.strip()
    ]
    normalized_evidence = " ".join(evidence.casefold().split())
    indices = [
        index
        for index, sentence in enumerate(sentences)
        if " ".join(sentence.casefold().split()) in normalized_evidence
    ]
    if not indices:
        return evidence
    start, end = min(indices), max(indices)
    if len(identifiers) == 1 and re.fullmatch(r"\d+\.\d+", next(iter(identifiers))):
        version = next(iter(identifiers))
        anchor = next(
            (
                index
                for index, sentence in enumerate(sentences)
                if set(re.findall(r"\b\d+\.\d+\b", sentence)) == {version}
            ),
            start,
        )
        right = anchor + 1
        while right < len(sentences) and right - anchor < 4:
            versions = set(re.findall(r"\b\d+\.\d+\b", sentences[right]))
            if versions and version not in versions:
                break
            right += 1
        return " ".join(sentences[anchor:right])
    if identifiers <= _question_identifiers(evidence):
        return evidence
    anchors = [
        index
        for index, sentence in enumerate(sentences)
        if identifiers & _question_identifiers(sentence)
    ]
    for anchor in sorted(
        anchors, key=lambda index: min(abs(index - start), abs(index - end))
    ):
        quote = " ".join(sentences[min(start, anchor) : max(end, anchor) + 1])
        if identifiers <= _question_identifiers(quote):
            return quote
    return evidence


def _ordered_questions(order: list[str], selected: list[str]) -> list[str]:
    selected = set(selected)
    return [question for question in order if question in selected]


def _literal_source_quotes(source_text: str, proposed_quote: str) -> list[str]:
    if whole := _whitespace_normalized_source_quote(source_text, proposed_quote):
        return [whole]
    return list(
        dict.fromkeys(
            exact
            for sentence in re.split(r"(?<=[.!?])\s+", proposed_quote.strip())
            if len(sentence.split()) >= 3
            and (exact := _whitespace_normalized_source_quote(source_text, sentence))
        )
    )


def _whitespace_normalized_source_quote(
    source_text: str, proposed_quote: str
) -> str | None:
    pattern = r"\s+".join(re.escape(word) for word in proposed_quote.split())
    match = re.search(pattern, source_text, re.IGNORECASE) if pattern else None
    return (
        proposed_quote
        if proposed_quote in source_text
        else match.group(0)
        if match
        else None
    )


def _reported_filename_matches(reported: str, actual: str) -> bool:
    return actual == reported or bool(
        re.search("\\.xlsx$", reported, re.IGNORECASE)
        and actual.casefold().startswith(reported.casefold() + "-")
    )


def _quote_matches_subquestion(question: str, quote: str) -> bool:
    identifiers = _question_identifiers(question)
    if identifiers - _question_identifiers(quote):
        return False
    instruction_words = {
        "cite",
        "compare",
        "connect",
        "contrast",
        "for",
        "how",
        "state",
        "then",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
    }
    named = {
        _normalize(token.casefold())
        for token in re.findall(r"\b[A-Z][A-Za-z0-9._-]*\b", question)
        if len(token) > 1
        and token.casefold() not in instruction_words
        and token.casefold() not in identifiers
    }
    return not named or named <= _focus_tokens(quote)


def _subquery_labels_match(reported: Any, expected: str) -> bool:
    return isinstance(reported, str) and " ".join(reported.casefold().split()).rstrip(
        "?!. "
    ) == " ".join(expected.casefold().split()).rstrip("?!. ")


def _verified_join_evidence_text(results: list[dict[str, Any]]) -> str:
    records = [
        f"SUBQUESTION: {item.get('question', '')}\nSOURCE: {evidence['filename']}\nQUOTE: {evidence['quote']}"
        for item in results
        if (evidence := item.get("verified_evidence", {})).get("filename")
        and evidence.get("quote")
    ]
    return "\n\n".join(records) or "(none)"


def _distinctive_join_terms(results: list[dict[str, Any]], subquery: str) -> set[str]:
    return (
        _focus_tokens(
            " ".join(
                str(item.get("verified_evidence", {}).get("quote", ""))
                for item in results
                if item.get("question") != subquery
                and item.get("verified_evidence", {}).get("quote")
            )
        )
        - _focus_tokens(subquery)
        - {
            "deployment",
            "document",
            "evidence",
            "exact",
            "id",
            "impact",
            "mitigat",
            "mitigation",
            "owner",
            "probability",
            "requirement",
            "risk",
            "statu",
            "system",
        }
    )


def _corrective_query_prompt(
    results: list[dict[str, Any]],
    missing: list[str],
    *,
    use_verified_join: bool = False,
) -> str:
    missing_results = [
        item for item in results if item.get("question", "") in set(missing)
    ]
    join_guidance = (
        " For a cross-document join, include two distinctive terms from VERIFIED FIRST-PASS QUOTES."
        if use_verified_join
        else ""
    )
    verified_context = (
        f"\n<verified_first_pass>\n{_verified_join_evidence_text(results)}\n</verified_first_pass>"
        if use_verified_join
        else ""
    )
    return f"""Write one focused retrieval query per missing subquestion using only terms present below. Preserve exact identifiers and return records in order; do not answer.{join_guidance}\n<output>{{"queries":[{{"missing_subquery":"exact missing subquestion","query":"focused query"}}]}}</output>\nMissing subquestions:\n- {"\n- ".join(missing)}\n<failed_leg_retrieval>\n{_subquestion_evidence_text(missing_results)}\n</failed_leg_retrieval>{verified_context}"""


def _fallback_corrective_queries(
    missing: list[str],
    *,
    results: list[dict[str, Any]] | None = None,
    use_verified_join: bool = False,
) -> dict[str, str]:
    return {
        subquery: (
            f"{subquery.rstrip('?')} related verified evidence {verified_quote}"
            if use_verified_join
            and (
                verified_quote := next(
                    (
                        str(item.get("verified_evidence", {}).get("quote", ""))
                        for item in results or []
                        if item.get("verified_evidence", {}).get("quote")
                    ),
                    "",
                )
            )
            else f"{subquery.rstrip('?')} exact documented evidence"
        )[:300].rstrip()
        for subquery in missing
    }


def _deterministic_join_queries(
    missing: list[str], results: list[dict[str, Any]]
) -> dict[str, str]:
    quote = next(
        (
            str(item.get("verified_evidence", {}).get("quote", ""))
            for item in results
            if item.get("verified_evidence", {}).get("quote")
        ),
        "",
    )
    identifiers = re.findall(r"\b[A-Za-z]{1,8}-\d{2,6}\b", quote)
    allowed = _focus_tokens(quote) - {
        "date",
        "filename",
        "high",
        "impact",
        "medium",
        "mitigation",
        "owner",
        "probability",
        "register",
        "risk",
        "sheet",
        "status",
    }
    terms = [
        raw
        for raw in re.findall(r"[A-Za-z0-9/.-]+", quote)
        if _normalize(raw.casefold().strip(".")) in allowed
    ]
    distinctive = " ".join(dict.fromkeys([*identifiers, *terms]))[:180].rstrip()
    return {
        subquery: f"Find incident fix matching {distinctive}"
        if re.search(r"\bincidents?\b", subquery, re.IGNORECASE)
        else f"{subquery.rstrip('?')} matching {distinctive}"
        for subquery in missing
    }


def _validate_corrective_queries(
    data: dict[str, Any],
    missing: list[str],
    results: list[dict[str, Any]] | None = None,
    *,
    allow_verified_join: bool = False,
) -> dict[str, str] | None:
    records = data.get("queries")
    if not isinstance(records, list) or len(records) != len(missing):
        return None
    planned, seen_queries = {}, set()
    verified_join_text = (
        " ".join(
            (
                str(item.get("verified_evidence", {}).get("quote", ""))
                for item in results or []
                if item.get("verified_evidence", {}).get("quote")
            )
        )
        if allow_verified_join
        else ""
    )
    evidence_by_question = {
        item.get("question", ""): " ".join(
            [
                *(
                    source.get("text", "")
                    for source in _safe_sources(item.get("sources", []))
                ),
                verified_join_text,
            ]
        ).strip()
        for item in results or []
    }
    for subquery, record in zip(missing, records, strict=True):
        if not isinstance(record, dict):
            return None
        reported_subquery, query = record.get("missing_subquery"), record.get("query")
        if (
            not isinstance(reported_subquery, str)
            or len(missing) > 1
            and reported_subquery != subquery
            or not isinstance(query, str)
        ):
            return None
        cleaned, evidence_text = (
            " ".join(query.split()).strip(),
            evidence_by_question.get(subquery, ""),
        )
        required_identifiers = _question_identifiers(subquery)
        query_identifiers = _question_identifiers(cleaned)
        allowed_identifiers = _question_identifiers(f"{subquery} {evidence_text}")
        query_terms = _focus_tokens(cleaned)
        lowered = cleaned.casefold()
        allowed_terms = (
            _focus_tokens(subquery)
            | _focus_tokens(evidence_text)
            | {
                "detail",
                "details",
                "document",
                "documented",
                "evidence",
                "exact",
                "find",
                "full",
                "information",
                "passage",
                "record",
                "records",
                "requirement",
                "requirements",
                "section",
                "source",
                "specification",
                "statement",
                "text",
            }
        )
        distinctive_join_terms = (
            _distinctive_join_terms(results or [], subquery)
            if allow_verified_join
            else set()
        )
        invalid = (
            not 5 <= len(cleaned) <= 300
            or "{" in cleaned
            or "}" in cleaned
            or lowered in seen_queries
            or any(identifier not in lowered for identifier in required_identifiers)
            or query_identifiers - allowed_identifiers
            or any(
                not _focus_token_allowed(token, allowed_terms) for token in query_terms
            )
            or allow_verified_join
            and len(distinctive_join_terms) >= 2
            and len(query_terms & distinctive_join_terms) < 2
        )
        if invalid:
            return None
        planned[subquery] = cleaned
        seen_queries.add(lowered)
    return planned


def extract_grounded_passages(
    question: str, sources: list[dict[str, Any]]
) -> str | None:
    named = _named_anchors(question)
    if (
        not re.search(
            r"\b(?:across|beyond|compare|contrast|distinguish|plus|two)\b",
            question,
            re.IGNORECASE,
        )
        and len(named) < 2
    ):
        return None
    question_terms = set(_ordered_terms(question)) - {
        "across",
        "beyond",
        "compare",
        "contrast",
        "corpu",
        "current",
        "document",
        "evidence",
        "establish",
        "good",
        "make",
        "reason",
        "specifi",
        "support",
        "two",
    }
    if len(question_terms) < 2:
        return None
    expansions = {
        "evaluation": {
            "answer",
            "citation",
            "correct",
            "evidence",
            "faithful",
            "ground",
            "refus",
            "retriev",
            "unsupported",
        },
        "health": {"404", "api", "backend", "curl", "direct", "reload", "site"},
        "host": {"external", "groq", "hybrid", "provider", "send"},
        "privacy": {"document", "excerpt", "external", "full", "local", "send"},
        "swap": {"2", "dependency", "gb", "install", "memory"},
        "unsafe": {
            "authentication",
            "external",
            "https",
            "public",
            "sharing",
            "temporary",
        },
    }
    semantic_terms = set(question_terms)
    for trigger, related in expansions.items():
        if trigger in question_terms:
            semantic_terms.update(related)
    if "app" in question_terms:
        question_terms.remove("app")
        semantic_terms.discard("app")
        semantic_terms.add("backend")
    ranked = []
    for position, sentence in enumerate(
        _evidence_sentences(sources, include_spreadsheets=True)
    ):
        if len(sentence.terms) > 55:
            continue
        original_hits = sum(
            _contains_all(sentence.terms, {term}) for term in question_terms
        )
        semantic_hits = sum(
            _contains_all(sentence.terms, {term})
            for term in semantic_terms - question_terms
        )
        named_hits = sum(_contains_all(sentence.terms, {term}) for term in named)
        if original_hits < 2 and not (named and named_hits >= min(2, len(named))):
            continue
        relation_bonus = (
            3
            if re.search(r"\bdistinguish\b", question, re.IGNORECASE)
            and re.search(r"\bif\b.+\bbut\b", sentence.text, re.IGNORECASE)
            else 0
        )
        score = original_hits * 3 + semantic_hits + named_hits * 4 + relation_bonus
        ranked.append(
            (
                score,
                original_hits,
                semantic_hits,
                named_hits,
                -len(sentence.terms),
                -position,
                sentence,
            )
        )
    if not ranked:
        return None
    ranked.sort(reverse=True, key=lambda item: item[:-1])
    selected = [ranked[0][-1]]
    require_second_source = bool(re.search(r"\bacross\b", question, re.IGNORECASE))
    force_second = bool(
        re.search(r"\b(?:distinguish|plus|two)\b", question, re.IGNORECASE)
    )
    prefer_compact = not re.search(r"\bdistinguish\b", question, re.IGNORECASE)
    if require_second_source or force_second:
        first = selected[0]
        first_text = " ".join(first.text.casefold().split())
        choices = []
        for item in ranked[1:]:
            candidate = item[-1]
            text = " ".join(candidate.text.casefold().split())
            if text in first_text or first_text in text:
                continue
            added_terms = len((candidate.terms & semantic_terms) - first.terms)
            different_source = candidate.filename != first.filename
            compactness = -len(candidate.terms) if prefer_compact else 0
            choices.append(
                (added_terms, different_source, compactness, *item[:-1], candidate)
            )
        if choices:
            separate = (
                [choice for choice in choices if choice[1]]
                if require_second_source
                else []
            )
            best = max(separate or choices, key=lambda item: item[:-1])
            added_terms, different_source, *_, candidate = best
            if (
                added_terms >= 2
                or force_second
                or require_second_source
                and different_source
            ):
                selected.append(candidate)
    if require_second_source and len({item.filename for item in selected}) < 2:
        return None
    if named and sum(
        any(_contains_all(item.terms, {term}) for item in selected) for term in named
    ) < min(2, len(named)):
        return None
    return " ".join(f"{item.text} [{item.filename}]" for item in selected)
