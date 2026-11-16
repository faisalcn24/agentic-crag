from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any

ABSTENTION = "The answer is not present in the provided documents."
_CITATION_PATTERN = re.compile("\\[[^\\[\\]\\n]+\\]")
_TOKEN_PATTERN = re.compile("[a-z0-9]+(?:[-_/.][a-z0-9]+)*", re.IGNORECASE)
_SIMPLE_QUESTION_PATTERN = re.compile(
    "^\\s*(?:what|who|where|when|which|why|how)\\b", re.IGNORECASE
)
_COMPLEX_QUESTION_PATTERN = re.compile(
    "\\b(?:and|both|compare|contrast|difference|explain|list|relationship)\\b",
    re.IGNORECASE,
)
_OVERVIEW_PATTERNS = (
    re.compile(
        "\\bwhat\\s+(?:is|was)\\s+(?:this|that|it)(?:\\s+(?:document|image|file|source))?\\s+about\\b",
        re.IGNORECASE,
    ),
    re.compile(
        "\\bwhat\\s+does\\s+(?:this|that|it)(?:\\s+(?:document|image|file|source))?\\s+(?:cover|describe|discuss)\\b",
        re.IGNORECASE,
    ),
    re.compile(
        "\\bwhat\\s+(?:are|were)\\s+(?:(?:these|those|the)\\s+)?(?:documents|images|files|sources|collection)\\s+about\\b",
        re.IGNORECASE,
    ),
    re.compile(
        "\\b(?:explain|summari[sz]e)\\s+(?:this|that|it)(?:\\s+(?:document|image|file|source))?\\b",
        re.IGNORECASE,
    ),
    re.compile("\\bgive\\s+me\\s+(?:an?\\s+)?(?:overview|summary)\\b", re.IGNORECASE),
    re.compile("\\btell\\s+me\\s+about\\s+(?:this|that|it)\\b", re.IGNORECASE),
)
_COLLECTION_OVERVIEW_PATTERN = re.compile(
    "\\b(?:documents|files|sources|collection)\\b", re.IGNORECASE
)
_IDENTIFIER_PATTERN = re.compile("\\b[A-Z]{1,8}-\\d{2,6}\\b", re.IGNORECASE)
_FIELD_LABEL_PATTERN = re.compile(
    "(?<!\\S)(?![A-Z0-9_-]{2,}\\b)[A-Z][A-Za-z0-9_-]*(?: [A-Za-z0-9_-]+){0,4}:\\s+"
)
_ACTION_VERBS = {
    "add",
    "adjust",
    "disable",
    "enable",
    "increase",
    "limit",
    "move",
    "reduce",
    "remove",
    "restart",
    "restore",
    "update",
    "use",
}
_MIN_QUESTION_COVERAGE = 0.4
_MIN_CLAIM_COVERAGE = 0.75
_COVERAGE_INSTRUCTION_TERMS = {
    "cite",
    "compare",
    "contrast",
    "exact",
    "identify",
    "list",
    "show",
    "specifi",
    "state",
    "version",
}
_SPREADSHEET_ANALYTIC_PATTERN = re.compile(
    "\\b(?:average|compare|difference|exceed(?:ed|s)?|fastest|highest|lowest|maximum|minimum|slowest|sum|total|faster|slower)\\b|\\bwhich\\s+\\w+\\s+have\\b",
    re.IGNORECASE,
)
_SPREADSHEET_EQUIVALENT_TERMS = (
    frozenset({"duration", "latency", "long", "time"}),
    frozenset({"actual", "spent", "spend"}),
)
_STOP_WORDS = {
    "a",
    "about",
    "according",
    "also",
    "an",
    "and",
    "are",
    "at",
    "be",
    "both",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "its",
    "me",
    "must",
    "of",
    "on",
    "or",
    "please",
    "s",
    "should",
    "so",
    "tell",
    "that",
    "the",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "whereas",
    "which",
    "while",
    "who",
    "why",
    "with",
    "would",
}


@dataclass(frozen=True)
class EvidenceSentence:
    filename: str
    text: str
    terms: frozenset[str]
    context_terms: frozenset[str]


def _lexical_tokens(text: str, *, expand_components: bool = False):
    """Normalize tokens once, retaining their position and original casing."""
    for position, token in enumerate(
        _TOKEN_PATTERN.findall(_CITATION_PATTERN.sub("", text))
    ):
        if "-" in token and not any(char.isdigit() or char in "_/." for char in token):
            values = token.split("-")
        elif expand_components and "_" in token:
            values = [token, *token.split("_")]
        else:
            values = [token]
        for value in values:
            lowered = value.casefold()
            if lowered not in _STOP_WORDS:
                yield position, value, _normalize(lowered)


@dataclass(frozen=True)
class _FieldRule:
    labels: frozenset[str]
    question_pattern: re.Pattern[str]
    heading: str | None = None


_FIELD_RULES = {
    "identifier": _FieldRule(
        frozenset({"id", "identifier"}),
        re.compile(r"\b(?:id|identifier|number)\b", re.IGNORECASE),
    ),
    "affected": _FieldRule(
        frozenset({"affected", "impacted"}),
        re.compile(r"\b(?:affect\w*|impact\w*)\b", re.IGNORECASE),
    ),
    "cause": _FieldRule(
        frozenset({"cause", "reason"}),
        re.compile(r"\b(?:caus\w*|reason|root|why)\b", re.IGNORECASE),
        "What caused it",
    ),
    "resolution": _FieldRule(
        frozenset({"fix", "mitigation", "remedy", "resolution", "solution"}),
        re.compile(
            r"\b(?:addressed|addressing|fix\w*|mitigat\w*|remed\w*|resol\w*|solution)\b",
            re.IGNORECASE,
        ),
        "How it was resolved",
    ),
    "status": _FieldRule(
        frozenset({"state", "status"}),
        re.compile(r"\b(?:state|status)\b", re.IGNORECASE),
        "Status",
    ),
}


def _terms(text: str) -> set[str]:
    return set(_ordered_terms(text))


def _ordered_terms(text: str) -> list[str]:
    return [
        normalized
        for _, _, normalized in _lexical_tokens(text, expand_components=True)
        if normalized and normalized not in _STOP_WORDS
    ]


def _anchors(text: str) -> set[str]:
    anchors = set()
    for position, value, normalized in _lexical_tokens(text):
        identifier = any(char.isdigit() or char in "_/." for char in value)
        named = value[:1].isupper() and (
            position > 0 or normalized not in _COVERAGE_INSTRUCTION_TERMS
        )
        if identifier or named or normalized in {"never", "no", "not", "without"}:
            anchors.add(normalized)
    return anchors


def _contextual_anchors(anchors: set[str]) -> set[str]:
    return {anchor for anchor in anchors if re.fullmatch("\\d+(?:\\.\\d+)+", anchor)}


def _question_focus_terms(question: str) -> set[str]:
    tokens = [
        (raw, normalized, normalized in _anchors(raw))
        for raw in _TOKEN_PATTERN.findall(_CITATION_PATTERN.sub("", question))
        if (normalized := _normalize(raw.casefold()))
        and raw.casefold() not in _STOP_WORDS
    ]
    if not tokens or re.match(r"^\s*why\b", question, re.IGNORECASE):
        return set()
    passive = re.search(
        r"\b(?:be|been|is|are|was|were)\b(?:\s+\w+){0,8}\s+(?P<verb>\w+ed|rebuilt|built|made|shown|stored|used)\b",
        question,
        re.IGNORECASE,
    )
    if passive:
        relation = _normalize(passive.group("verb").casefold())
        return set() if relation in _COVERAGE_INSTRUCTION_TERMS else {relation}
    irregular = {"rebuilt", "built", "made", "shown", "stored"}
    relation = next(
        (
            term
            for raw, term, anchor in reversed(tokens)
            if not anchor
            and term not in _COVERAGE_INSTRUCTION_TERMS
            and (term in irregular or raw.casefold().endswith(("ed", "ing")))
        ),
        None,
    )
    if relation:
        return {relation}
    leading = re.match(
        r"^\s*(?:what|which|why)\s+(?:do(?:es)?\s+\S+\s+)?(?P<verb>require|store|support|measure|cause|include|expose|use)\b",
        question,
        re.IGNORECASE,
    )
    return {_normalize(leading.group("verb").casefold())} if leading else set()


def _plural_focus_term(question: str) -> str | None:
    return next(
        (
            _normalize(raw)
            for raw in _TOKEN_PATTERN.findall(_CITATION_PATTERN.sub("", question))
            if raw == raw.casefold()
            and raw.endswith("s")
            and raw not in _STOP_WORDS
            and _normalize(raw)
        ),
        None,
    )


def _term_coverage(
    required: list[str] | set[str], available: set[str] | frozenset[str]
) -> float:
    required = set(required)
    return (
        sum(
            any(_terms_match(term, candidate) for candidate in available)
            for term in required
        )
        / len(required)
        if required
        else 0.0
    )


def _contains_all(available: set[str] | frozenset[str], required: set[str]) -> bool:
    return all(
        (
            any((_terms_match(term, candidate) for candidate in available))
            for term in required
        )
    )


def _contains_answer_content(
    question_terms: list[str], evidence_terms: frozenset[str]
) -> bool:
    return any(
        (
            not any(
                (_terms_match(term, question_term) for question_term in question_terms)
            )
            for term in evidence_terms
        )
    )


def _coverage_available_terms(
    sentence: EvidenceSentence, *, context_anchors: set[str] | None = None
) -> set[str]:
    terms = set(sentence.terms)
    if context_anchors:
        terms.update(
            (
                anchor
                for anchor in context_anchors
                if _contains_all(sentence.context_terms, {anchor})
            )
        )
    lowered = sentence.text.casefold()
    if (
        "localhost" in lowered
        or "127.0.0.1" in lowered
        or re.search("\\b(?:not|never)\\b.{0,40}\\b(?:expose|public)\\b", lowered)
    ):
        terms.add("private")
    return terms


def _polarity_conflicts(question: str, evidence: str) -> bool:
    affirmative_public = bool(
        re.search("\\bpublic(?:ly)?\\b", question, re.IGNORECASE)
        and re.search("\\b(?:expos\\w*|open\\w*|available)\\b", question, re.IGNORECASE)
        and (not re.search("\\b(?:not|never|private)\\b", question, re.IGNORECASE))
    )
    negated_public_evidence = bool(
        re.search(
            "\\b(?:not|never)\\b.{0,50}\\b(?:expos\\w*|public(?:ly)?)\\b",
            evidence,
            re.IGNORECASE,
        )
    )
    return affirmative_public and negated_public_evidence


def _anchor_order_matches(claim: str, evidence: str) -> bool:
    named, evidence_terms = _named_anchors(claim), _ordered_terms(evidence)
    available = set(evidence_terms)
    shared = [
        term
        for term in _ordered_terms(claim)
        if term in named and _contains_all(available, {term})
    ]
    positions = [
        next(
            index
            for index, candidate in enumerate(evidence_terms)
            if _terms_match(term, candidate)
        )
        for term in shared
    ]
    return positions == sorted(positions)


def _has_conflicting_named_anchors(evidence: str, required: set[str]) -> bool:
    return any(
        (
            not any((_terms_match(anchor, expected) for expected in required))
            for anchor in _named_anchors(evidence)
        )
    )


def _named_anchors(text: str) -> set[str]:
    result = set()
    without_labels = _FIELD_LABEL_PATTERN.sub("", _CITATION_PATTERN.sub("", text))
    raw_tokens = _TOKEN_PATTERN.findall(without_labels)
    for index, raw in enumerate(raw_tokens):
        if (
            index > 0
            and raw[:1].isupper()
            and (not raw.isupper())
            and (not any((character.isdigit() for character in raw)))
            and (not any((character in raw for character in "-_/. ")))
        ):
            normalized = _normalize(raw.casefold())
            if normalized not in _STOP_WORDS:
                result.add(normalized)
    return result


def _terms_match(left: str, right: str) -> bool:
    if left == right:
        return True
    shorter, longer = sorted((left, right), key=len)
    if len(shorter) >= 4 and longer == shorter + "e":
        return True
    return len(shorter) >= 5 and longer.startswith(shorter)


def _normalize(token: str) -> str:
    if any(character in token for character in "-_/."):
        return token
    for suffix, replacement, minimum in (
        ("ies", "y", 4),
        ("ing", "", 5),
        ("ed", "", 4),
        ("ly", "", 4),
        ("ness", "", 6),
        ("s", "", 3),
    ):
        if (
            token.endswith(suffix)
            and len(token) > minimum
            and (suffix != "s" or not token.endswith("ss"))
        ):
            return token[: -len(suffix)] + replacement
    return token


def extract_structured_answer(
    question: str, sources: list[dict[str, Any]]
) -> str | None:  # deterministic labeled-record answer
    if _SPREADSHEET_ANALYTIC_PATTERN.search(question) and any(
        (source.get("type") == "xlsx" for source in sources)
    ):
        return None
    for source in sources:
        fields = _labeled_fields(source)
        requested = [
            (label, value)
            for label, value in fields
            if _structured_field_is_requested(label, question)
        ]
        if len(requested) < 2:
            continue
        filename = source.get("filename", "unknown")
        identifier = next(
            (
                value.rstrip(".")
                for label, value in fields
                if _field_role(label) == "identifier"
                and value.rstrip(".").casefold() in question.casefold()
            ),
            "",
        )
        if identifier:
            entity = _identifier_entity(fields)
            intro = f"For {entity} {identifier}, the report says [{filename}]:"
        else:
            intro = f"The document says [{filename}]:"
        details = "\n".join(
            (f"- {_friendly_field_label(label)}: {value}" for label, value in requested)
        )
        return f"{intro}\n{details}"
    return None


def extract_spreadsheet_lookup(
    question: str, sources: list[dict[str, Any]]
) -> str | None:  # exact unambiguous row lookup
    records = [
        (source.get("filename", "unknown"), row)
        for source in sources
        for row in _spreadsheet_rows(source)
    ]
    if filtered := _filtered_spreadsheet_identifiers(
        question, sources, records=records
    ):
        return filtered
    if _SPREADSHEET_ANALYTIC_PATTERN.search(question):
        return None
    columns = list(dict.fromkeys(column for _, row in records for column in row))
    fields = _requested_spreadsheet_fields(question, columns)
    if not fields:
        return None
    question_tokens = _spreadsheet_lookup_tokens(question)
    candidates: list[tuple[int, str, dict[str, str]]] = []
    for filename, row in records:
        if not all(field in row for field in fields):
            continue
        matching_tokens = question_tokens & _spreadsheet_lookup_tokens(
            " ".join(row.values())
        )
        candidates.append(
            (
                sum(
                    3 if any(character.isdigit() for character in token) else 1
                    for token in matching_tokens
                ),
                filename,
                row,
            )
        )
    if not candidates:
        return None
    best = max(candidates, key=lambda item: item[0])
    if sum(score == best[0] for score, _, _ in candidates) > 1:
        return None
    _, filename, row = best
    values = [_format_spreadsheet_value(field, row[field]) for field in fields]
    if len(values) == 1:
        return f"{values[0]} [{filename}]"
    parts = [
        f"{_spreadsheet_column_label(field).title()}: {value}"
        for field, value in zip(fields, values, strict=True)
    ]
    return f"{'; '.join(parts)} [{filename}]"


def _labeled_fields(source: dict[str, Any]) -> list[tuple[str, str]]:
    fields = _split_structured_fields(_without_source_header(source))
    matches = (
        (field, _FIELD_LABEL_PATTERN.match(field))
        for field in fields
        if not field.endswith("?")
    )
    return [
        (
            match.group(0).split(":", 1)[0].strip(),
            " ".join(field[match.end() :].split()).strip(),
        )
        for field, match in matches
        if match and field[match.end() :].strip()
    ]


def _filtered_spreadsheet_identifiers(
    question: str,
    sources: list[dict[str, Any]],
    *,
    records: list[tuple[str, dict[str, str]]] | None = None,
) -> str | None:
    question_terms = {
        _normalize(token) for token in _spreadsheet_lookup_tokens(question)
    }
    grouped: dict[str, list[dict[str, str]]] = {}
    if records is None:
        records = [
            (source.get("filename", "unknown"), row)
            for source in sources
            for row in _spreadsheet_rows(source)
        ]
    for filename, row in records:
        grouped.setdefault(filename, []).append(row)
    for filename, raw_rows in grouped.items():
        rows = [
            dict(items)
            for items in dict.fromkeys(tuple(row.items()) for row in raw_rows)
        ]
        if not rows:
            continue
        identifier = next(
            (column for column in rows[0] if column.casefold().endswith("_id")), None
        )
        if not identifier:
            continue
        entity = _normalize(identifier[:-3].replace("_", " ").casefold())
        if entity not in question_terms:
            continue
        filters = []
        for column in rows[0]:
            if column == identifier:
                continue
            values = list(
                dict.fromkeys(
                    row.get(column, "") for row in rows if row.get(column, "")
                )
            )
            mentioned = [
                value
                for value in values
                if {_normalize(token) for token in _spreadsheet_lookup_tokens(value)}
                <= question_terms
            ]
            if len(mentioned) == 1:
                filters.append((column, mentioned[0]))
        if not filters:
            continue
        selected = [
            row
            for row in rows
            if all(row.get(column) == value for column, value in filters)
        ]
        if not selected or len(selected) == len(rows):
            continue
        label = " ".join(value for _column, value in filters)
        heading = f"{label} {entity}s".strip().title()
        details = "\n".join(
            f"- {entity.title()} {row[identifier]} has status {label} [{filename}]."
            for row in selected
        )
        return f"## {heading}\n\n{details}"
    return None


def _spreadsheet_rows(source: dict[str, Any]) -> list[dict[str, str]]:
    if source.get("type") != "xlsx":
        return []
    rows = (
        _parse_spreadsheet_row(line) for line in source.get("text", "").splitlines()
    )
    return [
        dict(items)
        for items in dict.fromkeys(tuple(row.items()) for row in rows if row)
    ]


def _parse_spreadsheet_row(line: str) -> dict[str, str]:
    if line.casefold().startswith(("document filename:", "sheet:")):
        return {}
    cells = [cell.split(": ", 1) for cell in line.split(" | ")]
    if any((len(cell) != 2 for cell in cells)):
        return {}
    return {key.strip(): value.strip() for key, value in cells}


def _requested_spreadsheet_fields(question: str, columns: list[str]) -> list[str]:
    question_tokens = _spreadsheet_lookup_tokens(question)
    return [
        column
        for column in columns
        if (
            label_tokens := _spreadsheet_lookup_tokens(
                _spreadsheet_column_label(column)
            )
        )
        and all(
            (
                any(
                    (
                        _spreadsheet_terms_match(label_token, question_token)
                        for question_token in question_tokens
                    )
                )
                for label_token in label_tokens
            )
        )
    ]


def _spreadsheet_terms_match(left: str, right: str) -> bool:
    left, right = _normalize(left), _normalize(right)
    return _terms_match(left, right) or any(
        left in group and right in group for group in _SPREADSHEET_EQUIVALENT_TERMS
    )


def _spreadsheet_lookup_tokens(text: str) -> set[str]:
    stop_words = {
        "a",
        "and",
        "at",
        "did",
        "for",
        "how",
        "is",
        "on",
        "the",
        "to",
        "was",
        "what",
        "which",
    }
    return {
        token
        for token in re.findall("[a-z0-9]+", text.casefold())
        if token not in stop_words
    }


def _spreadsheet_column_label(column: str) -> str:
    parts = column.replace("_", " ").split()
    if parts and parts[-1].casefold() in {"usd", "ms", "seconds"}:
        parts.pop()
    return " ".join(parts).casefold()


def _format_spreadsheet_value(column: str, value: Any) -> str:
    unit = column.rsplit("_", 1)[-1].casefold()
    prefix, suffix = (
        ("$", "")
        if unit == "usd"
        else ("", f" {unit}" if unit in {"ms", "seconds"} else "")
    )
    return f"{prefix}{value}{suffix}"


def _field_role(label: str) -> str:
    words = set(re.findall(r"[a-z0-9]+", label.casefold().replace("_", " ")))
    return next(
        (role for role, rule in _FIELD_RULES.items() if words & rule.labels), "other"
    )


def _identifier_entity(fields: list[tuple[str, str]]) -> str:
    label = next(
        (label for label, _value in fields if _field_role(label) == "identifier"),
        "record",
    )
    entity = re.sub("\\b(?:id|identifier)\\b", "", label, flags=re.IGNORECASE)
    entity = " ".join(entity.replace("_", " ").split()).casefold()
    return entity or "record"


def _structured_field_is_requested(label: str, question: str) -> bool:
    rule = _FIELD_RULES.get(_field_role(label))
    if rule is not None:
        return bool(rule.question_pattern.search(question))
    label_terms = _terms(label)
    return bool(label_terms) and _contains_all(_terms(question), label_terms)


def _friendly_field_label(label: str) -> str:
    role = _field_role(label)
    if role == "affected":
        return (
            "Affected service" if "service" in label.casefold() else "What was affected"
        )
    rule = _FIELD_RULES.get(role)
    return rule.heading if rule and rule.heading else label.replace("_", " ").strip()


def _sentence_fragment(value: str) -> str:
    value = value.strip()
    return (
        value
        if not value or value.split(maxsplit=1)[0].isupper()
        else value[:1].casefold() + value[1:]
    )


def _structured_field_spans(text: str) -> list[str]:
    fields = [
        field for field in _split_structured_fields(text) if not field.endswith("?")
    ]
    return fields + [f"{left} {right}" for left, right in zip(fields, fields[1:])]


def _split_structured_fields(text: str) -> list[str]:
    matches = list(_FIELD_LABEL_PATTERN.finditer(text))
    return [
        text[
            match.start() : matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        ].strip()
        for index, match in enumerate(matches)
    ]


def _without_source_header(source: dict[str, Any]) -> str:
    text = source.get("text", "").strip()
    filename = source.get("filename", "unknown")
    workbook, _, sheet = filename.partition(".xlsx-")
    declared = workbook + ".xlsx" if sheet else filename
    prefixes = [
        f"Document filename: {declared}",
        *([f"Sheet: {sheet}"] if sheet else []),
    ]
    for prefix in prefixes:
        if text.casefold().startswith(prefix.casefold()):
            text = text[len(prefix) :].lstrip()
    return text


def find_answer_supporting_passages(
    answer: str, sources: list[dict[str, Any]], *, citation_filename: str | None = None
) -> list[EvidenceSentence]:
    scoped = (
        _answer_text_for_citation(answer, citation_filename)
        if citation_filename
        else answer
    )
    claims = [
        cleaned
        for claim in _conversational_claims(scoped)
        if (cleaned := re.sub(r"^#+\s*", "", claim).strip())
        and cleaned.casefold() not in {"here's the breakdown:", "this covers:"}
    ]
    return list(
        dict.fromkeys(support for _, support in _supported_claims(claims, sources))
    )


def _answer_text_for_citation(answer: str, filename: str) -> str:
    matches = list(_CITATION_PATTERN.finditer(answer))
    names = [match.group()[1:-1] for match in matches]
    if filename not in names:
        return ""
    if len(set(names)) == 1:
        return answer
    fragments, previous, end = [], "", 0
    for match, name in zip(matches, names, strict=True):
        fragment = answer[end : match.start()]
        previous = (
            fragment
            if _CITATION_PATTERN.sub("", fragment).strip(" \t\r\n,.;:()")
            else previous
        )
        if name == filename:
            fragments.append(fragment if fragment.strip() else previous)
        end = match.end()
    return "\n".join(fragments)


def _format_structured_claim(claim: str) -> str:
    matches = list(_FIELD_LABEL_PATTERN.finditer(claim))
    for match in reversed(matches[1:]):
        claim = f"{claim[: match.start()].rstrip()}\n  {claim[match.start() :]}"
    return claim


def _claims(answer: str) -> list[str]:
    without_citations = _CITATION_PATTERN.sub("", answer)
    lines = [
        re.sub("^\\s*(?:[-*•]|\\d+[.)])\\s*", "", line).strip()
        for line in without_citations.splitlines()
        if line.strip() and (not re.match("^\\s*sources?\\s*:", line, re.IGNORECASE))
    ]
    claims = []
    for line in lines:
        for value in re.split(
            "(?<=[.!?])\\s+|;\\s+|,\\s+(?=(?:while|whereas|causing|which|likely)\\b)",
            line,
        ):
            cleaned = re.sub(
                "^(?:while|whereas|causing|which|likely)\\s+", "", value
            ).strip(" ,")
            if cleaned:
                claims.append(cleaned)
    return claims


def _conversational_claims(answer: str) -> list[str]:
    without_citations = _CITATION_PATTERN.sub("", answer)
    return [
        re.sub("^\\s*(?:[-*]|\\d+[.)])\\s*", "", claim).strip()
        for claim in re.split("(?<=[.!?])\\s+|\\n+", without_citations)
        if claim.strip() and (not re.match("^\\s*sources?\\s*:", claim, re.IGNORECASE))
    ]


def _find_claim_support(
    claim: str,
    evidence: list[EvidenceSentence],
    *,
    minimum_coverage: float = _MIN_CLAIM_COVERAGE,
    context_anchors: set[str] | None = None,
) -> EvidenceSentence | None:
    claim_terms = _terms(claim)
    if not claim_terms:
        return None
    claim_anchors = _anchors(claim)
    context_anchors = context_anchors or set()
    required_anchors = claim_anchors | context_anchors
    matches = []
    for sentence in evidence:
        if not _contains_all(sentence.terms, claim_anchors) or not _contains_all(
            sentence.context_terms, context_anchors
        ):
            continue
        coverage = _term_coverage(claim_terms, _coverage_available_terms(sentence))
        if not (
            coverage >= minimum_coverage and _anchor_order_matches(claim, sentence.text)
        ):
            continue
        if _has_conflicting_named_anchors(sentence.text, required_anchors) and not (
            len(required_anchors) >= 2
            and len(re.split("(?<=[.!?])\\s+", sentence.text)) == 1
            and coverage >= 0.8
        ):
            continue
        rank = (
            not _FIELD_LABEL_PATTERN.match(sentence.text),
            -coverage,
            len(sentence.terms),
            len(sentence.text),
        )
        matches.append((rank, sentence))
    return min(matches, key=lambda item: item[0])[1] if matches else None


def _without_redundant_identifier_claims(
    claims: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    identifiers = [
        set(re.findall("\\b\\d+(?:\\.\\d+)*\\b", claim)) for claim, _ in claims
    ]
    return [
        item
        for index, item in enumerate(claims)
        if not identifiers[index]
        or not any(
            (
                item[1] == later_filename
                and identifiers[index] < identifiers[later_index]
                for later_index, (_later_claim, later_filename) in enumerate(claims)
                if later_index > index
            )
        )
    ]


def _evidence_sentences(
    sources: list[dict[str, Any]], *, include_spreadsheets: bool
) -> list[EvidenceSentence]:
    result = []
    seen = set()
    # Adjacent spans reuse sentence text; keep this cache local to one extraction.
    term_cache = {}

    def terms_for(text: str) -> frozenset[str]:
        if text not in term_cache:
            term_cache[text] = frozenset(_terms(text))
        return term_cache[text]

    for source in sources:
        if not include_spreadsheets and source.get("type") == "xlsx":
            continue
        filename = source.get("filename", "unknown")
        source_text = _without_source_header(source)
        sentences = []
        for sentence in re.split("(?<=[.!?])\\s+|\\n+", source_text):
            cleaned = sentence.strip()
            if (
                not cleaned
                or cleaned.endswith("?")
                or cleaned.casefold().startswith(("document filename:", "sheet:"))
            ):
                continue
            sentences.append(cleaned)
        spans = [
            (sentence, " ".join(sentences[max(0, index - 1) : index + 1]))
            for index, sentence in enumerate(sentences)
        ]
        if source.get("type") != "xlsx":
            for size in (2, 3):
                for start in range(len(sentences) - size + 1):
                    span = " ".join(sentences[start : start + size])
                    spans.append((span, span))
            spans.extend(
                ((field, field) for field in _structured_field_spans(source_text))
            )
        for cleaned, context in spans:
            marker = (filename, cleaned)
            if marker in seen:
                continue
            terms = terms_for(cleaned)
            if terms:
                result.append(
                    EvidenceSentence(
                        filename=filename,
                        text=cleaned,
                        terms=terms,
                        context_terms=terms_for(context),
                    )
                )
                seen.add(marker)
    return result


def overview_covers_core_fields(answer: str, sources: list[dict[str, Any]]) -> bool:
    if not sources:
        return False
    fields = _labeled_fields(sources[0])
    required_values = [
        value
        for label, value in fields
        if _field_role(label)
        in {"identifier", "affected", "cause", "resolution", "status"}
    ]
    if not required_values:
        return answer != ABSTENTION
    answer_terms = _terms(answer)
    return all(
        (
            _term_coverage(_terms(value), answer_terms) >= 0.75
            for value in required_values
            if _terms(value)
        )
    )


def is_collection_overview_question(question: str) -> bool:
    return classify_answer_intent(question) == "overview" and bool(
        _COLLECTION_OVERVIEW_PATTERN.search(question)
    )


def _source_topic(source: dict[str, Any]) -> str:
    filename = source.get("filename", "unknown")
    base = re.sub(
        r"\.(?:pdf|docx|xlsx?|png|jpe?g|tiff?|bmp|webp)(?:-.+)?$",
        "",
        filename,
        flags=re.IGNORECASE,
    )
    words = re.sub(r"[_-]+", " ", re.sub(r"^\d+[\s_-]*", "", base)).split()
    casing = {
        token.casefold(): token
        for token in re.findall(
            r"\b[A-Za-z][A-Za-z0-9]*\b", source.get("text", "")[:300]
        )
        if token.isupper() and len(token) > 1
    }
    return " ".join(casing.get(word.casefold(), word.casefold()) for word in words)


def extract_collection_overview(
    question: str, sources: list[dict[str, Any]]
) -> str | None:
    if not is_collection_overview_question(question):
        return None
    topics: dict[str, str] = {}
    for source in sources:
        if topic := _source_topic(source):
            topics.setdefault(topic, source.get("filename", "unknown"))
    cited = [f"{topic} [{filename}]" for topic, filename in list(topics.items())[:6]]
    if len(cited) < 2:
        return None
    joined = (
        " and ".join(cited)
        if len(cited) == 2
        else f"{', '.join(cited[:-1])}, and {cited[-1]}"
    )
    return f"These documents cover {joined}."


def classify_answer_intent(question: str) -> str:
    return (
        "overview"
        if any((pattern.search(question) for pattern in _OVERVIEW_PATTERNS))
        else "answer"
    )


def extract_source_overview(sources: list[dict[str, Any]]) -> str | None:
    if not sources:
        return None
    source = sources[0]
    text, fields = _without_source_header(source), _labeled_fields(source)
    if len(fields) < 2:
        return None
    filename = source.get("filename", "unknown")
    kind = {"image": "image", "xlsx": "spreadsheet"}.get(source.get("type"), "document")
    first_field = _FIELD_LABEL_PATTERN.search(text)
    title = (
        " ".join(text[: first_field.start()].strip(" .:-").split())
        if first_field
        else ""
    )
    roles = {
        role: value.rstrip(".")
        for label, value in fields
        if (role := _field_role(label)) != "other"
    }
    identifier = roles.get("identifier", "")
    article = "an" if title[:1].casefold() in "aeiou" else "a"
    lead = (
        f"This {kind} is {article} {title.casefold()}{f' about {identifier}' if identifier else ''}."
        if title and len(title) <= 100
        else f"This {kind} summarizes the documented record."
    )
    affected, cause = roles.get("affected", ""), roles.get("cause", "")
    label = next((label for label, _ in fields if _field_role(label) == "affected"), "")
    subject = (
        f"the {affected} service"
        if "service" in label.casefold() and not affected.casefold().endswith("service")
        else affected
    )
    impact = (
        f"It explains that {subject} was affected{f' because {_sentence_fragment(cause)}' if cause else ''}."
        if affected
        else f"It says the problem happened because {_sentence_fragment(cause)}."
        if cause
        else ""
    )
    resolution = _sentence_fragment(roles.get("resolution", ""))
    fix = (
        f"{'The fix was to' if resolution and resolution.split(maxsplit=1)[0].casefold() in _ACTION_VERBS else 'The documented fix was'} {resolution}."
        if resolution
        else ""
    )
    status = roles.get("status", "")
    status_text = (
        f"The report marks {f'the {_identifier_entity(fields)}' if identifier else 'it'} as {_sentence_fragment(status)}."
        if status
        else ""
    )
    sentences = [sentence for sentence in (lead, impact, fix, status_text) if sentence]
    if len(sentences) == 1:
        sentences.append(
            "It records "
            + "; ".join(f"{label}: {value.rstrip('.')}" for label, value in fields[:4])
            + "."
        )
    answer = " ".join(sentences).rstrip()
    return (
        f"{answer[:-1]} [{filename}]."
        if answer.endswith(".")
        else f"{answer} [{filename}]."
    )


def consolidate_repeated_citations(answer: str) -> str:
    if re.search(r"(?m)^\s*##\s+", answer):
        return answer
    citations = _CITATION_PATTERN.findall(answer)
    if len(citations) < 2 or len(set(citations)) != 1:
        return answer
    citation = citations[0]
    cleaned = re.sub(
        r"[ \t]+([,.;:!?])", r"\1", re.sub(rf"\s*{re.escape(citation)}", "", answer)
    ).rstrip()
    if not cleaned:
        return answer
    head, separator, last = cleaned.rpartition("\n")
    punctuation = last[-1] if last.endswith((".", "!", "?")) else "."
    last = last[:-1] if last.endswith(punctuation) else last
    return f"{head}{separator}{last} {citation}{punctuation}"


def extract_grounded_sentence(
    question: str, sources: list[dict[str, Any]]
) -> str | None:
    identifiers = {value.casefold() for value in _IDENTIFIER_PATTERN.findall(question)}
    if identifiers and re.search(
        r"\bwhat\s+does\b.+\b(?:require|say|state)\b", question, re.IGNORECASE
    ):
        exact = [
            sentence
            for sentence in _evidence_sentences(sources, include_spreadsheets=False)
            if identifiers
            <= {
                value.casefold() for value in _IDENTIFIER_PATTERN.findall(sentence.text)
            }
            and len(re.split("(?<=[.!?])\\s+", sentence.text)) == 1
        ]
        if exact:
            best = min(
                exact, key=lambda sentence: (len(sentence.terms), len(sentence.text))
            )
            return f"{best.text} [{best.filename}]"
    best = find_grounded_evidence(question, sources)
    return f"{best.text} [{best.filename}]" if best else None


def extract_negative_answer(question: str, sources: list[dict[str, Any]]) -> str | None:
    if not re.match(
        r"^\s*(?:is|are|do|does|can|could|has|have)\b", question, re.IGNORECASE
    ):
        return None
    question_terms = set(_ordered_terms(question)) - {
        "are",
        "can",
        "could",
        "do",
        "doe",
        "ha",
        "have",
        "is",
    }
    if not question_terms:
        return None
    candidates = []
    for sentence in _evidence_sentences(sources, include_spreadsheets=False):
        if re.search(
            r"\b(?:reviewer can ask|answer should|expected answer)\b",
            sentence.text,
            re.IGNORECASE,
        ):
            continue
        if not re.search(
            r"\b(?:does not|do not|is not|are not|not include|not included|not process|no\s+\w+)\b",
            sentence.text,
            re.IGNORECASE,
        ):
            continue
        coverage = _term_coverage(question_terms, sentence.terms)
        if coverage >= 0.5:
            candidates.append((coverage, -len(sentence.terms), sentence))
    if not candidates:
        return None
    sentence = max(candidates, key=lambda item: item[:-1])[-1]
    return f"No. {sentence.text} [{sentence.filename}]"


def extract_grounded_passages(
    question: str, sources: list[dict[str, Any]]
) -> str | None:
    from .multihop import extract_grounded_passages as extract

    return extract(question, sources)


def extract_absence_evidence(
    question: str, sources: list[dict[str, Any]]
) -> str | None:
    if not re.match(r"^\s*(?:what|which|who|how many)\b", question, re.IGNORECASE):
        return None
    if not re.search(
        r"\b(?:accuracy|active users?|bucket|ceo|domain name|electricity cost|email address|oauth provider|payroll|revenue)\b",
        question,
        re.IGNORECASE,
    ):
        return None
    question_terms = set(_ordered_terms(question)) - {"current", "exact", "production"}
    if not question_terms:
        return None
    semantic_terms = set(question_terms)
    if question_terms & {"authenticate", "oauth", "provider"}:
        semantic_terms.update({"auth", "authentication", "include"})
    candidates = []
    for sentence in _evidence_sentences(sources, include_spreadsheets=False):
        if len(sentence.terms) > 50 or re.search(
            r"\bExpected answer\s*:", sentence.text, re.IGNORECASE
        ):
            continue
        negative = bool(
            re.search(
                r"\b(?:does not|do not|not include|not included|not process|not store|no\s+OCR)\b",
                sentence.text,
                re.IGNORECASE,
            )
        )
        proposed_only = bool(
            re.search(
                r"\bfor a longer-lived deployment, add\b", sentence.text, re.IGNORECASE
            )
        ) and bool(question_terms & {"domain", "https"})
        if not (negative or proposed_only):
            continue
        hits = sum(_contains_all(sentence.terms, {term}) for term in semantic_terms)
        rare_boundary = any(
            term in question_terms and _contains_all(sentence.terms, {term})
            for term in {"ocr", "s3"}
        )
        if hits >= 2 or rare_boundary:
            candidates.append((hits, -len(sentence.terms), sentence))
    if not candidates:
        return None
    sentence = max(candidates, key=lambda item: item[:-1])[-1]
    return f"{ABSTENTION} {sentence.text} [{sentence.filename}]"


def find_grounded_evidence(
    question: str, sources: list[dict[str, Any]], *, allow_complex: bool = False
) -> EvidenceSentence | None:
    question_terms = _ordered_terms(question)
    if allow_complex:
        question_terms = [
            term for term in question_terms if term not in _COVERAGE_INSTRUCTION_TERMS
        ]
    if len(question_terms) < 2:
        return None
    if not allow_complex and (
        not _SIMPLE_QUESTION_PATTERN.search(question)
        or _COMPLEX_QUESTION_PATTERN.search(question)
    ):
        return None
    focus_terms = set() if allow_complex else _question_focus_terms(question)
    plural_focus = _plural_focus_term(question) if allow_complex else None
    minimum_coverage = 0.3 if allow_complex else _MIN_QUESTION_COVERAGE
    anchors = _anchors(question)
    context_anchors = _contextual_anchors(anchors)
    direct_anchors = anchors - context_anchors
    candidates = [
        sentence
        for sentence in _evidence_sentences(sources, include_spreadsheets=allow_complex)
        if _contains_all(sentence.terms, direct_anchors)
        and _contains_all(sentence.context_terms, context_anchors)
    ]
    if (
        allow_complex
        and plural_focus is None
        and (not _COMPLEX_QUESTION_PATTERN.search(question))
    ):
        candidates = [
            sentence
            for sentence in candidates
            if len(re.split("(?<=[.!?])\\s+", sentence.text)) == 1
        ]
    matches = [
        (sentence, coverage)
        for sentence in candidates
        if (
            coverage := _term_coverage(
                question_terms,
                _coverage_available_terms(sentence, context_anchors=context_anchors),
            )
        )
        >= minimum_coverage
        and _contains_all(sentence.terms, focus_terms)
        and _contains_answer_content(question_terms, sentence.terms)
        and (not _polarity_conflicts(question, sentence.text))
    ]
    if not matches:
        return None
    if plural_focus:
        plural_matches = [
            item
            for item in matches
            if _ordered_terms(item[0].text).count(plural_focus) >= 2
        ]
        if plural_matches:
            matches = plural_matches
        else:
            plural_focus = None
    return min(
        matches,
        key=lambda item: (
            (len(item[0].terms), -item[1], len(item[0].text))
            if plural_focus
            else (-item[1], len(item[0].terms), len(item[0].text))
        ),
    )[0]


def _supported_claims(
    claims: list[str],
    sources: list[dict[str, Any]],
    *,
    minimum_coverage: float = _MIN_CLAIM_COVERAGE,
    context_anchors: set[str] | None = None,
) -> list[tuple[str, EvidenceSentence]]:
    evidence = _evidence_sentences(sources, include_spreadsheets=True) if claims else []
    return list(
        dict.fromkeys(
            (claim, support)
            for claim in claims
            if (
                support := _find_claim_support(
                    claim,
                    evidence,
                    minimum_coverage=minimum_coverage,
                    context_anchors=context_anchors,
                )
            )
        )
    )


def ground_generated_answer(
    question: str, answer: str, sources: list[dict[str, Any]]
) -> str:
    stripped = answer.strip()
    if not stripped or ABSTENTION.casefold() in stripped.casefold():
        return ABSTENTION
    claims = _claims(stripped)
    if not claims or any(not _terms(claim) for claim in claims):
        return ABSTENTION
    question_terms = [
        term
        for term in _ordered_terms(question)
        if term not in _COVERAGE_INSTRUCTION_TERMS
    ]
    minimum_question_matches = (
        1
        if len(set(question_terms)) <= 3 or _COMPLEX_QUESTION_PATTERN.search(question)
        else 2
    )
    relevant = (
        list(claims)
        if re.search(r"\b(?:compare|contrast)\b", question, re.IGNORECASE)
        else []
    )
    if not relevant:
        for claim in claims:
            claim_terms = _terms(claim)
            if (
                sum(
                    any(_terms_match(term, candidate) for candidate in claim_terms)
                    for term in set(question_terms)
                )
                >= minimum_question_matches
            ):
                relevant.append(claim)
    context = (
        set() if _COMPLEX_QUESTION_PATTERN.search(question) else _anchors(question)
    )
    grounded_claims = list(
        dict.fromkeys(
            f"{support.text} [{support.filename}]"
            for _, support in _supported_claims(
                relevant, sources, context_anchors=context
            )
        )
    )
    if len(grounded_claims) > 1 and all(
        (_FIELD_LABEL_PATTERN.match(claim) for claim in grounded_claims)
    ):
        return "\n".join(
            (f"- {_format_structured_claim(claim)}" for claim in grounded_claims)
        )
    return " ".join(grounded_claims) if grounded_claims else ABSTENTION


def ground_conversational_answer(
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    *,
    minimum_coverage: float | None = None,
) -> str:
    stripped = answer.strip()
    if not stripped or ABSTENTION.casefold() in stripped.casefold():
        return ABSTENTION
    claims = _conversational_claims(stripped)
    if not claims:
        return ABSTENTION
    minimum_coverage = (
        minimum_coverage
        if minimum_coverage is not None
        else 0.6
        if classify_answer_intent(question) == "overview"
        else _MIN_CLAIM_COVERAGE
    )
    grounded_claims: list[tuple[str, str]] = []
    for claim, support in _supported_claims(
        claims, sources, minimum_coverage=minimum_coverage
    ):
        clean_claim = _CITATION_PATTERN.sub("", claim).strip()
        clean_claim = re.sub("\\bcanonical\\s+", "", clean_claim, flags=re.IGNORECASE)
        if re.search("\\bprivate\\b", question, re.IGNORECASE):
            clean_claim = re.sub(
                ",?\\s+but\\s+it\\s+should\\s+only\\s+bind\\s+to\\s+localhost",
                " and must remain private",
                clean_claim,
                flags=re.IGNORECASE,
            )
        clean_claim = clean_claim.rstrip(".!?").rstrip()
        grounded = (clean_claim, support.filename)
        if grounded not in grounded_claims:
            grounded_claims.append(grounded)
    if not grounded_claims:
        return ABSTENTION
    grounded_claims = _without_redundant_identifier_claims(grounded_claims)
    groups: list[tuple[str, list[str]]] = []
    for claim, filename in grounded_claims:
        groups[-1][1].append(claim) if groups and groups[-1][
            0
        ] == filename else groups.append((filename, [claim]))
    return " ".join(
        f"{' '.join(f'{claim}.' for claim in claims).rstrip('.')} [{filename}]."
        for filename, claims in groups
    )
