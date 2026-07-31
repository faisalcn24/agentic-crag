from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


ABSTENTION = "The answer is not present in the provided documents."

_CITATION_PATTERN = re.compile(r"\[[^\[\]\n]+\]")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_/.][a-z0-9]+)*", re.IGNORECASE)
_SIMPLE_QUESTION_PATTERN = re.compile(
    r"^\s*(?:what|who|where|when|which|why)\b", re.IGNORECASE
)
_COMPLEX_QUESTION_PATTERN = re.compile(
    r"\b(?:and|both|compare|contrast|difference|explain|how|list|relationship)\b",
    re.IGNORECASE,
)
_MIN_QUESTION_COVERAGE = 0.40
_MIN_CLAIM_COVERAGE = 0.75
_STOP_WORDS = {
    "a",
    "about",
    "according",
    "an",
    "are",
    "at",
    "be",
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
    "of",
    "on",
    "please",
    "s",
    "should",
    "tell",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
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


def extract_grounded_sentence(
    question: str, sources: list[dict[str, Any]]
) -> str | None:
    """Return a directly matching evidence sentence for a simple factual question."""
    question_terms = _ordered_terms(question)
    if len(question_terms) < 2:
        return None

    focus_terms = _question_focus_terms(question)
    matches = [
        sentence
        for sentence in grounding_candidates(question, sources)
        if _term_coverage(question_terms, sentence.terms) >= _MIN_QUESTION_COVERAGE
        and _contains_all(sentence.terms, focus_terms)
        and _contains_answer_content(question_terms, sentence.terms)
    ]
    if not matches:
        return None

    best = min(
        matches,
        key=lambda item: (
            -_term_coverage(question_terms, item.terms),
            len(item.terms),
            len(item.text),
        ),
    )
    return f"{best.text} [{best.filename}]"


def grounding_candidates(
    question: str, sources: list[dict[str, Any]]
) -> list[EvidenceSentence]:
    if not _SIMPLE_QUESTION_PATTERN.search(question):
        return []
    if _COMPLEX_QUESTION_PATTERN.search(question):
        return []
    anchors = _anchors(question)
    return [
        sentence
        for sentence in _evidence_sentences(sources, include_spreadsheets=False)
        if _contains_all(sentence.terms, anchors)
    ]


def ground_generated_answer(
    question: str, answer: str, sources: list[dict[str, Any]]
) -> str:
    """Replace generated claims with the evidence sentences that support them."""
    stripped = answer.strip()
    if not stripped or ABSTENTION.casefold() in stripped.casefold():
        return ABSTENTION

    evidence = _evidence_sentences(sources, include_spreadsheets=True)
    claims = _claims(stripped)
    if not evidence or not claims:
        return ABSTENTION

    grounded_claims = []
    question_anchors = (
        set() if _COMPLEX_QUESTION_PATTERN.search(question) else _anchors(question)
    )
    for claim in claims:
        claim_terms = _terms(claim)
        if not claim_terms:
            return ABSTENTION
        claim_anchors = _anchors(claim)
        required_anchors = question_anchors | claim_anchors
        matches = [
            sentence
            for sentence in evidence
            if _contains_all(sentence.context_terms, question_anchors)
            and _contains_all(sentence.terms, claim_anchors)
            and not _has_conflicting_named_anchors(sentence.text, required_anchors)
            and _term_coverage(claim_terms, sentence.terms) >= _MIN_CLAIM_COVERAGE
            and _anchor_order_matches(claim, sentence.text)
        ]
        if not matches:
            continue
        support = min(
            matches,
            key=lambda item: (
                -_term_coverage(claim_terms, item.terms),
                len(item.terms),
                len(item.text),
            ),
        )
        grounded = f"{support.text} [{support.filename}]"
        if grounded not in grounded_claims:
            grounded_claims.append(grounded)
    return " ".join(grounded_claims) if grounded_claims else ABSTENTION


def _claims(answer: str) -> list[str]:
    without_citations = _CITATION_PATTERN.sub("", answer)
    lines = [
        re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        for line in without_citations.splitlines()
        if line.strip()
        and not re.match(r"^\s*sources?\s*:", line, re.IGNORECASE)
    ]
    claims = []
    for line in lines:
        for value in re.split(
            r"(?<=[.!?])\s+|;\s+|,\s+(?=(?:while|whereas)\b)", line
        ):
            cleaned = re.sub(r"^(?:while|whereas)\s+", "", value).strip(" ,")
            if cleaned:
                claims.append(cleaned)
    return claims


def _evidence_sentences(
    sources: list[dict[str, Any]], *, include_spreadsheets: bool
) -> list[EvidenceSentence]:
    result = []
    seen = set()
    for source in sources:
        if not include_spreadsheets and source.get("type") == "xlsx":
            continue
        filename = source.get("filename", "unknown")
        source_text = _without_source_header(source)
        sentences = []
        for sentence in re.split(
            r"(?<=[.!?])\s+|\n+", source_text
        ):
            cleaned = sentence.strip()
            if (
                not cleaned
                or cleaned.endswith("?")
                or re.match(
                    r"^(?:expected answer|question(?:\s+[a-z0-9]+)?)\s*:",
                    cleaned,
                    re.IGNORECASE,
                )
                or cleaned.casefold().startswith(("document filename:", "sheet:"))
            ):
                continue
            sentences.append(cleaned)

        spans = [
            (
                sentence,
                " ".join(sentences[max(0, index - 1) : index + 1]),
            )
            for index, sentence in enumerate(sentences)
        ]
        if source.get("type") != "xlsx":
            spans.extend(
                (f"{first} {second}", f"{first} {second}")
                for first, second in zip(sentences, sentences[1:])
            )
        for cleaned, context in spans:
            marker = (filename, cleaned)
            if marker in seen:
                continue
            terms = _terms(cleaned)
            if terms:
                result.append(
                    EvidenceSentence(
                        filename=filename,
                        text=cleaned,
                        terms=frozenset(terms),
                        context_terms=frozenset(_terms(context)),
                    )
                )
                seen.add(marker)
    return result


def _without_source_header(source: dict[str, Any]) -> str:
    text = source.get("text", "").strip()
    filename = source.get("filename", "unknown")
    declared_filename = filename
    sheet_name = ""
    if source.get("type") == "xlsx" and ".xlsx-" in filename.casefold():
        split_at = filename.casefold().index(".xlsx-") + len(".xlsx")
        declared_filename = filename[:split_at]
        sheet_name = filename[split_at + 1 :]

    prefixes = [f"Document filename: {declared_filename}"]
    if sheet_name:
        prefixes.append(f"Sheet: {sheet_name}")
    for prefix in prefixes:
        if text.casefold().startswith(prefix.casefold()):
            text = text[len(prefix) :].lstrip()
    return text


def _terms(text: str) -> set[str]:
    return set(_ordered_terms(text))


def _ordered_terms(text: str) -> list[str]:
    terms = []
    for raw in _TOKEN_PATTERN.findall(_CITATION_PATTERN.sub("", text)):
        values = [raw]
        if "-" in raw and not any(character.isdigit() for character in raw):
            values = raw.split("-")
        elif "_" in raw:
            values.extend(raw.split("_"))
        for value in values:
            lowered = value.casefold()
            if lowered in _STOP_WORDS:
                continue
            normalized = _normalize(lowered)
            if normalized and normalized not in _STOP_WORDS:
                terms.append(normalized)
    return terms


def _anchors(text: str) -> set[str]:
    anchors = set()
    for raw in _TOKEN_PATTERN.findall(_CITATION_PATTERN.sub("", text)):
        values = (
            raw.split("-")
            if "-" in raw
            and not any(character.isdigit() for character in raw)
            and not any(character in raw for character in "_/.")
            else [raw]
        )
        for value in values:
            lowered = value.casefold()
            normalized = _normalize(lowered)
            if (
                lowered not in _STOP_WORDS
                and (
                    any(character.isdigit() for character in value)
                    or any(character in value for character in "_/.")
                    or value[:1].isupper()
                    or normalized in {"never", "no", "not", "without"}
                )
            ):
                anchors.add(normalized)
    return anchors


def _question_focus_terms(question: str) -> set[str]:
    """Identify the requested relation or answer type without domain rules."""
    tokens = []
    for raw in _TOKEN_PATTERN.findall(_CITATION_PATTERN.sub("", question)):
        lowered = raw.casefold()
        normalized = _normalize(lowered)
        if normalized and lowered not in _STOP_WORDS:
            tokens.append((raw, normalized, normalized in _anchors(raw)))
    if not tokens:
        return set()

    first_anchor = next(
        (index for index, (_raw, _term, anchor) in enumerate(tokens) if anchor),
        len(tokens),
    )
    if re.search(r"\bdid\b", question, re.IGNORECASE) and first_anchor < len(tokens):
        after_anchor = [
            term for _raw, term, anchor in tokens[first_anchor + 1 :] if not anchor
        ]
        if after_anchor:
            return {after_anchor[0]}
    before_anchor = tokens[:first_anchor]
    for raw, term, _anchor in before_anchor[:2]:
        if raw.casefold().endswith(("ed", "ing")) or (
            len(raw) > 4 and raw.casefold().endswith("s")
        ):
            return {term}
    if before_anchor:
        return {before_anchor[0][1]}
    first_non_anchor = next(
        (term for _raw, term, anchor in tokens if not anchor), tokens[0][1]
    )
    return {first_non_anchor}


def _term_coverage(required: list[str] | set[str], available: set[str] | frozenset[str]) -> float:
    unique_required = set(required)
    if not unique_required:
        return 0.0
    matched = sum(
        any(_terms_match(term, candidate) for candidate in available)
        for term in unique_required
    )
    return matched / len(unique_required)


def _contains_all(available: set[str] | frozenset[str], required: set[str]) -> bool:
    return all(
        any(_terms_match(term, candidate) for candidate in available)
        for term in required
    )


def _contains_answer_content(
    question_terms: list[str], evidence_terms: frozenset[str]
) -> bool:
    return any(
        not any(_terms_match(term, question_term) for question_term in question_terms)
        for term in evidence_terms
    )


def _anchor_order_matches(claim: str, evidence: str) -> bool:
    claim_anchors = [
        term for term in _ordered_terms(claim) if term in _anchors(claim)
    ]
    shared = [term for term in claim_anchors if _contains_all(_terms(evidence), {term})]
    if len(shared) < 2:
        return True
    evidence_anchors = [
        term for term in _ordered_terms(evidence) if term in _anchors(evidence)
    ]
    positions = [
        next(
            index
            for index, candidate in enumerate(evidence_anchors)
            if _terms_match(term, candidate)
        )
        for term in shared
    ]
    return positions == sorted(positions)


def _has_conflicting_named_anchors(evidence: str, required: set[str]) -> bool:
    return any(
        not any(_terms_match(anchor, expected) for expected in required)
        for anchor in _named_anchors(evidence)
    )


def _named_anchors(text: str) -> set[str]:
    result = set()
    raw_tokens = _TOKEN_PATTERN.findall(_CITATION_PATTERN.sub("", text))
    for index, raw in enumerate(raw_tokens):
        if (
            index > 0
            and raw[:1].isupper()
            and not raw.isupper()
            and not any(character.isdigit() for character in raw)
            and not any(character in raw for character in "-_/. ")
        ):
            normalized = _normalize(raw.casefold())
            if normalized not in _STOP_WORDS:
                result.add(normalized)
    return result


def _terms_match(left: str, right: str) -> bool:
    if left == right:
        return True
    shorter, longer = sorted((left, right), key=len)
    return len(shorter) >= 5 and longer.startswith(shorter)


def _normalize(token: str) -> str:
    if any(character in token for character in "-_/."):
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("ly") and len(token) > 4:
        return token[:-2]
    if token.endswith("ness") and len(token) > 6:
        return token[:-4]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token
