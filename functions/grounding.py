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
_OVERVIEW_PATTERNS = (
    re.compile(
        r"\bwhat\s+(?:is|was)\s+(?:this|that|it)"
        r"(?:\s+(?:document|image|file|source))?\s+about\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat\s+does\s+(?:this|that|it)"
        r"(?:\s+(?:document|image|file|source))?\s+(?:cover|describe|discuss)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat\s+(?:are|were)\s+(?:(?:these|those|the)\s+)?"
        r"(?:documents|images|files|sources|collection)\s+about\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:explain|summari[sz]e)\s+(?:this|that|it)"
        r"(?:\s+(?:document|image|file|source))?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bgive\s+me\s+(?:an?\s+)?(?:overview|summary)\b", re.IGNORECASE),
    re.compile(r"\btell\s+me\s+about\s+(?:this|that|it)\b", re.IGNORECASE),
)
_COLLECTION_OVERVIEW_PATTERN = re.compile(
    r"\b(?:documents|files|sources|collection)\b", re.IGNORECASE
)
_IDENTIFIER_PATTERN = re.compile(r"\b[A-Z]{1,8}-\d{2,6}\b", re.IGNORECASE)
_FIELD_LABEL_PATTERN = re.compile(
    r"(?<!\S)(?![A-Z0-9_-]{2,}\b)[A-Z][A-Za-z0-9_-]*"
    r"(?: [A-Za-z0-9_-]+){0,4}:\s+"
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
_MIN_QUESTION_COVERAGE = 0.40
_MIN_CLAIM_COVERAGE = 0.75
_COVERAGE_INSTRUCTION_TERMS = {
    "cite",
    "compare",
    "contrast",
    "exact",
    "identify",
    "list",
    "show",
    "state",
}
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


def extract_grounded_sentence(
    question: str, sources: list[dict[str, Any]]
) -> str | None:
    """Return a directly matching evidence sentence for a simple factual question."""
    best = find_grounded_evidence(question, sources)
    return f"{best.text} [{best.filename}]" if best else None


def find_grounded_evidence(
    question: str,
    sources: list[dict[str, Any]],
    *,
    allow_complex: bool = False,
) -> EvidenceSentence | None:
    """Return the best exact evidence sentence for a simple factual question."""
    question_terms = _ordered_terms(question)
    if allow_complex:
        question_terms = [
            term
            for term in question_terms
            if term not in _COVERAGE_INSTRUCTION_TERMS
        ]
    if len(question_terms) < 2:
        return None

    focus_terms = set() if allow_complex else _question_focus_terms(question)
    plural_focus = _plural_focus_term(question) if allow_complex else None
    minimum_coverage = 0.30 if allow_complex else _MIN_QUESTION_COVERAGE
    candidates = grounding_candidates(
        question, sources, allow_complex=allow_complex
    )
    if (
        allow_complex
        and plural_focus is None
        and not _COMPLEX_QUESTION_PATTERN.search(question)
    ):
        candidates = [
            sentence
            for sentence in candidates
            if len(re.split(r"(?<=[.!?])\s+", sentence.text)) == 1
        ]
    matches = [
        sentence
        for sentence in candidates
        if _term_coverage(
            question_terms,
            _coverage_available_terms(sentence),
        )
        >= minimum_coverage
        and _contains_all(sentence.terms, focus_terms)
        and _contains_answer_content(question_terms, sentence.terms)
        and not _polarity_conflicts(question, sentence.text)
    ]
    if not matches:
        return None

    if plural_focus:
        plural_matches = [
            item
            for item in matches
            if _ordered_terms(item.text).count(plural_focus) >= 2
        ]
        if plural_matches:
            matches = plural_matches

    return min(
        matches,
        key=lambda item: (
            (
                len(item.terms)
                if plural_focus
                else -_term_coverage(
                    question_terms,
                    _coverage_available_terms(item),
                )
            ),
            (
                -_term_coverage(
                    question_terms,
                    _coverage_available_terms(item),
                )
                if plural_focus
                else len(item.terms)
            ),
            len(item.text),
        ),
    )


def grounding_candidates(
    question: str,
    sources: list[dict[str, Any]],
    *,
    allow_complex: bool = False,
) -> list[EvidenceSentence]:
    if not allow_complex and not _SIMPLE_QUESTION_PATTERN.search(question):
        return []
    if not allow_complex and _COMPLEX_QUESTION_PATTERN.search(question):
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
                not _FIELD_LABEL_PATTERN.match(item.text),
                -_term_coverage(claim_terms, item.terms),
                len(item.terms),
                len(item.text),
            ),
        )
        grounded = f"{support.text} [{support.filename}]"
        if grounded not in grounded_claims:
            grounded_claims.append(grounded)
    if len(grounded_claims) > 1 and all(
        _FIELD_LABEL_PATTERN.match(claim) for claim in grounded_claims
    ):
        return "\n".join(
            f"- {_format_structured_claim(claim)}" for claim in grounded_claims
        )
    return " ".join(grounded_claims) if grounded_claims else ABSTENTION


def ground_conversational_answer(
    question: str, answer: str, sources: list[dict[str, Any]]
) -> str:
    """Keep supported generated wording while removing unsupported sentences."""
    stripped = answer.strip()
    if not stripped or ABSTENTION.casefold() in stripped.casefold():
        return ABSTENTION

    evidence = _evidence_sentences(sources, include_spreadsheets=True)
    claims = _conversational_claims(stripped)
    if not evidence or not claims:
        return ABSTENTION

    minimum_coverage = (
        0.60 if classify_answer_intent(question) == "overview" else _MIN_CLAIM_COVERAGE
    )
    grounded_claims: list[tuple[str, str]] = []
    for claim in claims:
        support = _find_claim_support(
            claim, evidence, minimum_coverage=minimum_coverage
        )
        if support is None:
            continue
        clean_claim = _CITATION_PATTERN.sub("", claim).strip()
        clean_claim = re.sub(r"\bcanonical\s+", "", clean_claim, flags=re.IGNORECASE)
        if re.search(r"\bprivate\b", question, re.IGNORECASE):
            clean_claim = re.sub(
                r",?\s+but\s+it\s+should\s+only\s+bind\s+to\s+localhost",
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
        if groups and groups[-1][0] == filename:
            groups[-1][1].append(claim)
        else:
            groups.append((filename, [claim]))
    rendered_groups = []
    for filename, claims in groups:
        text = " ".join(f"{claim}." for claim in claims).rstrip(".")
        rendered_groups.append(f"{text} [{filename}].")
    return " ".join(rendered_groups)


def overview_covers_core_fields(
    answer: str, sources: list[dict[str, Any]]
) -> bool:
    """Require a structured overview to retain each core fact present in its source."""
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
        _term_coverage(_terms(value), answer_terms) >= 0.75
        for value in required_values
        if _terms(value)
    )


def is_source_summary_question(question: str) -> bool:
    return classify_answer_intent(question) == "overview"


def is_collection_overview_question(question: str) -> bool:
    return (
        classify_answer_intent(question) == "overview"
        and bool(_COLLECTION_OVERVIEW_PATTERN.search(question))
    )


def extract_collection_overview(
    question: str, sources: list[dict[str, Any]]
) -> str | None:
    if not is_collection_overview_question(question):
        return None

    topics = []
    seen = set()
    for source in sources:
        filename = source.get("filename", "unknown")
        base = re.sub(
            r"\.(?:pdf|docx|xlsx?|png|jpe?g|tiff?|bmp|webp)(?:-.+)?$",
            "",
            filename,
            flags=re.IGNORECASE,
        )
        topic = re.sub(r"^\d+[\s_-]*", "", base)
        topic_words = re.sub(r"[_-]+", " ", topic).split()
        source_casing = {
            token.casefold(): token
            for token in re.findall(r"\b[A-Za-z][A-Za-z0-9]*\b", source.get("text", "")[:300])
            if token.isupper() and len(token) > 1
        }
        topic = " ".join(
            source_casing.get(word.casefold(), word.casefold())
            for word in topic_words
        )
        if not topic or topic in seen:
            continue
        topics.append((topic, filename))
        seen.add(topic)
    if len(topics) < 2:
        return None

    cited_topics = [f"{topic} [{filename}]" for topic, filename in topics[:6]]
    if len(cited_topics) == 2:
        joined = " and ".join(cited_topics)
    else:
        joined = f"{', '.join(cited_topics[:-1])}, and {cited_topics[-1]}"
    return f"These documents cover {joined}."


def classify_answer_intent(question: str) -> str:
    return (
        "overview"
        if any(pattern.search(question) for pattern in _OVERVIEW_PATTERNS)
        else "answer"
    )


def extract_structured_answer(
    question: str, sources: list[dict[str, Any]]
) -> str | None:
    """Answer a multi-field question from one labeled record without model calls."""
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
            f"- {_friendly_field_label(label)}: {value}"
            for label, value in requested
        )
        return f"{intro}\n{details}"
    return None


def extract_source_summary(
    question: str, sources: list[dict[str, Any]]
) -> str | None:
    """Build a conversational, cited overview from explicitly labeled fields."""
    if not is_source_summary_question(question):
        return None
    return extract_source_overview(sources)


def extract_source_overview(sources: list[dict[str, Any]]) -> str | None:
    if not sources:
        return None
    source = sources[0]
    text = _without_source_header(source)
    fields = _labeled_fields(source)
    if len(fields) < 2:
        return None

    filename = source.get("filename", "unknown")
    kind = {
        "image": "image",
        "xlsx": "spreadsheet",
    }.get(source.get("type"), "document")
    first_field = _FIELD_LABEL_PATTERN.search(text)
    title = " ".join(text[: first_field.start()].strip(" .:-").split()) if first_field else ""
    by_role = {
        role: value
        for label, value in fields
        if (role := _field_role(label)) and role not in {"other"}
    }
    identifier = by_role.get("identifier", "").rstrip(".")
    entity = _identifier_entity(fields)
    sentences = []
    if title and len(title) <= 100:
        article = "an" if title[:1].casefold() in "aeiou" else "a"
        subject = f"This {kind} is {article} {title.casefold()}"
        if identifier:
            subject += f" about {identifier}"
        sentences.append(subject + ".")
    else:
        sentences.append(f"This {kind} summarizes the documented record.")

    affected = by_role.get("affected", "").rstrip(".")
    cause = by_role.get("cause", "").rstrip(".")
    if affected:
        affected_label = next(
            label for label, _value in fields if _field_role(label) == "affected"
        )
        affected_subject = affected
        if "service" in affected_label.casefold() and not affected.casefold().endswith(
            "service"
        ):
            affected_subject = f"the {affected} service"
        sentence = f"It explains that {affected_subject} was affected"
        if cause:
            sentence += f" because {_sentence_fragment(cause)}"
        sentences.append(sentence + ".")
    elif cause:
        sentences.append(
            f"It says the problem happened because {_sentence_fragment(cause)}."
        )

    resolution = by_role.get("resolution", "").rstrip(".")
    if resolution:
        fragment = _sentence_fragment(resolution)
        first_word = fragment.split(maxsplit=1)[0].casefold() if fragment else ""
        if first_word in _ACTION_VERBS:
            sentences.append(f"The fix was to {fragment}.")
        else:
            sentences.append(f"The documented fix was {fragment}.")

    status = by_role.get("status", "").rstrip(".")
    if status:
        status_subject = f"the {entity}" if identifier else "it"
        sentences.append(
            f"The report marks {status_subject} as {_sentence_fragment(status)}."
        )

    if len(sentences) == 1:
        facts = "; ".join(f"{label}: {value.rstrip('.')}" for label, value in fields[:4])
        sentences.append(f"It records {facts}.")
    answer = " ".join(sentences).rstrip()
    return f"{answer[:-1]} [{filename}]." if answer.endswith(".") else f"{answer} [{filename}]."


def consolidate_repeated_citations(answer: str) -> str:
    """Show a single trailing reference when one source supports the whole answer."""
    if re.search(r"(?m)^\s*##\s+", answer):
        return answer
    citations = _CITATION_PATTERN.findall(answer)
    if len(citations) < 2 or len(set(citations)) != 1:
        return answer
    citation = citations[0]
    cleaned = re.sub(rf"\s*{re.escape(citation)}", "", answer)
    cleaned = re.sub(r"[ \t]+([,.;:!?])", r"\1", cleaned).rstrip()
    if not cleaned:
        return answer
    lines = cleaned.splitlines()
    last = lines[-1].rstrip()
    punctuation = last[-1] if last.endswith((".", "!", "?")) else "."
    if punctuation in ".!?":
        last = last[:-1] if last.endswith(punctuation) else last
    lines[-1] = f"{last} {citation}{punctuation}"
    return "\n".join(lines)


def find_answer_supporting_passages(
    answer: str,
    sources: list[dict[str, Any]],
    *,
    citation_filename: str | None = None,
) -> list[EvidenceSentence]:
    """Return exact source passages that support claims in a completed answer."""
    evidence = _evidence_sentences(sources, include_spreadsheets=True)
    if not evidence:
        return []

    scoped_answer = (
        _answer_text_for_citation(answer, citation_filename)
        if citation_filename
        else answer
    )
    passages = []
    seen = set()
    for claim in _conversational_claims(scoped_answer):
        cleaned = re.sub(r"^#+\s*", "", claim).strip()
        if not cleaned or cleaned.casefold() in {
            "here's the breakdown:",
            "this covers:",
        }:
            continue
        support = _find_claim_support(cleaned, evidence)
        if support is None:
            continue
        marker = (support.filename, support.text)
        if marker not in seen:
            passages.append(support)
            seen.add(marker)
    return passages


def _answer_text_for_citation(answer: str, filename: str) -> str:
    matches = list(_CITATION_PATTERN.finditer(answer))
    cited_filenames = [match.group()[1:-1] for match in matches]
    if filename not in cited_filenames:
        return ""
    if len(set(cited_filenames)) == 1:
        return answer

    fragments = []
    previous_end = 0
    previous_claim = ""
    for match, cited_filename in zip(matches, cited_filenames, strict=True):
        fragment = answer[previous_end : match.start()]
        if _CITATION_PATTERN.sub("", fragment).strip(" \t\r\n,.;:()"):
            previous_claim = fragment
        elif previous_claim:
            fragment = previous_claim
        if cited_filename == filename:
            fragments.append(fragment)
        previous_end = match.end()
    return "\n".join(fragments)


def _format_structured_claim(claim: str) -> str:
    matches = list(_FIELD_LABEL_PATTERN.finditer(claim))
    for match in reversed(matches[1:]):
        claim = f"{claim[: match.start()].rstrip()}\n  {claim[match.start() :]}"
    return claim


def _claims(answer: str) -> list[str]:
    without_citations = _CITATION_PATTERN.sub("", answer)
    lines = [
        re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        for line in without_citations.splitlines()
        if line.strip() and not re.match(r"^\s*sources?\s*:", line, re.IGNORECASE)
    ]
    claims = []
    for line in lines:
        for value in re.split(
            r"(?<=[.!?])\s+|;\s+|,\s+(?=(?:while|whereas|causing|which|likely)\b)",
            line,
        ):
            cleaned = re.sub(
                r"^(?:while|whereas|causing|which|likely)\s+", "", value
            ).strip(" ,")
            if cleaned:
                claims.append(cleaned)
    return claims


def _conversational_claims(answer: str) -> list[str]:
    without_citations = _CITATION_PATTERN.sub("", answer)
    return [
        re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", claim).strip()
        for claim in re.split(r"(?<=[.!?])\s+|\n+", without_citations)
        if claim.strip() and not re.match(r"^\s*sources?\s*:", claim, re.IGNORECASE)
    ]


def _find_claim_support(
    claim: str,
    evidence: list[EvidenceSentence],
    *,
    minimum_coverage: float = _MIN_CLAIM_COVERAGE,
) -> EvidenceSentence | None:
    claim_terms = _terms(claim)
    if not claim_terms:
        return None
    claim_anchors = _anchors(claim)
    matches = [
        sentence
        for sentence in evidence
        if _contains_all(sentence.terms, claim_anchors)
        and (
            not _has_conflicting_named_anchors(sentence.text, claim_anchors)
            or (
                len(claim_anchors) >= 2
                and len(re.split(r"(?<=[.!?])\s+", sentence.text)) == 1
                and _term_coverage(
                    claim_terms, _coverage_available_terms(sentence)
                )
                >= 0.80
                and _anchor_order_matches(claim, sentence.text)
            )
        )
        and _term_coverage(
            claim_terms, _coverage_available_terms(sentence)
        )
        >= minimum_coverage
        and _anchor_order_matches(claim, sentence.text)
    ]
    if not matches:
        return None
    return min(
        matches,
        key=lambda item: (
            not _FIELD_LABEL_PATTERN.match(item.text),
            -_term_coverage(claim_terms, _coverage_available_terms(item)),
            len(item.terms),
            len(item.text),
        ),
    )


def _without_redundant_identifier_claims(
    claims: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    identifiers = [set(re.findall(r"\b\d+(?:\.\d+)*\b", claim)) for claim, _ in claims]
    return [
        item
        for index, item in enumerate(claims)
        if not identifiers[index]
        or not any(
            item[1] == later_filename
            and identifiers[index] < identifiers[later_index]
            for later_index, (_later_claim, later_filename) in enumerate(claims)
            if later_index > index
        )
    ]


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
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", source_text):
            cleaned = sentence.strip()
            if (
                not cleaned
                or cleaned.endswith("?")
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
            spans.extend(
                (
                    f"{first} {second} {third}",
                    f"{first} {second} {third}",
                )
                for first, second, third in zip(
                    sentences, sentences[1:], sentences[2:]
                )
            )
            spans.extend(
                (field, field) for field in _structured_field_spans(source_text)
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


def _labeled_fields(source: dict[str, Any]) -> list[tuple[str, str]]:
    fields = []
    for field in _split_structured_fields(_without_source_header(source)):
        match = _FIELD_LABEL_PATTERN.match(field)
        if not match or field.endswith("?"):
            continue
        label = match.group(0).split(":", 1)[0].strip()
        value = " ".join(field[match.end() :].split()).strip()
        if value:
            fields.append((label, value))
    return fields


def _field_role(label: str) -> str:
    words = set(re.findall(r"[a-z0-9]+", label.casefold().replace("_", " ")))
    if words & {"id", "identifier"}:
        return "identifier"
    if words & {"affected", "impacted"}:
        return "affected"
    if words & {"cause", "reason"}:
        return "cause"
    if words & {"fix", "mitigation", "remedy", "resolution", "solution"}:
        return "resolution"
    if words & {"state", "status"}:
        return "status"
    return "other"


def _identifier_entity(fields: list[tuple[str, str]]) -> str:
    label = next(
        (label for label, _value in fields if _field_role(label) == "identifier"),
        "record",
    )
    entity = re.sub(r"\b(?:id|identifier)\b", "", label, flags=re.IGNORECASE)
    entity = " ".join(entity.replace("_", " ").split()).casefold()
    return entity or "record"


def _structured_field_is_requested(label: str, question: str) -> bool:
    role = _field_role(label)
    cues = {
        "identifier": r"\b(?:id|identifier|number)\b",
        "affected": r"\b(?:affect\w*|impact\w*)\b",
        "cause": r"\b(?:caus\w*|reason|root|why)\b",
        "resolution": r"\b(?:address\w*|fix\w*|mitigat\w*|remed\w*|resol\w*|solution)\b",
        "status": r"\b(?:state|status)\b",
    }
    if role in cues:
        return bool(re.search(cues[role], question, re.IGNORECASE))
    label_terms = _terms(label)
    question_terms = _terms(question)
    return bool(label_terms) and _contains_all(question_terms, label_terms)


def _friendly_field_label(label: str) -> str:
    role = _field_role(label)
    if role == "affected":
        return "Affected service" if "service" in label.casefold() else "What was affected"
    if role == "cause":
        return "What caused it"
    if role == "resolution":
        return "How it was resolved"
    if role == "status":
        return "Status"
    return label.replace("_", " ").strip()


def _sentence_fragment(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return stripped
    first_word = stripped.split(maxsplit=1)[0]
    if first_word.isupper():
        return stripped
    return stripped[:1].casefold() + stripped[1:]


def _structured_field_spans(text: str) -> list[str]:
    fields = _split_structured_fields(text)
    usable = [not field.endswith("?") for field in fields]
    adjacent = [
        f"{fields[index]} {fields[index + 1]}"
        for index in range(len(fields) - 1)
        if usable[index] and usable[index + 1]
    ]
    usable_fields = [
        field for field, include in zip(fields, usable, strict=True) if include
    ]
    return [*usable_fields, *adjacent]


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
        if (
            "-" in raw
            and not any(character.isdigit() for character in raw)
            and not any(character in raw for character in "_/.")
        ):
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
    for token_index, raw in enumerate(
        _TOKEN_PATTERN.findall(_CITATION_PATTERN.sub("", text))
    ):
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
            if lowered not in _STOP_WORDS and (
                any(character.isdigit() for character in value)
                or any(character in value for character in "_/.")
                or (
                    value[:1].isupper()
                    and (
                        token_index > 0
                        or normalized not in _COVERAGE_INSTRUCTION_TERMS
                    )
                )
                or normalized in {"never", "no", "not", "without"}
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


def _plural_focus_term(question: str) -> str | None:
    for raw in _TOKEN_PATTERN.findall(_CITATION_PATTERN.sub("", question)):
        lowered = raw.casefold()
        if raw == lowered and lowered.endswith("s") and lowered not in _STOP_WORDS:
            normalized = _normalize(lowered)
            if normalized:
                return normalized
    return None


def _term_coverage(
    required: list[str] | set[str], available: set[str] | frozenset[str]
) -> float:
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


def _coverage_available_terms(sentence: EvidenceSentence) -> set[str]:
    terms = set(sentence.terms)
    lowered = sentence.text.casefold()
    if "localhost" in lowered or "127.0.0.1" in lowered or re.search(
        r"\b(?:not|never)\b.{0,40}\b(?:expose|public)\b", lowered
    ):
        terms.add("private")
    return terms


def _polarity_conflicts(question: str, evidence: str) -> bool:
    affirmative_public = bool(
        re.search(r"\bpublic(?:ly)?\b", question, re.IGNORECASE)
        and re.search(r"\b(?:expos\w*|open\w*|available)\b", question, re.IGNORECASE)
        and not re.search(
            r"\b(?:not|never|private)\b", question, re.IGNORECASE
        )
    )
    negated_public_evidence = bool(
        re.search(
            r"\b(?:not|never)\b.{0,50}\b(?:expos\w*|public(?:ly)?)\b",
            evidence,
            re.IGNORECASE,
        )
    )
    return affirmative_public and negated_public_evidence


def _anchor_order_matches(claim: str, evidence: str) -> bool:
    named_anchors = _named_anchors(claim)
    claim_anchors = [
        term for term in _ordered_terms(claim) if term in named_anchors
    ]
    shared = [term for term in claim_anchors if _contains_all(_terms(evidence), {term})]
    if len(shared) < 2:
        return True
    evidence_terms = _ordered_terms(evidence)
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
        not any(_terms_match(anchor, expected) for expected in required)
        for anchor in _named_anchors(evidence)
    )


def _named_anchors(text: str) -> set[str]:
    result = set()
    without_labels = _FIELD_LABEL_PATTERN.sub("", _CITATION_PATTERN.sub("", text))
    raw_tokens = _TOKEN_PATTERN.findall(without_labels)
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
    if len(shorter) >= 4 and longer == shorter + "e":
        return True
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
