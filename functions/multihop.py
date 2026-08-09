from __future__ import annotations

import re
from typing import Any


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
_MULTI_HOP_CUES = (
    re.compile(r"\b(?:compare|contrast|connect|relate)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:across|between)\b.+\b(?:documents?|sources?|versions?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:change|difference|progression|relationship)\b.+\b(?:between|from|across)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:first|second|another|other)\s+(?:document|source)\b.+"
        r"\b(?:first|second|another|other)\s+(?:document|source)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bhow do\b.+\band\b.+\b(?:differ|relate|support|work)\b", re.IGNORECASE
    ),
)


def is_multi_hop_question(question: str) -> bool:
    identifiers = {value.casefold() for value in _IDENTIFIER.findall(question)}
    return len(identifiers) >= 2 or any(
        pattern.search(question) for pattern in _MULTI_HOP_CUES
    )


def decomposition_prompt(question: str) -> str:
    return (
        "Break this document question into two to four independent retrieval questions. "
        "Each subquestion must ask for one required fact or calculation. Preserve exact "
        "identifiers, names, versions, paths, numbers, and the scope of each clause. Do not "
        "carry a qualifier such as a platform, version, or location from one clause into a "
        "different clause. Keep directly related public/private or before/after facts together "
        "when one passage is likely to state them together. Do not answer the question or add "
        "background. Return the schema-constrained object only.\n"
        f"Question: {question}"
    )


def sentence_decomposition(question: str) -> list[str] | None:
    """Preserve two or three explicitly separated requirement sentences."""
    sentences = [
        value.strip()
        for value in re.split(r"(?<=[.!?])\s+", question.strip())
        if value.strip()
        and not re.match(
            r"^(?:cite|include|provide|show)\b.*\b(?:citation|citations|source|sources)\b",
            value.strip(),
            re.IGNORECASE,
        )
    ]
    if not 2 <= len(sentences) <= 3:
        return None
    result = []
    for sentence in sentences:
        cleaned = re.sub(r"^then\s+", "", sentence, flags=re.IGNORECASE)
        clauses = re.split(
            r"\s+and\s+(?=(?:what|which|who|where|when|why|how)\b)",
            cleaned,
            flags=re.IGNORECASE,
        )
        result.extend(_as_question(clause) for clause in clauses)
    return result if 2 <= len(result) <= 3 else None


def validate_decomposition(data: Any, original_question: str) -> list[str] | None:
    if not isinstance(data, dict) or not isinstance(data.get("subquestions"), list):
        return None

    subquestions = []
    seen = set()
    for value in data["subquestions"]:
        if not isinstance(value, str):
            return None
        cleaned = " ".join(value.split()).strip()
        key = cleaned.casefold()
        if (
            not 5 <= len(cleaned) <= 300
            or not cleaned.endswith("?")
            or key in seen
            or "{" in cleaned
            or "}" in cleaned
        ):
            return None
        subquestions.append(cleaned)
        seen.add(key)

    if not 2 <= len(subquestions) <= 4:
        return None
    if all(item.casefold() == original_question.casefold() for item in subquestions):
        return None
    return subquestions


def fallback_decomposition(question: str) -> list[str]:
    identifiers = []
    for identifier in _IDENTIFIER.findall(question):
        if identifier.casefold() not in {item.casefold() for item in identifiers}:
            identifiers.append(identifier)
    if len(identifiers) >= 2:
        return [
            f"What does {identifier} state that is relevant to the question?"
            for identifier in identifiers[:4]
        ]

    versions = _version_range(question)
    if versions:
        return [f"What changed in version {version}?" for version in versions]

    comparison = re.search(
        r"\b(?:compare|contrast)\s+(.+?)\s+(?:and|with|to)\s+(.+?)[?.]?$",
        question,
        re.IGNORECASE,
    )
    if comparison:
        return [
            f"What do the documents say about {comparison.group(1).strip()}?",
            f"What do the documents say about {comparison.group(2).strip()}?",
        ]

    parts = re.split(
        r",\s*and\s+(?=what|how|why|which)",
        question,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    if len(parts) == 2:
        return [_as_question(part) for part in parts]

    clean = question.strip().rstrip("?")
    return [
        f"What evidence supports the first required part of this question: {clean}?",
        f"What evidence supports the second required part of this question: {clean}?",
    ]


def _version_range(question: str) -> list[str]:
    match = re.search(
        r"\bversions?\s+(\d+)\.(\d+)\s+(?:to|through)\s+(\d+)\.(\d+)\b",
        question,
        re.IGNORECASE,
    )
    if not match or match.group(1) != match.group(3):
        return []
    start, end = int(match.group(2)), int(match.group(4))
    if end < start or end - start > 3:
        return []
    return [f"{match.group(1)}.{minor}" for minor in range(start, end + 1)]


def _as_question(text: str) -> str:
    cleaned = text.strip().rstrip(".?")
    return cleaned[:1].upper() + cleaned[1:] + "?"
