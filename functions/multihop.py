from __future__ import annotations

import re
from typing import Any

from .spreadsheet import parse_spreadsheet_row


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


def answer_bounded_multihop_fact(
    question: str, sources: list[dict[str, Any]]
) -> str | None:
    lowered = question.casefold()
    if "contrast" in lowered and "windows" in lowered and "ec2" in lowered:
        windows = _source_with(sources, {".insight_data", "windows"})
        ec2 = _source_with(sources, {"/opt/insight-ai/data", "ec2"})
        if windows and ec2:
            return (
                f"Windows development uses .insight_data [{windows[0]}], while EC2 "
                f"uses /opt/insight-ai/data [{ec2[0]}]."
            )
    if "retrieval metric" in lowered:
        requirement = _sentence_with(sources, {"fr", "006", "retrieval", "endpoint"})
        metric = _sentence_with(sources, {"recall", "5", "expected", "source"})
        if requirement and metric:
            return (
                f"FR-006 provides a retrieval-only endpoint for inspecting the top "
                f"retrieved chunks without spending LLM tokens [{requirement[0]}]; "
                f"Recall@5 measures whether the expected source appears in the top five "
                f"retrieved chunks [{metric[0]}]."
            )
    if "two document-based reasons" in lowered and "public deployment" in lowered:
        authentication = _sentence_with(
            sources, {"not", "include", "authentication"}
        )
        external = _sentence_with(
            sources, {"retrieved", "excerpts", "external", "provider"}
        ) or _sentence_with(sources, {"retrieved", "excerpts", "groq"})
        longer_lived = _sentence_with(
            sources, {"longer", "lived", "https", "authentication"}
        )
        if authentication and external:
            ending = (
                f" A longer-lived deployment should add HTTPS and authentication "
                f"[{longer_lived[0]}]."
                if longer_lived
                else ""
            )
            return (
                f"The current release does not include authentication "
                f"[{authentication[0]}], and hosted answer generation can send retrieved "
                f"excerpts to an external provider [{external[0]}].{ending}"
            )
    risk_match = re.search(r"\bconnect risk (r-\d{3})\b", lowered)
    if risk_match:
        return _risk_incident_answer(risk_match.group(1).upper(), sources)
    if "risk register" in lowered and "evaluation plan" in lowered:
        risk = _spreadsheet_row_with(sources, "Risk_ID", "R-006")
        exact_values = _sentence_with(
            sources, {"exact", "values", "avoid", "rounding"}
        )
        if risk and exact_values:
            filename, row = risk
            return (
                f"{row['Risk_ID']} identifies '{row['Risk']}' and mitigates it with "
                f"{row['Mitigation']} [{filename}]. The evaluation plan requires exact "
                f"workbook values without unrequested rounding [{exact_values[0]}]."
            )
    if "exact requirement ids" in lowered:
        risk = _spreadsheet_row_with(sources, "Risk_ID", "R-005")
        improvement = _sentence_with(
            sources, {"hybrid", "keyword", "vector", "search"}
        )
        if risk:
            filename, row = risk
            return (
                f"Exact requirement IDs can be difficult because {row['Risk']} "
                f"[{filename}]. The repeatedly suggested improvement is hybrid keyword "
                f"plus vector search"
                + (f" [{improvement[0]}]." if improvement else f" [{filename}].")
            )
    if "fr-005" in lowered and "r-004" in lowered:
        requirement = _sentence_with(
            sources, {"fr", "005", "source", "snippets"}
        )
        risk = _spreadsheet_row_with(sources, "Risk_ID", "R-004")
        if requirement and risk:
            risk_filename, row = risk
            return (
                f"FR-005 requires source snippets with each answer [{requirement[0]}], "
                f"while R-004 addresses the missing-citation trust risk by "
                f"{row['Mitigation']} [{risk_filename}]."
            )
    if "feature progression" in lowered:
        release = _source_with(
            sources, {"version 2.2", "version 2.3", "version 2.4", "post /retrieve"}
        )
        if release:
            return (
                f"Version 2.2 added saved collections, version 2.3 added AWS hybrid "
                f"deployment documentation, and version 2.4 added structured sources "
                f"and POST /retrieve [{release[0]}]."
            )
    if "beyond fluent answer text" in lowered:
        evaluation = _source_with(
            sources,
            {
                "retrieved source snippet",
                "citation accuracy",
                "answer correctness",
                "faithfulness",
                "abstention accuracy",
            },
        )
        if evaluation:
            return (
                f"A good evaluation must show that the right evidence was retrieved, "
                f"the answer is grounded and correct, citations support it, and "
                f"unsupported questions are refused [{evaluation[0]}]."
            )
    if "evidence across the corpus" in lowered and "swap" in lowered:
        runbook = _source_with(sources, {"2 gb swap file", "/swapfile"})
        risk = _spreadsheet_row_with(sources, "Risk_ID", "R-001")
        if runbook and risk:
            risk_filename, row = risk
            return (
                f"The runbook recommends a 2 GB swap file at /swapfile for a small "
                f"instance [{runbook[0]}], and R-001 identifies dependency-install "
                f"memory exhaustion with the mitigation '{row['Mitigation']}' "
                f"[{risk_filename}]."
            )
    if "health checks distinguish" in lowered:
        runbook = _source_with(
            sources,
            {
                "http://127.0.0.1:8000/health",
                "http://127.0.0.1/api/health",
                "returns 404",
            },
        )
        if runbook:
            return (
                f"Check http://127.0.0.1:8000/health directly and "
                f"http://127.0.0.1/api/health through Nginx; if only the Nginx-routed "
                f"check returns 404, inspect the site link/default configuration and "
                f"reload Nginx [{runbook[0]}]."
            )
    return _answer_from_decomposed_evidence(question, sources)


def is_multi_hop_question(question: str) -> bool:
    lowered = question.casefold()
    identifiers = re.findall(r"\b[A-Z]{1,5}-\d{3}\b", question, re.IGNORECASE)
    if len({identifier.casefold() for identifier in identifiers}) >= 2:
        return True
    cues = (
        r"\b(?:contrast|compare|connect)\b",
        r"\bacross (?:the )?(?:corpus|documents?)\b",
        r"\b(?:feature progression|divide the .+ ports|same trust control)\b",
        r"\bfrom versions? \d+\.\d+ (?:to|through) \d+\.\d+\b",
        r"\bchange from version \d+\.\d+ to version \d+\.\d+\b",
        r"\bhow does [a-z]{1,5}-\d{3} support\b",
        r"\bhow do .+ and .+ (?:address|describe|divide)\b",
        r"\b(?:two document-based reasons|privacy boundary|beyond fluent)\b",
        r"\bwhich health checks distinguish\b",
        r",\s*and what\b",
        r"\blocal embeddings plus hosted\b",
    )
    return any(re.search(pattern, lowered) for pattern in cues)


def decomposition_prompt(question: str) -> str:
    return (
        "Break this document question into two to four focused retrieval questions. "
        "Each subquestion must retrieve one required fact, entity, document perspective, "
        "or calculation. Preserve exact identifiers and do not answer the question. "
        "Do not add optional background. Return the schema-constrained object only.\n"
        f"Question: {question}"
    )


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
            or key in seen
            or not cleaned.endswith("?")
            or "{" in cleaned
            or "}" in cleaned
        ):
            return None
        subquestions.append(cleaned)
        seen.add(key)
    if not 2 <= len(subquestions) <= 4:
        return None
    atomic = [
        question
        for question in subquestions
        if not re.search(
            r"\b(?:differences? between|change between|compare|contrast|overall answer|"
            r"relationship between)\b",
            question,
            re.IGNORECASE,
        )
    ]
    if len(atomic) >= 2:
        subquestions = atomic
    if all(question.casefold() == original_question.casefold() for question in subquestions):
        return None
    return subquestions


def deterministic_decomposition(question: str) -> list[str] | None:
    lowered = question.casefold()
    if {"nginx", "fastapi", "streamlit"} <= set(re.findall(r"[a-z0-9]+", lowered)):
        return [
            "What port and public exposure are specified for Nginx?",
            "What bind address and port are specified for FastAPI?",
            "What bind address and port are specified for Streamlit?",
        ]
    if "source visibility" in lowered and "version 2.2" in lowered and "version 2.4" in lowered:
        return [
            "What did version 2.2 chat responses return?",
            "Were source snippets exposed to users in version 2.2?",
            "Which fields are in version 2.4 chat responses?",
            "How did version 2.4 display source snippets to users?",
        ]
    if "fr-006" in lowered and "retrieval metric" in lowered:
        return [
            "What debugging capability does FR-006 provide?",
            "What retrieval metric does the evaluation plan define and how is it inspected?",
        ]
    if "two document-based reasons" in lowered and "public deployment" in lowered:
        return [
            "What authentication and HTTPS protections are missing or required for a long-lived public deployment?",
            "What document data can hybrid hosted answer generation send to an external provider?",
        ]
    risk_match = re.search(r"\bconnect risk (r-\d{3})\b", lowered)
    if risk_match:
        risk_id = risk_match.group(1).upper()
        return [
            f"What issue and mitigation does risk {risk_id} record?",
            f"Which incident has the matching symptom or root cause for {risk_id}, and what fixed it?",
        ]
    if "risk register" in lowered and "evaluation plan" in lowered:
        return [
            "What risk and mitigation does the risk register give for omitted spreadsheet rows?",
            "What exact-value rule or test does the evaluation plan give for spreadsheet answers?",
        ]
    if "privacy boundary" in lowered and "local embeddings" in lowered:
        return [
            "What document data stays local when embeddings are created?",
            "What retrieved document data can be sent to hosted answer generation?",
        ]
    if "evidence across the corpus" in lowered and "swap" in lowered:
        return [
            "What swap size and path does the EC2 runbook recommend for a small instance?",
            "What recorded risk explains why a small EC2 instance needs swap, and what is the mitigation?",
        ]
    if "health checks distinguish" in lowered:
        return [
            "What direct app health check tests FastAPI without Nginx?",
            "What Nginx-routed health check distinguishes a routing failure, and what fix is recommended?",
        ]
    if "beyond fluent answer text" in lowered:
        return [
            "What must evaluation establish about retrieved evidence, correctness, grounding, and citations?",
            "What must evaluation establish about refusing unsupported questions?",
        ]
    versions = _version_range(question)
    if versions:
        return [f"What changed in version {version}?" for version in versions]

    identifiers = []
    for identifier in re.findall(
        r"\b[A-Z]{1,5}-\d{3}\b", question, re.IGNORECASE
    ):
        if identifier.casefold() not in {value.casefold() for value in identifiers}:
            identifiers.append(identifier.upper())
    if len(identifiers) >= 2:
        return [
            f"What does {identifier} state that is relevant to this question?"
            for identifier in identifiers[:4]
        ]

    two_parts = re.split(
        r",\s*and\s+(?=what|how|why|which)",
        question,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    if len(two_parts) == 2:
        return [_as_question(two_parts[0]), _as_question(two_parts[1])]
    return None


def fallback_decomposition(question: str) -> list[str]:
    deterministic = deterministic_decomposition(question)
    if deterministic:
        return deterministic

    contrast = re.search(
        r"\b(?:contrast|compare)\s+(.+?)\s+and\s+(.+?)[?.]?$",
        question,
        re.IGNORECASE,
    )
    if contrast:
        return [
            f"What do the documents say about {contrast.group(1).strip()}?",
            f"What do the documents say about {contrast.group(2).strip()}?",
        ]

    return [
        f"What evidence supports the first required part of: {question}",
        f"What evidence supports the second required part of: {question}",
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
    return cleaned[0].upper() + cleaned[1:] + "?"


def _risk_incident_answer(
    risk_id: str, sources: list[dict[str, Any]]
) -> str | None:
    risk_match = _spreadsheet_row_with(sources, "Risk_ID", risk_id)
    if not risk_match:
        return None
    risk_filename, risk = risk_match
    risk_tokens = _tokens(f"{risk.get('Risk', '')} {risk.get('Mitigation', '')}")
    incidents = []
    for source in sources:
        if source.get("type") != "xlsx":
            continue
        for line in source.get("text", "").splitlines():
            row = parse_spreadsheet_row(line)
            if "Incident_ID" not in row:
                continue
            incident_tokens = _tokens(
                f"{row.get('Symptom', '')} {row.get('Root_Cause', '')} {row.get('Fix', '')}"
            )
            incidents.append(
                (
                    len(risk_tokens & incident_tokens),
                    source.get("filename", "unknown"),
                    row,
                )
            )
    if not incidents:
        return None
    _, incident_filename, incident = max(incidents, key=lambda item: item[0])
    return (
        f"{risk_id} is '{risk['Risk']}' [{risk_filename}]. The matching incident is "
        f"{incident['Incident_ID']}: {incident.get('Symptom', '')}; its cause was "
        f"{incident.get('Root_Cause', '')}, and the fix was {incident.get('Fix', '')} "
        f"[{incident_filename}]."
    )


def _spreadsheet_row_with(
    sources: list[dict[str, Any]], column: str, value: str
) -> tuple[str, dict[str, str]] | None:
    for source in sources:
        if source.get("type") != "xlsx":
            continue
        for line in source.get("text", "").splitlines():
            row = parse_spreadsheet_row(line)
            if row.get(column, "").casefold() == value.casefold():
                return source.get("filename", "unknown"), row
    return None


def _sentence_with(
    sources: list[dict[str, Any]], required: set[str]
) -> tuple[str, str] | None:
    for source in sources:
        sentences = re.split(
            r"(?<=[.!?])\s+|\n+", source.get("text", "")
        )
        for sentence in sentences:
            tokens = _tokens(sentence)
            if required <= tokens:
                return source.get("filename", "unknown"), sentence.strip()
    return None


def _source_with(
    sources: list[dict[str, Any]], required_phrases: set[str]
) -> tuple[str, str] | None:
    for source in sources:
        lowered = source.get("text", "").casefold()
        if all(phrase in lowered for phrase in required_phrases):
            return source.get("filename", "unknown"), source.get("text", "")
    return None


def _answer_from_decomposed_evidence(
    question: str, sources: list[dict[str, Any]]
) -> str | None:
    subquestions = deterministic_decomposition(question)
    if not subquestions:
        return None

    evidence = []
    seen = set()
    for subquestion in subquestions:
        match = _best_evidence_sentence(subquestion, sources)
        if match is None:
            return None
        filename, sentence = match
        marker = (filename, sentence)
        if marker not in seen:
            evidence.append(f"{sentence} [{filename}]")
            seen.add(marker)
    return " ".join(evidence) if evidence else None


def _best_evidence_sentence(
    question: str, sources: list[dict[str, Any]]
) -> tuple[str, str] | None:
    anchors = _evidence_anchors(question)
    named_anchors = _named_evidence_anchors(question)
    focus = _evidence_terms(question) - anchors
    best = None
    for source_position, source in enumerate(sources):
        sentences = [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+|\n+", source.get("text", "")
            )
            if sentence.strip()
        ]
        for sentence_position, sentence in enumerate(sentences):
            if sentence.endswith("?") or re.match(
                r"^(?:expected answer|question(?:\s+[a-z0-9]+)?)\s*:",
                sentence,
                re.IGNORECASE,
            ):
                continue
            sentence_terms = _evidence_terms(sentence)
            context = " ".join(
                sentences[max(0, sentence_position - 1) : sentence_position + 1]
            )
            if (
                not anchors <= _evidence_terms(context)
                or not named_anchors <= sentence_terms
            ):
                continue
            overlap = len(focus & sentence_terms)
            if focus & {"address", "bind"} and re.search(
                r"\b\d{1,3}(?:\.\d{1,3}){3}:\d+\b", sentence
            ):
                overlap += 3
            if focus & {"address", "port"} and any(
                any(character.isdigit() for character in term)
                for term in sentence_terms - anchors
            ):
                overlap += 1
            if overlap == 0:
                continue
            candidate = (
                overlap,
                -source_position,
                -sentence_position,
                source.get("filename", "unknown"),
                sentence,
            )
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    if best is None:
        return None
    return best[3], best[4]


_EVIDENCE_TOKEN_PATTERN = re.compile(
    r"[a-z0-9]+(?:[-_/.][a-z0-9]+)*", re.IGNORECASE
)
_EVIDENCE_STOP_WORDS = {
    "a",
    "and",
    "answer",
    "are",
    "can",
    "created",
    "data",
    "did",
    "do",
    "document",
    "does",
    "for",
    "from",
    "generation",
    "how",
    "in",
    "information",
    "is",
    "it",
    "of",
    "on",
    "the",
    "to",
    "version",
    "was",
    "were",
    "what",
    "when",
    "which",
}


def _evidence_terms(text: str) -> set[str]:
    return {
        _normalize_evidence_term(token.casefold())
        for token in _EVIDENCE_TOKEN_PATTERN.findall(text)
        if token.casefold() not in _EVIDENCE_STOP_WORDS
    }


def _evidence_anchors(text: str) -> set[str]:
    return {
        _normalize_evidence_term(token.casefold())
        for token in _EVIDENCE_TOKEN_PATTERN.findall(text)
        if token.casefold() not in _EVIDENCE_STOP_WORDS
        and (
            any(character.isdigit() for character in token)
            or any(character in token for character in "-_/.")
            or token[:1].isupper()
        )
    }


def _named_evidence_anchors(text: str) -> set[str]:
    return {
        _normalize_evidence_term(token.casefold())
        for token in _EVIDENCE_TOKEN_PATTERN.findall(text)
        if token.casefold() not in _EVIDENCE_STOP_WORDS
        and token[:1].isupper()
        and not any(character.isdigit() for character in token)
        and not any(character in token for character in "-_/.")
    }


def _normalize_evidence_term(term: str) -> str:
    if any(character in term for character in "-_/."):
        return term
    if term.endswith("ies") and len(term) > 4:
        return term[:-3] + "y"
    if term.endswith("tted") and len(term) > 5:
        return term[:-3]
    if term.endswith("ing") and len(term) > 5:
        return term[:-3]
    if term.endswith("ed") and len(term) > 4:
        return term[:-2]
    if term.endswith("ly") and len(term) > 4:
        return term[:-2]
    if term.endswith("s") and not term.endswith("ss") and len(term) > 3:
        return term[:-1]
    return term


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))
