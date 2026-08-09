from __future__ import annotations

import os
from pathlib import Path

import pytest
from llama_index.core.schema import NodeWithScore, TextNode
from PIL import Image, ImageDraw, ImageFont

from functions import rag
from functions.agent import run_agent
from functions.grounding import extract_grounded_sentence, find_grounded_evidence


RUN_LIVE_QUALITY_TESTS = os.getenv("AGENTIC_CRAG_RUN_LIVE_QUALITY_TESTS") == "1"
ABSTENTION = "The answer is not present in the provided documents."


def test_plural_documents_question_is_an_overview_request():
    assert rag.classify_answer_intent("What are the documents about?") == "overview"


def test_conversational_grounding_keeps_plain_private_port_summary():
    sources = [
        {
            "filename": "runbook.docx",
            "type": "docx",
            "text": (
                "The canonical public web port is 80, not 8501. The canonical backend "
                "port is 8000, but it should only bind to localhost. The canonical UI "
                "port is 8501, but it should only bind to localhost. FastAPI listens on "
                "127.0.0.1:8000 and Streamlit listens on 127.0.0.1:8501."
            ),
        }
    ]

    answer = rag.ground_conversational_answer(
        "Which port is public and which application ports remain private?",
        (
            "The canonical public web port is 80, not 8501. The canonical backend port "
            "is 8000, but it should only bind to localhost. The canonical UI port is "
            "8501, but it should only bind to localhost. FastAPI uses port 8000 and "
            "Streamlit uses port 8501, so both remain private on localhost."
        ),
        sources,
    )

    assert "canonical" not in answer.casefold()
    assert "backend port is 8000" not in answer
    assert "UI port is 8501" not in answer
    assert "FastAPI uses port 8000" in answer
    assert "Streamlit uses port 8501" in answer
    assert answer.count("[runbook.docx]") == 1


def test_real_ocr_reads_image_and_scanned_pdf(tmp_path: Path):
    image_path = tmp_path / "asset.png"
    image = Image.new("RGB", (1200, 240), "white")
    ImageDraw.Draw(image).text(
        (40, 70),
        "ASSET ZX-9001 COBALT",
        fill="black",
        font=ImageFont.load_default(size=64),
    )
    image.save(image_path)

    image_docs = rag.load_document_file(image_path)
    assert "ZX-9001" in image_docs[0]["text"]

    pdf_path = tmp_path / "scan.pdf"
    image.save(pdf_path, "PDF", resolution=150)
    pdf_docs = rag.load_document_file(pdf_path)
    assert "ZX-9001" in pdf_docs[0]["text"]


def test_pdf_extraction_reads_embedded_text(tmp_path: Path):
    path = tmp_path / "grounded-fact.pdf"
    _write_text_pdf(path, "Project Aster uses launch code ZEPHYR-17.")

    documents = rag.load_document_file(path)

    assert documents == [
        {
            "filename": "grounded-fact.pdf",
            "text": (
                "Document filename: grounded-fact.pdf\n\n"
                "Project Aster uses launch code ZEPHYR-17."
            ),
            "type": "pdf",
        }
    ]


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        (
            "Who designed Project Aster's launch badge?",
            (
                "The premise is incorrect; there is no information about Project "
                "Aster in the provided evidence.\n\nSource: [aster.docx]"
            ),
        ),
        (
            "What material is Project Aster's launch badge made from?",
            (
                "There is no information about the badge material in the provided "
                "evidence."
            ),
        ),
    ],
)
def test_evidence_insufficiency_variants_become_exact_abstentions(
    question: str, answer: str
):
    sources = [
        {
            "filename": "aster.docx",
            "type": "docx",
            "text": "Project Aster's launch badge is cobalt blue.",
        }
    ]

    assert rag.ground_generated_answer(question, answer, sources) == ABSTENTION


def test_supported_negative_fact_is_not_mistaken_for_an_abstention():
    answer = "The current release does not include authentication."
    sources = [
        {
            "filename": "requirements.docx",
            "type": "docx",
            "text": answer,
        }
    ]

    assert rag.ground_generated_answer(
        "Does the current release include authentication?", answer, sources
    ) == ("The current release does not include authentication. [requirements.docx]")


def test_grounder_rejects_cross_entity_fact_transfer():
    sources = [
        {
            "filename": "aster.docx",
            "type": "docx",
            "text": "Project Aster's launch badge is cobalt blue.",
        },
        {
            "filename": "orion.docx",
            "type": "docx",
            "text": "Leda Noor designed Project Orion's launch badge.",
        },
    ]

    assert (
        rag.ground_generated_answer(
            "Who designed Project Aster's launch badge?",
            "Leda Noor designed Project Aster's launch badge. [orion.docx]",
            sources,
        )
        == ABSTENTION
    )


def test_grounder_rebuilds_citation_from_the_supporting_source():
    sources = [
        {
            "filename": "orion.docx",
            "type": "docx",
            "text": "Project Orion's launch badge is red.",
        },
        {
            "filename": "aster.docx",
            "type": "docx",
            "text": (
                "Document filename: aster.docx The color of Project Aster's "
                "launch badge is cobalt blue."
            ),
        },
    ]

    answer = rag.ground_generated_answer(
        "What color is Project Aster's launch badge?",
        "The color of Project Aster's launch badge is cobalt blue. [orion.docx]",
        sources,
    )

    assert answer == (
        "The color of Project Aster's launch badge is cobalt blue. [aster.docx]"
    )


def test_extractive_grounding_requires_all_substantive_question_terms():
    sources = [
        {
            "filename": "aster.docx",
            "type": "docx",
            "text": (
                "Document filename: aster.docx The color of Project Aster's "
                "launch badge is cobalt blue."
            ),
        },
        {
            "filename": "orion.docx",
            "type": "docx",
            "text": "Leda Noor designed Project Orion's launch badge.",
        },
    ]

    assert (
        rag.extract_grounded_sentence(
            "What color is Project Aster's launch badge?", sources
        )
        == "The color of Project Aster's launch badge is cobalt blue. [aster.docx]"
    )
    assert (
        rag.extract_grounded_sentence(
            "Who designed Project Aster's launch badge?", sources
        )
        is None
    )


def test_grounder_uses_source_context_but_returns_only_the_supporting_sentence():
    sources = [
        {
            "filename": "runbook.docx",
            "type": "docx",
            "text": (
                "EC2 deployment configuration. FastAPI listens on 127.0.0.1:8000."
            ),
        }
    ]

    answer = rag.ground_generated_answer(
        "Where should FastAPI listen in the EC2 deployment?",
        (
            "FastAPI listens on 127.0.0.1:8000. "
            "This is the recommended production convention."
        ),
        sources,
    )

    assert answer == "FastAPI listens on 127.0.0.1:8000. [runbook.docx]"


def test_grounder_keeps_flattened_structured_fields_concise():
    sources = [
        {
            "filename": "incident.png",
            "type": "image",
            "text": (
                "Document filename: incident.png INCIDENT REPORT "
                "Incident ID: OCR-417 Affected service: document-ingestion "
                "Root cause: OCR batches exceeded the worker memory limit. "
                "Resolution: Limit each OCR batch to eight pages. Status: Resolved"
            ),
        }
    ]

    answer = rag.ground_generated_answer(
        (
            "For incident OCR-417, which service was affected, what caused it, "
            "and how was it resolved?"
        ),
        (
            "The document-ingestion service was affected in incident OCR-417. "
            "The root cause of the incident was that OCR batches exceeded the worker "
            "memory limit. The resolution to the incident was to limit each OCR batch "
            "to eight pages."
        ),
        sources,
    )

    assert answer == (
        "- Incident ID: OCR-417\n"
        "  Affected service: document-ingestion [incident.png]\n"
        "- Root cause: OCR batches exceeded the worker memory limit. [incident.png]\n"
        "- Resolution: Limit each OCR batch to eight pages. [incident.png]"
    )


def test_adjacent_evidence_sentences_preserve_section_context():
    sources = [
        {
            "filename": "releases.docx",
            "type": "docx",
            "text": (
                "Version 2.4 added structured source reporting to chat responses. "
                "The chat response now includes an answer field and a sources list."
            ),
        }
    ]

    answer = rag.extract_grounded_sentence(
        "What fields are in a version 2.4 chat response?", sources
    )

    assert answer == (
        "Version 2.4 added structured source reporting to chat responses. "
        "The chat response now includes an answer field and a sources list. "
        "[releases.docx]"
    )


def test_natural_hyphenated_terms_are_grounded_compositionally():
    source_text = (
        "The product success target for the demo is to answer at least 85 percent "
        "of fact-seeking test questions correctly when the answer exists."
    )
    sources = [{"filename": "requirements.docx", "type": "docx", "text": source_text}]

    answer = rag.ground_generated_answer(
        "What answer-correctness target is defined for fact-seeking demo questions?",
        "The answer-correctness target is 85 percent for fact-seeking demo questions.",
        sources,
    )

    assert answer == f"{source_text} [requirements.docx]"


def test_extractive_grounding_handles_a_lexical_paraphrase_without_nli():
    question = "What command validates the Nginx configuration?"
    sources = [
        {
            "filename": "runbook.docx",
            "type": "docx",
            "text": "Validate Nginx config with sudo nginx -t.",
        }
    ]

    assert extract_grounded_sentence(question, sources) == (
        "Validate Nginx config with sudo nginx -t. [runbook.docx]"
    )


def test_complex_port_coverage_prefers_the_exact_public_and_private_sentences():
    sources = [
        {
            "filename": "runbook.docx",
            "type": "docx",
            "text": (
                "Security Group Rules Inbound SSH on port 22 should be restricted. "
                "Inbound HTTP on port 80 may be opened temporarily. "
                "The EC2 instance should not expose FastAPI port 8000 or Streamlit "
                "port 8501 directly to the public internet. Nginx listens publicly "
                "on port 80."
            ),
        }
    ]

    public = find_grounded_evidence(
        "Which port should be publicly exposed?", sources, allow_complex=True
    )
    private = find_grounded_evidence(
        "Which application ports must remain private?", sources, allow_complex=True
    )

    assert public is not None
    assert public.text == "Nginx listens publicly on port 80."
    assert private is not None
    assert private.text == (
        "The EC2 instance should not expose FastAPI port 8000 or Streamlit port "
        "8501 directly to the public internet."
    )


@pytest.mark.parametrize(
    ("question", "source_text", "expected"),
    [
        (
            "What does Recall@5 measure in the evaluation plan?",
            "Recall@5 measures whether the expected source appears in the top five retrieved chunks.",
            "Recall@5 measures whether the expected source appears in the top five retrieved chunks.",
        ),
        (
            "What source-transparency pass criterion is specified?",
            "For source transparency, at least 80 percent of correct answers should show a supporting source snippet.",
            "For source transparency, at least 80 percent of correct answers should show a supporting source snippet.",
        ),
        (
            "Why are public demo documents recommended in hybrid mode?",
            "Public demo documents should be used because retrieved excerpts are sent to Groq in hybrid mode.",
            "Public demo documents should be used because retrieved excerpts are sent to Groq in hybrid mode.",
        ),
    ],
)
def test_extractive_grounding_handles_metric_and_rationale_questions(
    question: str, source_text: str, expected: str
):
    sources = [{"filename": "evidence.docx", "type": "docx", "text": source_text}]

    assert extract_grounded_sentence(question, sources) == (
        f"{expected} [evidence.docx]"
    )


def test_extractive_grounding_uses_the_relation_after_a_version_identifier():
    sources = [
        {
            "filename": "releases.docx",
            "type": "docx",
            "text": (
                "Version 2.2 introduced a registry file. "
                "The registry stores folder_path, folder_name, and document types."
            ),
        }
    ]

    assert extract_grounded_sentence(
        "Which fields did version 2.2 store in the registry?", sources
    ) == (
        "Version 2.2 introduced a registry file. The registry stores "
        "folder_path, folder_name, and document types. [releases.docx]"
    )


def test_generated_words_align_with_underscored_evidence_identifiers():
    sources = [
        {
            "filename": "releases.docx",
            "type": "docx",
            "text": "The registry stores folder_path and folder_name.",
        }
    ]

    assert (
        rag.ground_generated_answer(
            "Which fields are stored in the registry?",
            "The registry stores folder path and folder name.",
            sources,
        )
        == "The registry stores folder_path and folder_name. [releases.docx]"
    )


@pytest.mark.parametrize(
    "question",
    [
        "Who designed Project Aster's launch badge?",
        "What material is Project Aster's launch badge made from?",
    ],
)
def test_relevance_without_the_requested_relation_is_not_support(question: str):
    sources = [
        {
            "filename": "aster.docx",
            "type": "docx",
            "text": (
                "The color of Project Aster's launch badge is cobalt blue. "
                "Its launch site is Harbor Nine."
            ),
        }
    ]

    assert extract_grounded_sentence(question, sources) is None


def test_grounder_does_not_transfer_a_fact_between_entities_in_one_chunk():
    sources = [
        {
            "filename": "projects.docx",
            "type": "docx",
            "text": (
                "Project Aster's launch badge is cobalt blue. "
                "Project Orion's launch badge is made from titanium."
            ),
        }
    ]

    assert (
        rag.ground_generated_answer(
            "What material is Project Aster's launch badge made from?",
            "Project Aster's launch badge is made from titanium.",
            sources,
        )
        == ABSTENTION
    )


def test_grounder_rejects_reversed_relation_arguments():
    sources = [
        {
            "filename": "ranking.docx",
            "type": "docx",
            "text": "Project Aster ranks above Project Orion.",
        }
    ]

    assert (
        rag.ground_generated_answer(
            "How do Project Aster and Project Orion rank?",
            "Project Orion ranks above Project Aster.",
            sources,
        )
        == ABSTENTION
    )


def test_question_and_expected_answer_text_cannot_become_evidence():
    sources = [
        {
            "filename": "evaluation.docx",
            "type": "docx",
            "text": (
                "Question F: What was the company payroll for 2026?\n"
                "Expected answer: The answer is not present."
            ),
        }
    ]

    question = "What was the company payroll for 2026?"
    assert rag.extract_grounded_sentence(question, sources) is None
    assert rag.ground_generated_answer(question, question, sources) == ABSTENTION


@pytest.mark.skipif(
    not RUN_LIVE_QUALITY_TESTS,
    reason="set AGENTIC_CRAG_RUN_LIVE_QUALITY_TESTS=1 to load the real reranker",
)
def test_real_reranker_promotes_semantically_matching_passage():
    candidates = [
        NodeWithScore(
            node=TextNode(text="Bananas are yellow tropical fruit."), score=0.99
        ),
        NodeWithScore(
            node=TextNode(text="The EC2 storage path is /opt/agentic-crag/data."),
            score=0.01,
        ),
        NodeWithScore(node=TextNode(text="Penguins live in cold regions."), score=0.80),
    ]

    ranked = rag.get_reranker(top_n=3).postprocess_nodes(
        candidates, query_str="What storage path should EC2 use?"
    )

    assert ranked[0].node.text == ("The EC2 storage path is /opt/agentic-crag/data.")
    assert ranked[0].score > max(item.score for item in ranked[1:])


@pytest.mark.skipif(
    not RUN_LIVE_QUALITY_TESTS,
    reason="set AGENTIC_CRAG_RUN_LIVE_QUALITY_TESTS=1 and run Ollama",
)
def test_live_rag_stays_grounded_with_entity_distractors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("AGENTIC_CRAG_STORAGE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENTIC_CRAG_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:3b")
    monkeypatch.setenv("OLLAMA_PLANNER_MODEL", "llama3.2:3b")

    index = rag.build_index(
        [
            {
                "filename": "aster.docx",
                "type": "docx",
                "text": (
                    "Document filename: aster.docx\n\n"
                    "The color of Project Aster's launch badge is cobalt blue. "
                    "Its launch site is Harbor Nine."
                ),
            },
            {
                "filename": "orion.docx",
                "type": "docx",
                "text": (
                    "Document filename: orion.docx\n\n"
                    "Project Orion's launch badge is red and made from titanium. "
                    "Leda Noor designed Project Orion's badge in 2035."
                ),
            },
        ],
        "synthetic-hallucination-check",
    )

    questions = {
        "supported": "What color is Project Aster's launch badge?",
        "designer": "Who designed Project Aster's launch badge?",
        "material": "What material is Project Aster's launch badge made from?",
    }
    answers = {
        (mode, name): _ask(index, mode, question)
        for mode in ("single", "agent")
        for name, question in questions.items()
    }

    failures = []
    for mode in ("single", "agent"):
        supported = answers[(mode, "supported")]
        if "cobalt blue" not in supported.casefold():
            failures.append(f"{mode} supported answer missed cobalt blue: {supported}")
        if "[aster.docx]" not in supported:
            failures.append(f"{mode} supported answer missed its citation: {supported}")
        for distractor in ("titanium", "Leda Noor", "2035"):
            if distractor.casefold() in supported.casefold():
                failures.append(
                    f"{mode} supported answer copied {distractor}: {supported}"
                )

        for name in ("designer", "material"):
            answer = answers[(mode, name)]
            if answer != ABSTENTION:
                failures.append(f"{mode} {name} did not abstain exactly: {answer}")

    assert not failures, "\n".join(failures)


def _ask(index, mode: str, question: str) -> str:
    if mode == "single":
        return rag.ask_index_with_sources(index, question)["answer"]
    return run_agent(index, question)["answer"]


def _write_text_pdf(path: Path, text: str) -> None:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    ]

    document = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, value in enumerate(objects, start=1):
        offsets.append(len(document))
        document.extend(f"{number} 0 obj\n".encode("ascii"))
        document.extend(value)
        document.extend(b"\nendobj\n")

    xref_offset = len(document)
    document.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    document.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    document.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(document)
