import json
from pathlib import Path
from types import SimpleNamespace

import openpyxl
from docx import Document

from functions import rag
from functions.rag import load_document_file, load_documents


def test_agent_planner_requests_schema_from_both_providers(monkeypatch):
    requests = []

    class FakeCompletions:
        def create(self, **request):
            requests.append(request)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"query":"standalone"}')
                    )
                ],
                usage=None,
            )

    monkeypatch.setattr(
        rag,
        "OpenAI",
        lambda **_kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        ),
    )
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    for provider in ("ollama", "groq"):
        monkeypatch.setenv("INSIGHT_LLM_PROVIDER", provider)
        text, _ = rag.call_model("plan this", node="planner")
        assert text == '{"query":"standalone"}'

    for request in requests:
        response_format = request["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        assert response_format["json_schema"]["schema"] == {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 500,
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        }


def test_every_control_schema_is_requested_from_both_providers(monkeypatch):
    requests = []

    class FakeCompletions:
        def create(self, **request):
            requests.append(request)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
                usage=None,
            )

    monkeypatch.setattr(
        rag,
        "OpenAI",
        lambda **_kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        ),
    )
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    expected = []
    for provider in ("ollama", "groq"):
        monkeypatch.setenv("INSIGHT_LLM_PROVIDER", provider)
        for node, schema in rag.STRUCTURED_OUTPUT_SCHEMAS.items():
            rag.call_model("control", node=node)
            expected.append((node, schema))

    assert len(requests) == len(expected)
    for request, (node, schema) in zip(requests, expected, strict=True):
        value = request["response_format"]["json_schema"]
        assert value["name"] == f"insight_{node}"
        assert value["strict"] is True
        assert value["schema"] == schema


def test_load_docx(tmp_path: Path):
    path = tmp_path / "sample.docx"
    doc = Document()
    doc.add_paragraph("INSIGHT AI project overview")
    doc.save(path)

    docs = load_document_file(path)

    assert docs == [
        {
            "filename": "sample.docx",
            "text": "Document filename: sample.docx\n\nINSIGHT AI project overview",
            "type": "docx",
        }
    ]


def test_load_xlsx(tmp_path: Path):
    path = tmp_path / "sample.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Budget"
    ws.append(["Item", "Amount"])
    ws.append(["Hosting", 25])
    wb.save(path)

    docs = load_document_file(path)

    assert len(docs) == 1
    assert docs[0]["filename"] == "sample.xlsx-Budget"
    assert docs[0]["type"] == "xlsx"
    assert "Item: Hosting | Amount: 25" in docs[0]["text"]


def test_unsupported_file_is_reported(tmp_path: Path):
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

    docs, warnings = load_documents(tmp_path)

    assert docs == []
    assert warnings == ["Skipped unsupported file: notes.txt"]


def test_remove_index_removes_registry_entry_and_index_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("INSIGHT_STORAGE_DIR", str(tmp_path))
    index_dir = rag.get_indexes_dir() / "demo"
    index_dir.mkdir(parents=True)
    registry_file = tmp_path / "registry.json"
    registry_file.write_text(
        json.dumps({"demo": {"documents": []}, "keep": {"documents": []}}),
        encoding="utf-8",
    )

    assert rag.remove_index("demo") is True
    assert not index_dir.exists()
    assert json.loads(registry_file.read_text(encoding="utf-8")) == {
        "keep": {"documents": []}
    }


def test_embedding_setup_reuses_one_model(monkeypatch):
    created = []
    settings = SimpleNamespace(embed_model=None)

    def embedding(model_name):
        model = object()
        created.append((model_name, model))
        return model

    monkeypatch.setattr(rag, "_embedding_model", None)
    monkeypatch.setattr(rag, "Settings", settings)
    monkeypatch.setattr(rag, "HuggingFaceEmbedding", embedding)

    rag.setup_embeddings()
    first = settings.embed_model
    rag.setup_embeddings()

    assert created == [("BAAI/bge-small-en-v1.5", first)]
    assert settings.embed_model is first


def test_llm_setup_reuses_matching_provider_configuration(monkeypatch):
    created = []
    settings = SimpleNamespace(llm=None)

    monkeypatch.setenv("INSIGHT_LLM_PROVIDER", "ollama")
    monkeypatch.setattr(rag, "_llms", {})
    monkeypatch.setattr(rag, "Settings", settings)
    monkeypatch.setattr(
        rag,
        "_build_ollama_llm",
        lambda: created.append(object()) or created[-1],
    )

    rag.setup_llm()
    first = settings.llm
    rag.setup_llm()

    assert created == [first]
    assert settings.llm is first


def test_load_index_reuses_cached_instance(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("INSIGHT_STORAGE_DIR", str(tmp_path))
    index_dir = rag.get_indexes_dir() / "demo"
    index_dir.mkdir(parents=True)
    loaded = []
    expected = object()

    monkeypatch.setattr(rag, "_loaded_indexes", {})
    monkeypatch.setattr(rag, "setup_embeddings", lambda: None)
    monkeypatch.setattr(
        rag.StorageContext,
        "from_defaults",
        lambda persist_dir: SimpleNamespace(persist_dir=persist_dir),
    )
    monkeypatch.setattr(
        rag,
        "load_index_from_storage",
        lambda storage: loaded.append(storage.persist_dir) or expected,
    )

    assert rag.load_index("demo") is expected
    assert rag.load_index("demo") is expected
    assert loaded == [str(index_dir)]


def test_answer_entrypoint_configures_llm_it_uses(monkeypatch):
    configured = []

    class Response:
        source_nodes = []

        def __str__(self):
            return "The answer is not present in the provided documents."

    class Engine:
        @staticmethod
        def query(_message):
            return Response()

    class Index:
        @staticmethod
        def as_query_engine(**_kwargs):
            return Engine()

    monkeypatch.setattr(rag, "setup_llm", lambda: configured.append(True))
    monkeypatch.setattr(rag, "get_reranker", lambda: object())

    result = rag.ask_index_with_sources(Index(), "Unknown fact?")

    assert configured == [True]
    assert result["answer"] == "The answer is not present in the provided documents."


def test_missing_citations_are_appended_from_returned_sources():
    answer = rag.ensure_source_citations(
        "The public port is 80.",
        [
            {"filename": "runbook.docx"},
            {"filename": "runbook.docx"},
            {"filename": "requirements.docx"},
        ],
    )

    assert answer == "The public port is 80.\n\nSource: [runbook.docx]"


def test_abstention_does_not_gain_irrelevant_citations():
    answer = "The answer is not present in the provided documents."

    assert (
        rag.ensure_source_citations(answer, [{"filename": "runbook.docx"}]) == answer
    )


def test_spreadsheet_lookup_matches_requested_row_and_column():
    source = {
        "filename": "budget.xlsx-Quarterly Budget",
        "type": "xlsx",
        "text": (
            "Quarter: Q3 | Category: EC2 hosting | Budget_USD: 30 | Actual_USD: 0\n"
            "Quarter: Q3 | Category: Groq API | Budget_USD: 75 | Actual_USD: 0"
        ),
    }

    assert (
        rag.answer_spreadsheet_lookup(
            "What is the projected Q3 Groq API budget?", [source]
        )
        == "$75 [budget.xlsx-Quarterly Budget]"
    )


def test_spreadsheet_lookup_ignores_notes_when_matching_a_row():
    source = {
        "filename": "budget.xlsx-Quarterly Budget",
        "type": "xlsx",
        "text": (
            "Quarter: Q1 | Category: Groq API | Budget_USD: 40 | Actual_USD: 28 | "
            "Notes: Used for final answers\n"
            "Quarter: Q3 | Category: Groq API | Budget_USD: 75 | Actual_USD: 0 | "
            "Notes: Projected token spend"
        ),
    }

    assert rag.answer_spreadsheet_lookup(
        "What was the Q1 Groq API budget and actual spend?", [source]
    ) == "Budget: $40; Actual spend: $28 [budget.xlsx-Quarterly Budget]"


def test_spreadsheet_lookup_uses_requested_latency_field():
    source = {
        "filename": "benchmarks.xlsx-Indexing Benchmarks",
        "type": "xlsx",
        "text": (
            "Document_Set: demo-corpus | Index_Time_Seconds: 42.8 | Retrieval_Time_ms: 240\n"
            "Document_Set: spreadsheet-heavy | Index_Time_Seconds: 58.1 | Retrieval_Time_ms: 295"
        ),
    }

    assert (
        rag.answer_spreadsheet_lookup(
            "What retrieval latency was recorded for the spreadsheet-heavy benchmark?",
            [source],
        )
        == "295 ms [benchmarks.xlsx-Indexing Benchmarks]"
    )


def test_direct_fact_extraction_returns_exact_identifier_sentence():
    source = {
        "filename": "requirements.docx",
        "type": "docx",
        "text": (
            "FR-005: The system returns source snippets. "
            "FR-006: The system exposes a retrieval-only endpoint."
        ),
    }

    assert rag.answer_direct_fact("What does FR-006 require?", [source]) == (
        "FR-006: The system exposes a retrieval-only endpoint. [requirements.docx]"
    )


def test_direct_fact_extraction_preserves_explicit_feature_negation():
    sources = [
        {
            "filename": "evaluation.docx",
            "type": "docx",
            "text": "Question E: Is authentication included in the current release?",
        },
        {
            "filename": "requirements.docx",
            "type": "docx",
            "text": "The current release does not include authentication.",
        },
    ]

    assert rag.answer_direct_fact("Is authentication included?", sources) == (
        "The current release does not include authentication. [requirements.docx]"
    )


def test_direct_fact_extraction_handles_does_include_question():
    source = {
        "filename": "requirements.docx",
        "type": "docx",
        "text": "The current release does not include authentication.",
    }

    assert rag.answer_direct_fact(
        "Does the current release include authentication?", [source]
    ) == "The current release does not include authentication. [requirements.docx]"


def test_direct_fact_extraction_corrects_unsupported_which_premise():
    source = {
        "filename": "requirements.docx",
        "type": "docx",
        "text": (
            "The current release does not store collections in S3 or a managed "
            "vector database. Local disk is used for simplicity."
        ),
    }

    answer = rag.answer_direct_fact(
        "Which managed vector database is used in production?", [source]
    )

    assert answer == (
        "The current release does not store collections in S3 or a managed vector "
        "database. [requirements.docx]"
    )


def test_direct_fact_extraction_keeps_attached_negative_condition():
    source = {
        "filename": "requirements.docx",
        "type": "docx",
        "text": (
            "The current release does not process scanned PDFs with OCR. "
            "Text must be extractable from the uploaded files."
        ),
    }

    answer = rag.answer_direct_fact(
        "Can the current release process scanned PDFs with OCR?", [source]
    )

    assert answer == (
        "The current release does not process scanned PDFs with OCR. "
        "Text must be extractable from the uploaded files. [requirements.docx]"
    )


def test_direct_fact_extraction_answers_rebuild_timing():
    source = {
        "filename": "releases.docx",
        "type": "docx",
        "text": (
            "Compatibility Notes. Collections built before a chunking or embedding "
            "change should be rebuilt to ensure retrieval uses the current settings."
        ),
    }

    assert rag.answer_direct_fact(
        "When should existing collections be rebuilt?", [source]
    ) == (
        "Collections built before a chunking or embedding change should be rebuilt "
        "to ensure retrieval uses the current settings. [releases.docx]"
    )


def test_direct_fact_extraction_returns_local_windows_storage_path():
    source = {
        "filename": "releases.docx",
        "type": "docx",
        "text": (
            "Version 2.3 added a default local storage recommendation of "
            ".insight_data for Windows development."
        ),
    }

    assert rag.answer_direct_fact(
        "What local Windows storage path was recommended in version 2.3?", [source]
    ) == (
        "Version 2.3 added a default local storage recommendation of .insight_data "
        "for Windows development. [releases.docx]"
    )


def test_direct_fact_extraction_answers_named_services():
    source = {
        "filename": "runbook.docx",
        "type": "docx",
        "text": "The intended service names are insight-api and insight-ui.",
    }

    assert rag.answer_direct_fact(
        "Which systemd services run the application?", [source]
    ) == "The intended service names are insight-api and insight-ui. [runbook.docx]"


def test_direct_fact_extraction_summarizes_deployment_ports():
    source = {
        "filename": "runbook.docx",
        "type": "docx",
        "text": (
            "FastAPI listens on 127.0.0.1:8000 and Streamlit listens on "
            "127.0.0.1:8501. Nginx listens publicly on port 80."
        ),
    }

    answer = rag.answer_direct_fact("Summarize the deployment ports.", [source])

    assert "127.0.0.1:8000" in answer
    assert "127.0.0.1:8501" in answer
    assert "port 80" in answer
    assert "[runbook.docx]" in answer


def test_spreadsheet_lookup_corrects_false_budget_exceed_premise():
    source = {
        "filename": "budget.xlsx-Quarterly Budget",
        "type": "xlsx",
        "text": (
            "Quarter: Q3 | Category: Groq API | Budget_USD: 75 | Actual_USD: 0"
        ),
    }

    answer = rag.answer_spreadsheet_lookup(
        "How much did Q3 Groq spending exceed its $75 budget?", [source]
    )

    assert answer == (
        "It did not exceed the budget; actual spend was $0 against a $75 budget. "
        "[budget.xlsx-Quarterly Budget]"
    )


def test_direct_fact_extraction_corrects_negative_feature_premise():
    source = {
        "filename": "requirements.docx",
        "type": "docx",
        "text": (
            "The current release does not store collections in S3 or a managed vector "
            "database. Local disk is used for simplicity and cost control. "
            "The current release does not process scanned PDFs with OCR. Text must be "
            "extractable from the uploaded files."
        ),
    }

    s3_answer = rag.answer_direct_fact(
        "Which S3 bucket stores the production indexes?", [source]
    )
    ocr_answer = rag.answer_direct_fact(
        "What accuracy did the OCR pipeline achieve?", [source]
    )

    assert "does not store collections in S3" in s3_answer
    assert "Local disk is used" in s3_answer
    assert "does not process scanned PDFs with OCR" in ocr_answer
    assert "Text must be extractable" in ocr_answer


def test_negative_feature_extraction_does_not_hijack_unrelated_question():
    source = {
        "filename": "risks.xlsx-Risks",
        "type": "xlsx",
        "text": "Risk_ID: R-008 | Risk: No authentication | Status: Open",
    }

    assert rag.answer_direct_fact("List the open risks.", [source]) is None


def test_public_deployment_risks_prioritize_auth_and_external_provider():
    source = {
        "filename": "risks.xlsx-Risk Register",
        "type": "xlsx",
        "text": (
            "Risk_ID: R-003 | Risk: Wrong public IP in SSH security group | "
            "Mitigation: Update inbound SSH source\n"
            "Risk_ID: R-007 | Risk: User uploads confidential files to hybrid demo | "
            "Mitigation: Warn that excerpts are sent to external LLM provider\n"
            "Risk_ID: R-008 | Risk: No authentication on public demo | "
            "Mitigation: Add auth before long-lived sharing"
        ),
    }

    answer = rag.answer_public_deployment_risks(
        "What two reasons make a long-lived public deployment unsafe?", [source]
    )

    assert "R-007" in answer
    assert "R-008" in answer
    assert "R-003" not in answer


def test_formatted_spreadsheet_source_preserves_rows_and_full_sheet():
    class FakeNode:
        metadata = {"filename": "sheet.xlsx-Data", "type": "xlsx"}

        @staticmethod
        def get_content(metadata_mode="none"):
            assert metadata_mode == "none"
            return "A: first | B: 1\nA: second | B: 2\n" + ("x" * 3000)

    text = rag.format_source_nodes([FakeNode()])[0]["text"]

    assert "A: first | B: 1\nA: second | B: 2" in text
    assert len(text) > rag.MAX_SOURCE_TEXT_CHARS


def test_agent_retrieval_fetches_wide_candidates_then_reranks(monkeypatch):
    calls = {}

    class Retriever:
        @staticmethod
        def retrieve(query):
            calls["retrieve_query"] = query
            return ["candidate"]

    class Index:
        @staticmethod
        def as_retriever(similarity_top_k):
            calls["similarity_top_k"] = similarity_top_k
            return Retriever()

    class Reranker:
        @staticmethod
        def postprocess_nodes(nodes, query_str):
            calls["rerank"] = (nodes, query_str)
            return []

    def get_reranker(top_n):
        calls["top_n"] = top_n
        return Reranker()

    monkeypatch.setattr(rag, "get_reranker", get_reranker)

    assert rag.retrieve_sources(Index(), "FR-006", top_k=5) == []
    assert calls == {
        "similarity_top_k": rag.RETRIEVE_TOP_K,
        "retrieve_query": "FR-006",
        "top_n": rag.RETRIEVE_TOP_K,
        "rerank": (["candidate"], "FR-006"),
    }


def test_spreadsheet_excerpt_keeps_rows_relevant_to_query():
    text = "\n".join(
        [
            "Document filename: budget.xlsx",
            "Quarter: Q1 | Category: Hosting | Budget_USD: 18",
            "Quarter: Q2 | Category: Hosting | Budget_USD: 22",
            "Quarter: Q3 | Category: Hosting | Budget_USD: 30",
            "Quarter: Q3 | Category: Groq API | Budget_USD: 75",
            "Quarter: Q4 | Category: Database | Budget_USD: 120",
        ]
    )

    excerpt = rag._spreadsheet_excerpt(text, "Q3 Groq API budget", max_rows=2)

    assert "Quarter: Q3 | Category: Groq API | Budget_USD: 75" in excerpt
    assert excerpt.count("Quarter:") == 2


def test_spreadsheet_excerpt_keeps_all_rows_for_analytical_query():
    text = "\n".join(
        [
            "Document filename: benchmarks.xlsx",
            "Document_Set: one | Retrieval_Time_ms: 10",
            "Document_Set: two | Retrieval_Time_ms: 20",
            "Document_Set: three | Retrieval_Time_ms: 30",
        ]
    )

    excerpt = rag._spreadsheet_excerpt(
        text, "Which benchmark has the slowest retrieval time?", max_rows=1
    )

    assert excerpt == text
