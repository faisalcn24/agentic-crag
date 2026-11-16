import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import openpyxl
from docx import Document
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import NodeWithScore, TextNode

from functions import rag
from functions.rag import load_document_file, load_documents


def test_registry_updates_do_not_lose_concurrent_collections(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_CRAG_STORAGE_DIR", str(tmp_path))

    def register(number: int) -> None:
        rag.update_registry(
            f"collection-{number}",
            tmp_path / f"collection-{number}",
            [{"filename": f"document-{number}.pdf", "type": "pdf"}],
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(register, range(16)))

    assert set(rag.load_registry()) == {
        f"collection-{number}" for number in range(16)
    }


def test_agent_planner_requests_schema_from_both_providers(monkeypatch):
    groq_requests = []
    ollama_requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(
                {
                    "message": {
                        "content": '{"query":"standalone","intent":"answer"}'
                    },
                    "prompt_eval_count": 4,
                    "eval_count": 2,
                }
            ).encode()

    def fake_urlopen(request, timeout):
        ollama_requests.append((json.loads(request.data), timeout))
        return FakeResponse()

    class FakeCompletions:
        def create(self, **request):
            groq_requests.append(request)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"query":"standalone","intent":"answer"}'
                        )
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
    monkeypatch.setattr(rag, "urlopen", fake_urlopen)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    for provider in ("ollama", "groq"):
        monkeypatch.setenv("AGENTIC_CRAG_LLM_PROVIDER", provider)
        text, _ = rag.call_model("plan this", node="planner")
        assert text == '{"query":"standalone","intent":"answer"}'

    expected_schema = rag.STRUCTURED_OUTPUT_SCHEMAS["planner"]
    assert ollama_requests[0][0]["format"] == expected_schema
    assert ollama_requests[0][0]["think"] is False
    response_format = groq_requests[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == expected_schema


def test_every_control_schema_is_requested_from_both_providers(monkeypatch):
    groq_requests = []
    ollama_requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"message":{"content":"{}"},"prompt_eval_count":1,"eval_count":1}'

    def fake_urlopen(request, timeout):
        ollama_requests.append(json.loads(request.data))
        return FakeResponse()

    class FakeCompletions:
        def create(self, **request):
            groq_requests.append(request)
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
    monkeypatch.setattr(rag, "urlopen", fake_urlopen)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    expected = []
    for provider in ("ollama", "groq"):
        monkeypatch.setenv("AGENTIC_CRAG_LLM_PROVIDER", provider)
        for node, schema in rag.STRUCTURED_OUTPUT_SCHEMAS.items():
            rag.call_model("control", node=node)
            expected.append((node, schema))

    count = len(rag.STRUCTURED_OUTPUT_SCHEMAS)
    assert len(ollama_requests) == len(groq_requests) == count
    for request, (_node, schema) in zip(ollama_requests, expected[:count], strict=True):
        assert request["format"] == schema
    for request, (node, schema) in zip(groq_requests, expected[count:], strict=True):
        value = request["response_format"]["json_schema"]
        assert value["name"] == f"agentic_crag_{node}"
        assert value["strict"] is True
        assert value["schema"] == schema


def test_load_docx(tmp_path: Path):
    path = tmp_path / "sample.docx"
    doc = Document()
    doc.add_paragraph("Agentic CRAG project overview")
    doc.save(path)

    docs = load_document_file(path)

    assert docs == [
        {
            "filename": "sample.docx",
            "text": "Document filename: sample.docx\n\nAgentic CRAG project overview",
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
    assert docs[0]["rows"] == [
        {"row": 2, "text": "Item: Hosting | Amount: 25"}
    ]


def test_xlsx_builds_one_node_per_row_with_structured_provenance(tmp_path: Path):
    path = tmp_path / "sample.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Budget"
    sheet.append(["Item", "Amount"])
    sheet.append(["Hosting", 25])
    sheet.append(["Logging", 4])
    workbook.save(path)

    nodes = rag._build_nodes(load_document_file(path))

    assert len(nodes) == 2
    assert [node.metadata["row"] for node in nodes] == [2, 3]
    assert all(node.metadata["workbook"] == "sample.xlsx" for node in nodes)
    assert all(node.metadata["sheet"] == "Budget" for node in nodes)
    assert "Item: Hosting | Amount: 25" in nodes[0].get_content()


def test_single_column_spreadsheet_row_can_be_looked_up():
    source = {
        "filename": "status.xlsx-Data",
        "type": "xlsx",
        "text": "Document filename: status.xlsx\nSheet: Data\nStatus: Ready",
    }

    assert (
        rag.extract_spreadsheet_lookup("What is the status?", [source])
        == "Ready [status.xlsx-Data]"
    )


def test_load_image_uses_ocr(tmp_path: Path, monkeypatch):
    path = tmp_path / "label.png"
    path.write_bytes(b"image bytes")
    monkeypatch.setattr(rag, "_ocr_image_text", lambda image: "Asset ID ZX-9001")

    docs = load_document_file(path)

    assert docs == [
        {
            "filename": "label.png",
            "text": "Document filename: label.png\n\nAsset ID ZX-9001",
            "type": "image",
        }
    ]


def test_scanned_pdf_pages_fall_back_to_ocr(tmp_path: Path, monkeypatch):
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"pdf bytes")
    pages = [
        SimpleNamespace(
            extract_text=lambda: "Embedded text is already long enough to keep."
        ),
        SimpleNamespace(extract_text=lambda: ""),
    ]
    monkeypatch.setattr(rag, "PdfReader", lambda _path: SimpleNamespace(pages=pages))
    monkeypatch.setattr(
        rag,
        "_ocr_pdf_pages",
        lambda _path, page_numbers: {page_numbers[0]: "Scanned asset ZX-9001"},
    )

    docs = load_document_file(path)

    assert "Page 1:\nEmbedded text is already long enough to keep." in docs[0]["text"]
    assert "Page 2:\nScanned asset ZX-9001" in docs[0]["text"]


def test_ocr_engine_is_loaded_once(monkeypatch):
    engines = []
    monkeypatch.setattr(rag, "_ocr_engine", None)
    monkeypatch.setattr(
        rag, "_build_ocr_engine", lambda: engines.append(object()) or engines[-1]
    )

    assert rag._get_ocr_engine() is rag._get_ocr_engine()
    assert len(engines) == 1


def test_unsupported_file_is_reported(tmp_path: Path):
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

    docs, warnings = load_documents(tmp_path)

    assert docs == []
    assert warnings == ["Skipped unsupported file: notes.txt"]


def test_remove_index_removes_registry_entry_and_index_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENTIC_CRAG_STORAGE_DIR", str(tmp_path))
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


def test_load_index_reuses_cached_instance(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENTIC_CRAG_STORAGE_DIR", str(tmp_path))
    index_dir = rag.get_indexes_dir() / "demo"
    index_dir.mkdir(parents=True)
    loaded = []
    expected = object()

    monkeypatch.setattr(rag, "_loaded_indexes", {})
    monkeypatch.setattr(rag, "setup_embeddings", lambda: None)
    vector_store = object()
    monkeypatch.setattr(
        rag,
        "_chroma_vector_store",
        lambda _index_id, recreate=False: vector_store,
    )
    monkeypatch.setattr(
        rag.StorageContext,
        "from_defaults",
        lambda persist_dir, vector_store: SimpleNamespace(
            persist_dir=persist_dir, vector_store=vector_store
        ),
    )
    monkeypatch.setattr(
        rag,
        "load_index_from_storage",
        lambda storage: loaded.append(storage.persist_dir) or expected,
    )

    assert rag.load_index("demo") is expected
    assert rag.load_index("demo") is expected
    assert loaded == [str(index_dir)]


def test_chroma_index_persists_multi_file_nodes(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENTIC_CRAG_STORAGE_DIR", str(tmp_path))
    monkeypatch.setattr(rag, "_embedding_model", MockEmbedding(embed_dim=8))
    monkeypatch.setattr(rag, "_loaded_indexes", {})

    rag.build_index(
        [
            {
                "filename": "alpha.docx",
                "type": "docx",
                "text": "Document filename: alpha.docx\n\nProject ORBIT-99 is cobalt.",
            },
            {
                "filename": "beta.pdf",
                "type": "pdf",
                "text": "Document filename: beta.pdf\n\nProject NOVA-42 is amber.",
            },
            {
                "filename": "budget.xlsx-Budget",
                "type": "xlsx",
                "text": (
                    "Document filename: budget.xlsx\nSheet: Budget\n\n"
                    "Project: SOLAR-7 | Budget_USD: 125"
                ),
            },
        ],
        "multi-file",
    )

    collection = rag._chroma_client().get_collection(
        rag._chroma_collection_name("multi-file"), embedding_function=None
    )
    assert collection.count() == 3
    assert (rag.get_indexes_dir() / "multi-file" / "docstore.json").exists()

    rag._loaded_indexes.clear()
    loaded = rag.load_index("multi-file")
    filenames = {
        node.metadata.get("filename") for node in loaded.docstore.docs.values()
    }
    assert filenames == {"alpha.docx", "beta.pdf", "budget.xlsx-Budget"}
    vector_matches = loaded.as_retriever(similarity_top_k=3).retrieve("Project ORBIT-99")
    assert {item.node.metadata["filename"] for item in vector_matches} == filenames
    keyword_matches = rag._bm25_candidates(loaded, "ORBIT-99", top_k=2)
    assert keyword_matches[0].node.metadata["filename"] == "alpha.docx"


def test_remove_index_deletes_its_chroma_collection(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENTIC_CRAG_STORAGE_DIR", str(tmp_path))
    index_dir = rag.get_indexes_dir() / "demo"
    index_dir.mkdir(parents=True)
    rag.get_registry_file().write_text(
        json.dumps({"demo": {"documents": []}}), encoding="utf-8"
    )
    client = rag._chroma_client()
    client.create_collection(
        rag._chroma_collection_name("demo"), embedding_function=None
    )

    assert rag.remove_index("demo") is True
    assert not index_dir.exists()
    assert rag._chroma_collection_name("demo") not in {
        collection.name for collection in client.list_collections()
    }


def test_answer_entrypoint_skips_model_when_retrieval_is_empty(monkeypatch):
    calls = []
    monkeypatch.setattr(
        rag,
        "retrieve_sources",
        lambda _index, query, top_k: calls.append(("retrieve", query, top_k)) or [],
    )
    monkeypatch.setattr(
        rag,
        "call_model",
        lambda _prompt, *, node, timeout: (
            calls.append(("complete", node, timeout))
            or ("The answer is not present in the provided documents.", 1)
        ),
    )

    result = rag.ask_index_with_sources(object(), "Unknown fact?")

    assert calls == [("retrieve", "Unknown fact?", rag.RERANK_TOP_N)]
    assert result["answer"] == "The answer is not present in the provided documents."


def test_direct_rag_uses_shared_conversational_overview(monkeypatch):
    source = {
        "filename": "incident.png",
        "type": "image",
        "score": 0.99,
        "text": (
            "INCIDENT REPORT Incident ID: OCR-417 Affected service: "
            "document-ingestion Root cause: OCR batches exceeded the worker memory "
            "limit. Resolution: Limit each OCR batch to eight pages. Status: Resolved"
        ),
    }
    queries = []
    monkeypatch.setattr(
        rag,
        "retrieve_sources",
        lambda _index, query, top_k: queries.append((query, top_k)) or [source],
    )
    called_nodes = []

    def call_model(_prompt, node, **_kwargs):
        called_nodes.append(node)
        responses = {
            "planner": '{"query":"OCR-417 incident report overview","intent":"overview"}',
            "overview": json.dumps(
                {
                    "answer": (
                        "This incident report covers OCR-417. The document-ingestion "
                        "service was affected because OCR batches exceeded the worker "
                        "memory limit. The issue was resolved by limiting each OCR batch "
                        "to eight pages. The status is resolved."
                    )
                }
            ),
        }
        return responses[node], 10

    monkeypatch.setattr(rag, "call_model", call_model)

    result = rag.ask_index_with_sources(
        object(),
        "What is this about?",
        history=[
            {"role": "user", "content": "Tell me about incident OCR-417."},
            {
                "role": "assistant",
                "content": "OCR-417 affected document-ingestion [incident.png].",
                "source_filenames": ["incident.png"],
            },
        ],
    )

    assert queries == [
        ("OCR-417 incident report overview", rag.RERANK_TOP_N)
    ]
    assert called_nodes == ["planner", "overview"]
    assert result["answer"] == (
        "This incident report covers OCR-417. The document-ingestion service was "
        "affected because OCR batches exceeded the worker memory limit. The issue was "
        "resolved by limiting each OCR batch to eight pages. The status is resolved "
        "[incident.png]."
    )


def test_direct_rag_uses_shared_structured_multi_field_answer(monkeypatch):
    source = {
        "filename": "incident.png",
        "type": "image",
        "score": 0.99,
        "text": (
            "INCIDENT REPORT Incident ID: OCR-417 Affected service: "
            "document-ingestion Root cause: OCR batches exceeded the worker memory "
            "limit. Resolution: Limit each OCR batch to eight pages. Status: Resolved"
        ),
    }
    monkeypatch.setattr(rag, "retrieve_sources", lambda *_args, **_kwargs: [source])
    monkeypatch.setattr(
        rag,
        "call_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("structured fields should not call the model")
        ),
    )

    result = rag.ask_index_with_sources(
        object(),
        (
            "For incident OCR-417, which service was affected, what caused it, "
            "and how was it resolved?"
        ),
    )

    assert result["answer"] == (
        "For incident OCR-417, the report says [incident.png]:\n"
        "- Affected service: document-ingestion\n"
        "- What caused it: OCR batches exceeded the worker memory limit.\n"
        "- How it was resolved: Limit each OCR batch to eight pages."
    )


def test_supporting_sources_follow_citation_order_and_hide_retrieval_scores():
    sources = [
        {
            "filename": "runbook.docx",
            "type": "docx",
            "score": 0.91,
            "text": "Nginx listens publicly on port 80.",
        },
        {
            "filename": "release.docx",
            "type": "docx",
            "score": 0.04,
            "text": "Windows uses .agentic_crag_data and EC2 uses /opt/data.",
        },
        {
            "filename": "noise.docx",
            "type": "docx",
            "score": 0.99,
            "text": "This passage is retrieved but does not support the answer.",
        },
    ]

    grouped = rag.supporting_source_groups(
        (
            "Windows uses .agentic_crag_data and EC2 uses /opt/data "
            "[release.docx]. Nginx listens publicly on port 80 [runbook.docx]."
        ),
        sources,
    )

    assert [source["filename"] for source in grouped] == [
        "release.docx",
        "runbook.docx",
    ]
    assert all("score" not in source for source in grouped)
    assert grouped[0]["passages"] == [
        {"text": "Windows uses .agentic_crag_data and EC2 uses /opt/data."}
    ]


def test_supporting_sources_extract_the_exact_cited_passage():
    grouped = rag.supporting_source_groups(
        "The configured value is 42 [requirements.docx].",
        [
            {
                "filename": "requirements.docx",
                "type": "docx",
                "score": 0.9,
                "text": (
                    "Unrelated introductory material. "
                    "The configured value is 42. "
                    "Unrelated closing material."
                ),
            }
        ],
    )

    assert grouped[0]["passages"] == [
        {"text": "The configured value is 42."}
    ]


def test_supporting_sources_reject_a_cited_document_that_does_not_support_claim():
    grouped = rag.supporting_source_groups(
        "The configured value is 42 [noise.docx].",
        [
            {
                "filename": "noise.docx",
                "type": "docx",
                "score": 0.99,
                "text": "This document only discusses deployment scheduling.",
            }
        ],
    )

    assert grouped == []


def test_supporting_sources_validate_the_claim_attached_to_each_citation():
    grouped = rag.supporting_source_groups(
        (
            "The configured value is 42 [requirements.docx]. "
            "Deployment happens on Tuesday [noise.docx]."
        ),
        [
            {
                "filename": "requirements.docx",
                "type": "docx",
                "text": "The configured value is 42.",
            },
            {
                "filename": "noise.docx",
                "type": "docx",
                "text": "An archive also records the configured value as 42.",
            },
        ],
    )

    assert [source["filename"] for source in grouped] == ["requirements.docx"]


def test_direct_rag_returns_grouped_supporting_documents(monkeypatch):
    incident = {
        "filename": "incident.png",
        "type": "image",
        "score": 0.2,
        "text": (
            "Incident ID: OCR-417 Affected service: document-ingestion "
            "Root cause: memory pressure."
        ),
    }
    noise = {
        "filename": "noise.docx",
        "type": "docx",
        "score": 0.99,
        "text": "Unrelated deployment notes.",
    }
    monkeypatch.setattr(
        rag, "retrieve_sources", lambda *_args, **_kwargs: [noise, incident]
    )
    monkeypatch.setattr(
        rag,
        "call_model",
        lambda *_args, **_kwargs: (
            "OCR-417 affected document-ingestion [incident.png].",
            10,
        ),
    )

    result = rag.ask_index_with_sources(object(), "Which service did OCR-417 affect?")

    assert [source["filename"] for source in result["sources"]] == ["incident.png"]
    assert result["sources"][0]["passages"] == [
        {"text": "Incident ID: OCR-417 Affected service: document-ingestion"}
    ]
    assert "score" not in result["sources"][0]


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
        rag.extract_spreadsheet_lookup(
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

    assert (
        rag.extract_spreadsheet_lookup(
            "What was the Q1 Groq API budget and actual spend?", [source]
        )
        == "Budget: $40; Actual: $28 [budget.xlsx-Quarterly Budget]"
    )


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
        rag.extract_spreadsheet_lookup(
            "What retrieval time was recorded for the spreadsheet-heavy benchmark?",
            [source],
        )
        == "295 ms [benchmarks.xlsx-Indexing Benchmarks]"
    )


def test_spreadsheet_lookup_lists_identifiers_filtered_by_status():
    sources = [
        {
            "filename": "risks.xlsx-Risk Register",
            "type": "xlsx",
            "text": text,
        }
        for text in (
            "Risk_ID: R-005 | Risk: Exact IDs can be missed | Status: Open",
            "Risk_ID: R-006 | Risk: Rows can be omitted | Status: Monitoring",
            "Risk_ID: R-008 | Risk: No authentication | Status: Open",
        )
    ]

    answer = rag.extract_spreadsheet_lookup("List the open risks.", sources)

    assert "R-005 has status Open" in answer
    assert "R-008 has status Open" in answer
    assert "R-006" not in answer


def test_spreadsheet_lookup_does_not_intercept_analytics_or_false_premises():
    source = {
        "filename": "benchmarks.xlsx-Data",
        "type": "xlsx",
        "text": (
            "Name: small | Retrieval_Time_ms: 180 | Budget_USD: 75 | Actual_USD: 0\n"
            "Name: large | Retrieval_Time_ms: 410 | Budget_USD: 90 | Actual_USD: 20"
        ),
    }

    assert rag.extract_spreadsheet_lookup(
        "Which benchmark had the slowest retrieval time?", [source]
    ) is None
    assert rag.extract_spreadsheet_lookup(
        "How much did spending exceed its $75 budget?", [source]
    ) is None
    assert rag.extract_spreadsheet_lookup(
        "How much faster was small retrieval than large retrieval?", [source]
    ) is None


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
    candidate = NodeWithScore(node=TextNode(text="semantic candidate"), score=0.8)

    class Retriever:
        @staticmethod
        def retrieve(query):
            calls["retrieve_query"] = query
            return [candidate]

    class Index:
        docstore = SimpleNamespace(docs={})

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
    assert calls["similarity_top_k"] == rag.RETRIEVE_TOP_K
    assert calls["retrieve_query"] == "FR-006"
    assert calls["top_n"] == rag.RETRIEVE_TOP_K
    fused, rerank_query = calls["rerank"]
    assert rerank_query == "FR-006"
    assert [item.node_id for item in fused] == [candidate.node_id]


def test_hybrid_retrieval_promotes_exact_identifier(monkeypatch):
    semantic = TextNode(
        text="A general deployment configuration overview.",
        metadata={"filename": "overview.docx", "type": "docx"},
    )
    exact = TextNode(
        text="Asset ZX-9001 belongs to the cobalt configuration.",
        metadata={"filename": "assets.docx", "type": "docx"},
    )

    class Retriever:
        @staticmethod
        def retrieve(_query):
            return [NodeWithScore(node=semantic, score=0.99)]

    class Index:
        docstore = SimpleNamespace(
            docs={semantic.node_id: semantic, exact.node_id: exact}
        )

        @staticmethod
        def as_retriever(similarity_top_k):
            assert similarity_top_k == rag.RETRIEVE_TOP_K
            return Retriever()

    class Reranker:
        @staticmethod
        def postprocess_nodes(nodes, query_str):
            assert query_str == "Which configuration belongs to ZX-9001?"
            return nodes

    monkeypatch.setattr(rag, "_keyword_indexes", {})
    monkeypatch.setattr(rag, "get_reranker", lambda top_n: Reranker())

    sources = rag.retrieve_sources(
        Index(), "Which configuration belongs to ZX-9001?", top_k=2
    )

    assert [source["filename"] for source in sources] == [
        "assets.docx",
        "overview.docx",
    ]


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


def test_spreadsheet_lookup_parses_each_source_once(monkeypatch):
    from functions import grounding

    sources = [
        {"filename": "budget.xlsx", "type": "xlsx", "text": "Quarter: Q1 | Budget_USD: 40"},
        {"filename": "budget.xlsx", "type": "xlsx", "text": "Quarter: Q3 | Budget_USD: 75"},
    ]
    original = grounding._spreadsheet_rows
    calls = []
    def rows(source):
        calls.append(source)
        return original(source)
    monkeypatch.setattr(grounding, "_spreadsheet_rows", rows)
    assert grounding.extract_spreadsheet_lookup("What is the Q3 budget?", sources) == "$75 [budget.xlsx]"
    assert calls == sources


def test_keyword_corpus_extracts_each_nodes_text_once(monkeypatch):
    calls = []
    class Node:
        def __init__(self, text):
            self.text = text
        def get_content(self, metadata_mode):
            calls.append(self)
            return self.text
    class Index:
        docstore = SimpleNamespace(docs={str(i): Node(text) for i, text in enumerate(["Alpha alpha", " ", "Beta"])})
    monkeypatch.setattr(rag, "_keyword_indexes", {})
    index = Index()
    corpus = rag._get_keyword_corpus(index)
    assert calls == list(index.docstore.docs.values())
    assert [dict(counts) for counts in corpus.token_counts] == [{"alpha": 2}, {"beta": 1}]
    assert corpus.average_length == 1.5
    assert rag._get_keyword_corpus(index) is corpus
    assert len(calls) == 3


def test_evidence_tokenization_reuses_overlapping_spans(monkeypatch):
    from functions import grounding

    text = "Alpha stores reports. Beta reads reports. Gamma exports reports."
    original = grounding._terms
    calls = []
    def terms(value):
        calls.append(value)
        return original(value)
    monkeypatch.setattr(grounding, "_terms", terms)
    sources = [{"filename": name, "text": text, "type": "pdf"} for name in ("a.pdf", "b.pdf")]
    evidence = grounding._evidence_sentences(sources, include_spreadsheets=True)
    assert len(evidence) == 12
    assert {item.filename for item in evidence} == {"a.pdf", "b.pdf"}
    assert len(calls) == len(set(calls)) == 6
    assert evidence[0].context_terms == evidence[0].terms
    assert evidence[1].context_terms == evidence[0].terms | evidence[1].terms


def test_sheet_rows_streams_input_and_preserves_row_numbers():
    class Rows:
        def __init__(self):
            self.rows = iter([("ID", None, "Value"), ("A", "ignored", 0), (None, None, None), ("B", None, False)])
        def __iter__(self):
            return self
        def __next__(self):
            return next(self.rows)
        def __length_hint__(self):
            raise AssertionError("Do not materialize all worksheet rows")
    sheet = SimpleNamespace(iter_rows=lambda **_: Rows())
    assert rag._sheet_rows(sheet) == [
        {"row": 2, "text": "ID: A | Value: 0"},
        {"row": 4, "text": "ID: B | Value: False"},
    ]


def test_shared_tokenizer_preserves_identifier_and_word_matching():
    from functions import grounding

    text = "use alpha_beta and solar-panel at 1.2"
    assert grounding._ordered_terms(text) == [
        "use", "alpha_beta", "alpha", "beta", "solar", "panel", "1.2",
    ]
    assert grounding._anchors(text) == {"alpha_beta", "1.2"}


def test_structured_field_aliases_keep_requested_values_and_headings():
    from functions import grounding

    source = {
        "filename": "incident.pdf", "type": "pdf",
        "text": "Incident ID: ZX-123 Root Cause: stale cache. Mitigation: restart service. State: closed.",
    }
    result = grounding.extract_structured_answer(
        "For ZX-123, why did it fail and how was it mitigated?", [source],
    )
    assert result == (
        "For incident ZX-123, the report says [incident.pdf]:\n"
        "- What caused it: stale cache.\n"
        "- How it was resolved: restart service."
    )
