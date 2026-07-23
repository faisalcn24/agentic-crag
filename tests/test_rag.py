import json
from pathlib import Path

import openpyxl
from docx import Document

from functions import rag
from functions.rag import load_document_file, load_documents


def test_load_docx(tmp_path: Path):
    path = tmp_path / "sample.docx"
    doc = Document()
    doc.add_paragraph("INSIGHT AI project overview")
    doc.save(path)

    docs = load_document_file(path)

    assert docs == [{
        "filename": "sample.docx",
        "text": "Document filename: sample.docx\n\nINSIGHT AI project overview",
        "type": "docx",
    }]


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
    registry_file.write_text(json.dumps({"demo": {"documents": []}, "keep": {"documents": []}}), encoding="utf-8")

    assert rag.remove_index("demo") is True
    assert not index_dir.exists()
    assert json.loads(registry_file.read_text(encoding="utf-8")) == {"keep": {"documents": []}}
