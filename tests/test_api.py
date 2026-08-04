from __future__ import annotations

from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

from functions.api import app, validate_upload_metadata


client = TestClient(app)


def upload(filename: str, content_type: str, size: int = 3) -> UploadFile:
    return UploadFile(
        file=BytesIO(b"abc"),
        filename=filename,
        size=size,
        headers=Headers({"content-type": content_type}),
    )


def test_root_serves_graphical_chat_page():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AGENTIC CRAG" in response.text
    assert 'fetch(`/${elements.mode.value}`' in response.text
    assert "SOURCE_PREVIEW_LENGTH = 300" in response.text
    assert 'toggle.textContent = "Show more"' in response.text
    assert 'toggle.textContent = expanded ? "Show more" : "Show less"' in response.text
    assert '"closest retrieved passage"' in response.text


def test_upload_allowlist_accepts_documents_and_images():
    validate_upload_metadata([
        upload("one.pdf", "application/pdf"),
        upload("two.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        upload("three.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        upload("four.png", "image/png"),
        upload("five.jpg", "image/jpeg"),
        upload("six.tiff", "image/tiff"),
        upload("seven.webp", "image/webp"),
    ])


def test_upload_allowlist_rejects_legacy_xls():
    with pytest.raises(HTTPException) as exc_info:
        validate_upload_metadata([upload("legacy.xls", "application/vnd.ms-excel")])

    assert exc_info.value.status_code == 415


def test_upload_limit_rejects_oversized_file(monkeypatch):
    monkeypatch.setenv("AGENTIC_CRAG_MAX_UPLOAD_MB", "0.000001")

    with pytest.raises(HTTPException) as exc_info:
        validate_upload_metadata([upload("large.pdf", "application/pdf", size=10)])

    assert exc_info.value.status_code == 413
