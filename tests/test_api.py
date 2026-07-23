from __future__ import annotations

from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from functions.api import validate_upload_metadata


def upload(filename: str, content_type: str, size: int = 3) -> UploadFile:
    return UploadFile(
        file=BytesIO(b"abc"),
        filename=filename,
        size=size,
        headers=Headers({"content-type": content_type}),
    )


def test_upload_allowlist_accepts_pdf_docx_and_xlsx():
    validate_upload_metadata([
        upload("one.pdf", "application/pdf"),
        upload("two.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        upload("three.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    ])


def test_upload_allowlist_rejects_legacy_xls():
    with pytest.raises(HTTPException) as exc_info:
        validate_upload_metadata([upload("legacy.xls", "application/vnd.ms-excel")])

    assert exc_info.value.status_code == 415


def test_upload_limit_rejects_oversized_file(monkeypatch):
    monkeypatch.setenv("INSIGHT_MAX_UPLOAD_MB", "0.000001")

    with pytest.raises(HTTPException) as exc_info:
        validate_upload_metadata([upload("large.pdf", "application/pdf", size=10)])

    assert exc_info.value.status_code == 413
