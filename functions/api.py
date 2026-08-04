from __future__ import annotations

import os
import shutil
import uuid
from time import perf_counter
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent import run_agent

from .rag import (
    ask_index_with_sources,
    build_index,
    get_uploads_dir,
    load_documents,
    load_index,
    load_registry,
    remove_index,
    retrieve_sources,
    sanitize_index_id,
    update_registry,
)
from .telemetry import log_query_result, summarize_events


CHAT_PAGE = Path(__file__).with_name("static") / "chat.html"
ALLOWED_UPLOAD_TYPES = {
    ".pdf": {"application/pdf"},
    ".docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
    ".xlsx": {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
    ".png": {"image/png"},
    ".jpg": {"image/jpeg"},
    ".jpeg": {"image/jpeg"},
    ".tif": {"image/tiff"},
    ".tiff": {"image/tiff"},
    ".bmp": {"image/bmp", "image/x-ms-bmp"},
    ".webp": {"image/webp"},
}

app = FastAPI(title="Agentic CRAG API")


class HistoryTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    index_id: str
    message: str
    history: list[HistoryTurn] = Field(default_factory=list)


class RetrieveRequest(BaseModel):
    index_id: str
    query: str
    top_k: int = 5


@app.get("/", include_in_schema=False)
def chat_page() -> FileResponse:
    return FileResponse(CHAT_PAGE)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/indexes")
def list_indexes() -> dict:
    return {"indexes": [{"index_id": key, **value} for key, value in load_registry().items()]}


@app.post("/indexes")
def create_index(index_id: str | None = Form(default=None), files: list[UploadFile] = File(...)) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="At least one document is required")

    validate_upload_metadata(files)
    safe_index_id = sanitize_index_id(_default_index_id(index_id, files))

    staging_dir = _new_staging_dir(safe_index_id)
    try:
        _save_uploads(files, staging_dir)
        raw_docs, warnings = load_documents(staging_dir)
        if not raw_docs:
            raise HTTPException(status_code=400, detail={"message": "No readable documents found", "warnings": warnings})
        build_index(raw_docs, safe_index_id)
        upload_dir = _promote_upload_dir(staging_dir, safe_index_id)
        metadata = update_registry(safe_index_id, upload_dir, raw_docs)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
    return {"index_id": safe_index_id, "documents": metadata["documents"], "warnings": warnings}


@app.delete("/indexes/{index_id}")
def delete_index(index_id: str) -> dict:
    safe_index_id = _existing_index_id(index_id)
    if not remove_index(safe_index_id):
        raise HTTPException(status_code=500, detail="Could not remove index")
    upload_dir = get_uploads_dir() / safe_index_id
    if upload_dir.exists():
        shutil.rmtree(upload_dir, ignore_errors=True)
    return {"deleted": safe_index_id}


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message is required")
    safe_index_id = _existing_index_id(request.index_id)
    try:
        started = perf_counter()
        index = load_index(safe_index_id)
        result = ask_index_with_sources(index, request.message, [turn.model_dump() for turn in request.history])
        log_query_result(mode="single", iterations=1, latency_ms=(perf_counter() - started) * 1000)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc
    return result


@app.post("/agent")
def agent(request: ChatRequest) -> dict:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message is required")
    safe_index_id = _existing_index_id(request.index_id)
    try:
        index = load_index(safe_index_id)
        return run_agent(index, request.message, [turn.model_dump() for turn in request.history])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent failed: {exc}") from exc


@app.post("/retrieve")
def retrieve(request: RetrieveRequest) -> dict:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query is required")
    safe_index_id = _existing_index_id(request.index_id)
    try:
        index = load_index(safe_index_id)
        sources = retrieve_sources(index, request.query, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieve failed: {exc}") from exc
    return {"sources": sources}


@app.get("/metrics")
def metrics() -> dict:
    return summarize_events()


def _default_index_id(index_id: str | None, files: list[UploadFile]) -> str:
    if index_id:
        return index_id
    first_filename = files[0].filename or f"upload-{uuid.uuid4().hex[:8]}"
    return Path(first_filename).stem


def _new_staging_dir(index_id: str) -> Path:
    staging_dir = get_uploads_dir() / f".incoming-{index_id}-{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=True, exist_ok=False)
    return staging_dir


def _promote_upload_dir(staging_dir: Path, index_id: str) -> Path:
    upload_dir = get_uploads_dir() / index_id
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    staging_dir.replace(upload_dir)
    return upload_dir


def _save_uploads(files: list[UploadFile], upload_dir: Path) -> None:
    total = 0
    limit = max_upload_bytes()
    for upload in files:
        if not upload.filename:
            continue
        destination = upload_dir / Path(upload.filename).name
        with destination.open("wb") as out_file:
            while chunk := upload.file.read(1024 * 1024):
                total += len(chunk)
                if total > limit:
                    raise HTTPException(status_code=413, detail="Combined uploads exceed the upload size limit")
                out_file.write(chunk)


def _existing_index_id(index_id: str) -> str:
    safe_index_id = sanitize_index_id(index_id)
    if safe_index_id not in load_registry():
        raise HTTPException(status_code=404, detail="Index not found")
    return safe_index_id


def validate_upload_metadata(files: list[UploadFile]) -> None:
    limit = max_upload_bytes()
    declared_total = 0
    for upload in files:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in ALLOWED_UPLOAD_TYPES:
            raise HTTPException(status_code=415, detail=f"Unsupported file extension: {suffix or 'none'}")
        if upload.content_type not in ALLOWED_UPLOAD_TYPES[suffix]:
            raise HTTPException(status_code=415, detail=f"MIME type {upload.content_type or 'missing'} is not allowed for {suffix}")
        if upload.size is not None:
            if upload.size > limit:
                raise HTTPException(status_code=413, detail=f"{upload.filename} exceeds the upload size limit")
            declared_total += upload.size
    if declared_total > limit:
        raise HTTPException(status_code=413, detail="Combined uploads exceed the upload size limit")


def max_upload_bytes() -> int:
    return int(float(os.getenv("AGENTIC_CRAG_MAX_UPLOAD_MB", "10")) * 1024 * 1024)
