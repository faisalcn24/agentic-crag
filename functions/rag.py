from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Generator
from pathlib import Path
from typing import Any

import openpyxl
from docx import Document
from llama_index.core import Document as LlamaDocument
from llama_index.core import (
    PromptTemplate,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.llms import ChatMessage, CompletionResponse, CustomLLM, LLMMetadata, MessageRole
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from openai import OpenAI
from pypdf import PdfReader


DEFAULT_STORAGE_DIR = Path.home() / "INSIGHT_AI_storage"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx"}

TEXT_CHUNK_SIZE = 500
TEXT_CHUNK_OVERLAP = 50
SPREADSHEET_CHUNK_SIZE = 4000
SPREADSHEET_CHUNK_OVERLAP = 0

# Advanced retrieval: fetch a wide candidate set from the vector store, then use a
# cross-encoder re-ranker to keep only the most relevant chunks for the LLM.
RERANKER_MODEL = "BAAI/bge-reranker-base"
RETRIEVE_TOP_K = 20          # candidates pulled from the vector store before re-ranking
RERANK_TOP_N = 3             # chunks kept after re-ranking and sent to the LLM
MAX_SOURCE_TEXT_CHARS = 700

SYSTEM_PROMPT = (
    "You are a document analysis assistant that answers questions based ONLY on the provided documents. "
    "Be concise and cite the source filename for every factual claim. "
    "Never make up or infer information not present in the documents. "
    "When comparing documents, identify differences and cite which version contains each detail. "
    "For spreadsheets, report values exactly as they appear and do not omit relevant rows. "
    "For calculations, show the raw figures and working. "
    "If the retrieved context does not contain the answer, say that the answer is not present in the provided documents."
)

# Answer-synthesis prompt for the query engine. Carries the grounding/citation rules
# above so they still apply when running through the re-ranking query engine.
QA_PROMPT = PromptTemplate(
    SYSTEM_PROMPT
    + "\n\n"
    "Context information from the documents is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Using only the context above and not prior knowledge, answer the question.\n"
    "Question: {query_str}\n"
    "Answer: "
)


class GroqCompatibleLLM(CustomLLM):
    model: str
    api_key: str
    api_base: str = "https://api.groq.com/openai/v1"
    timeout: float = 120.0
    temperature: float = 0.0
    max_tokens: int | None = None

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=131072,
            num_output=self.max_tokens or 1024,
            is_chat_model=True,
            model_name=self.model,
        )

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        client = OpenAI(api_key=self.api_key, base_url=self.api_base, timeout=self.timeout)
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.temperature),
        }
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens is not None:
            request["max_tokens"] = max_tokens

        response = client.chat.completions.create(**request)
        text = response.choices[0].message.content or ""
        return CompletionResponse(text=text, raw=response)

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> Generator[CompletionResponse, None, None]:
        yield self.complete(prompt, formatted=formatted, **kwargs)


def load_dotenv_file(path: str | Path | None = None) -> None:
    env_path = Path(path) if path else Path.cwd() / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


load_dotenv_file()


def get_storage_dir() -> Path:
    return Path(os.getenv("INSIGHT_STORAGE_DIR", str(DEFAULT_STORAGE_DIR))).expanduser().resolve()


def get_uploads_dir() -> Path:
    return get_storage_dir() / "uploads"


def get_indexes_dir() -> Path:
    return get_storage_dir() / "indexes"


def get_registry_file() -> Path:
    return get_storage_dir() / "registry.json"


def ensure_storage_dirs() -> None:
    get_uploads_dir().mkdir(parents=True, exist_ok=True)
    get_indexes_dir().mkdir(parents=True, exist_ok=True)


def sanitize_index_id(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip()).strip("-")
    if not cleaned:
        raise ValueError("Index name must include at least one letter or number")
    return cleaned


def load_registry() -> dict:
    ensure_storage_dirs()
    registry_file = get_registry_file()
    if not registry_file.exists():
        return {}
    return json.loads(registry_file.read_text(encoding="utf-8"))


def save_registry(registry: dict) -> None:
    ensure_storage_dirs()
    get_registry_file().write_text(json.dumps(registry, indent=2), encoding="utf-8")


def load_documents(documents_dir: str | Path) -> tuple[list[dict], list[str]]:
    docs, warnings = [], []
    for path in sorted(Path(documents_dir).iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            warnings.append(f"Skipped unsupported file: {path.name}")
            continue
        try:
            docs.extend(load_document_file(path))
        except Exception as exc:
            warnings.append(f"Could not load {path.name}: {exc}")
    return docs, warnings


def load_document_file(filepath: str | Path) -> list[dict]:
    path = Path(filepath)
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _single_document(path, _load_docx_text(path), "docx")
    if suffix == ".pdf":
        return _single_document(path, _load_pdf_text(path), "pdf")
    if suffix == ".xlsx":
        return _load_workbook_docs(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def build_index(raw_docs: list[dict], index_id: str):
    index_dir = get_indexes_dir() / sanitize_index_id(index_id)
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    nodes = _build_nodes(raw_docs)
    if not nodes:
        raise ValueError("No readable document text found to index")

    index = VectorStoreIndex(nodes, show_progress=True)
    index.storage_context.persist(persist_dir=str(index_dir))
    return index


def load_index(index_id: str):
    storage = StorageContext.from_defaults(persist_dir=str(get_indexes_dir() / sanitize_index_id(index_id)))
    return load_index_from_storage(storage)


def remove_index(index_id: str) -> bool:
    try:
        index_dir = get_indexes_dir() / sanitize_index_id(index_id)
        if index_dir.exists():
            shutil.rmtree(index_dir, ignore_errors=True)
        registry = load_registry()
        registry.pop(index_id, None)
        save_registry(registry)
        return True
    except Exception:
        return False


def update_registry(index_id: str, folder_path: Path, raw_docs: list[dict]) -> dict:
    index_id = sanitize_index_id(index_id)
    registry = load_registry()
    registry[index_id] = {
        "folder_path": str(folder_path),
        "folder_name": index_id,
        "documents": [{"filename": doc["filename"], "type": doc["type"]} for doc in raw_docs],
    }
    save_registry(registry)
    return registry[index_id]


def setup_embeddings() -> None:
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


def setup_llm() -> None:
    """Configure the answer LLM from INSIGHT_LLM_PROVIDER."""
    provider = os.getenv("INSIGHT_LLM_PROVIDER", "ollama").strip().lower()
    if provider == "ollama":
        Settings.llm = _build_ollama_llm()
    elif provider == "groq":
        Settings.llm = _build_groq_llm()
    else:
        raise RuntimeError(f"Unknown INSIGHT_LLM_PROVIDER '{provider}' (expected 'groq' or 'ollama')")


def _build_groq_llm() -> GroqCompatibleLLM:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required for Groq-backed chat")
    return GroqCompatibleLLM(model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL), api_key=api_key)


def _build_ollama_llm():
    from llama_index.llms.ollama import Ollama

    return Ollama(
        model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        request_timeout=120.0,
        temperature=0.0,
        context_window=int(os.getenv("OLLAMA_CONTEXT_WINDOW", "8192")),
    )


def call_model(prompt: str, *, node: str, timeout: float = 30.0, model: str | None = None) -> tuple[str, int]:
    """Call the configured provider directly for bounded agent nodes."""
    provider = os.getenv("INSIGHT_LLM_PROVIDER", "ollama").strip().lower()
    if provider == "ollama":
        selected_model = model or os.getenv("OLLAMA_PLANNER_MODEL" if node != "synthesis" else "OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/") + "/v1"
        client = OpenAI(api_key="ollama", base_url=base_url, timeout=timeout, max_retries=0)
    elif provider == "groq":
        selected_model = model or os.getenv("GROQ_PLANNER_MODEL" if node != "synthesis" else "GROQ_MODEL", DEFAULT_GROQ_MODEL)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is required for Groq-backed agent calls")
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=timeout,
            max_retries=0,
        )
    else:
        raise RuntimeError(f"Unknown INSIGHT_LLM_PROVIDER '{provider}'")

    request: dict[str, Any] = {
        "model": selected_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": {"planner": 256, "sufficiency": 256, "reformulate": 128}.get(node, 1024),
    }
    if provider == "ollama":
        request["extra_body"] = {"options": {"num_ctx": int(os.getenv("OLLAMA_CONTEXT_WINDOW", "8192"))}}
    response = client.chat.completions.create(**request)
    text = response.choices[0].message.content or ""
    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else estimate_tokens(prompt)
    output_tokens = usage.completion_tokens if usage else estimate_tokens(text)
    return text, input_tokens + output_tokens


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


_reranker: SentenceTransformerRerank | None = None


def get_reranker() -> SentenceTransformerRerank:
    """Load the cross-encoder re-ranker once and reuse it across requests."""
    global _reranker
    if _reranker is None:
        _reranker = SentenceTransformerRerank(model=RERANKER_MODEL, top_n=RERANK_TOP_N)
    return _reranker


def _build_query_engine(index):
    return index.as_query_engine(
        similarity_top_k=RETRIEVE_TOP_K,
        node_postprocessors=[get_reranker()],
        text_qa_template=QA_PROMPT,
    )


def ask_index_with_sources(index, message: str, history: list[dict] | None = None) -> dict[str, Any]:
    query_engine = _build_query_engine(index)
    chat_history = _to_chat_messages(history)
    if chat_history:
        # Condense the conversation + latest message into a standalone query (one Groq
        # call) so vector search sees a searchable question instead of a bare follow-up.
        response = CondenseQuestionChatEngine.from_defaults(query_engine=query_engine).chat(
            message, chat_history=chat_history
        )
    else:
        response = query_engine.query(message)
    return {"answer": str(response), "sources": format_source_nodes(getattr(response, "source_nodes", []))}


def _to_chat_messages(history: list[dict] | None) -> list[ChatMessage]:
    if not history:
        return []
    roles = {"user": MessageRole.USER, "assistant": MessageRole.ASSISTANT, "system": MessageRole.SYSTEM}
    messages = []
    for turn in history:
        content = (turn.get("content") or "").strip()
        if content:
            messages.append(ChatMessage(role=roles.get(turn.get("role"), MessageRole.USER), content=content))
    return messages


def retrieve_sources(index, query: str, top_k: int = RERANK_TOP_N) -> list[dict[str, Any]]:
    return format_source_nodes(index.as_retriever(similarity_top_k=top_k).retrieve(query))


def format_source_nodes(source_nodes) -> list[dict[str, Any]]:
    sources = []
    for source_node in source_nodes:
        node = getattr(source_node, "node", source_node)
        metadata = getattr(node, "metadata", {}) or {}
        score = getattr(source_node, "score", None)
        sources.append({
            "filename": metadata.get("filename", "unknown"),
            "type": metadata.get("type", "unknown"),
            "score": round(float(score), 4) if score is not None else None,
            "text": _shorten_text(_node_text(node)),
        })
    return sources


def _single_document(path: Path, text: str, doc_type: str) -> list[dict]:
    if not text.strip():
        return []
    return [{"filename": path.name, "text": _source_text(path.name, text), "type": doc_type}]


def _load_docx_text(path: Path) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _load_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(text for page in reader.pages if (text := page.extract_text()))


def _load_workbook_docs(path: Path) -> list[dict]:
    workbook = openpyxl.load_workbook(path, data_only=True)
    docs = []
    for sheet_name in workbook.sheetnames:
        text = _sheet_to_text(workbook[sheet_name])
        if text.strip():
            docs.append({
                "filename": f"{path.name}-{sheet_name}",
                "text": _source_text(path.name, text, sheet_name),
                "type": "xlsx",
            })
    return docs


def _sheet_to_text(sheet) -> str:
    rows = list(sheet.iter_rows(values_only=True))
    if not rows:
        return ""
    headers = [str(cell) if cell is not None else "" for cell in rows[0]]
    lines = []
    for row in rows[1:]:
        line = " | ".join(f"{header}: {cell}" for header, cell in zip(headers, row) if cell is not None)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _source_text(filename: str, text: str, sheet_name: str | None = None) -> str:
    header = f"Document filename: {filename}"
    if sheet_name:
        header += f"\nSheet: {sheet_name}"
    return f"{header}\n\n{text}"


def _build_nodes(raw_docs: list[dict]) -> list:
    text_docs = [_llama_doc(doc) for doc in raw_docs if doc["type"] != "xlsx"]
    sheet_docs = [_llama_doc(doc) for doc in raw_docs if doc["type"] == "xlsx"]
    nodes = []

    if text_docs:
        nodes.extend(SentenceSplitter(chunk_size=TEXT_CHUNK_SIZE, chunk_overlap=TEXT_CHUNK_OVERLAP).get_nodes_from_documents(text_docs))
    if sheet_docs:
        nodes.extend(
            SentenceSplitter(chunk_size=SPREADSHEET_CHUNK_SIZE, chunk_overlap=SPREADSHEET_CHUNK_OVERLAP).get_nodes_from_documents(sheet_docs)
        )
    return nodes


def _llama_doc(doc: dict) -> LlamaDocument:
    return LlamaDocument(text=doc["text"], metadata={"filename": doc["filename"], "type": doc["type"]})


def _node_text(node) -> str:
    if hasattr(node, "get_content"):
        return node.get_content(metadata_mode="none")
    return getattr(node, "text", "")


def _shorten_text(text: str) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= MAX_SOURCE_TEXT_CHARS:
        return cleaned
    return cleaned[:MAX_SOURCE_TEXT_CHARS].rstrip() + "..."
