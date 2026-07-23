from __future__ import annotations

import json
import os

from mcp.server.fastmcp import FastMCP

from .agent import run_agent
from .rag import load_index, load_registry, retrieve_sources, sanitize_index_id, setup_embeddings


mcp = FastMCP(
    "Insight AI Document Analysis",
    instructions="Search and answer questions over locally indexed document collections.",
    json_response=True,
)


@mcp.tool()
def search_corpus(index_id: str, query: str, top_k: int = 5) -> dict:
    """Retrieve the most similar chunks from a named local collection."""
    index = _collection(index_id)
    return {"sources": retrieve_sources(index, query, top_k=_clamp(top_k))}


@mcp.tool()
def answer_question(index_id: str, question: str) -> dict:
    """Run the bounded retrieval agent over a named local collection."""
    return run_agent(_collection(index_id), question)


@mcp.tool()
def inspect_retrieval(index_id: str, query: str, top_k: int = 20) -> dict:
    """Inspect raw vector retrieval without an answer-model call."""
    index = _collection(index_id)
    return {"sources": retrieve_sources(index, query, top_k=_clamp(top_k))}


@mcp.resource("collections://all")
def collections() -> str:
    """List locally persisted document collections."""
    payload = [{"index_id": key, **value} for key, value in load_registry().items()]
    return json.dumps({"indexes": payload}, ensure_ascii=False)


def _collection(index_id: str):
    safe_id = sanitize_index_id(index_id)
    if safe_id not in load_registry():
        raise ValueError(f"Collection not found: {safe_id}")
    setup_embeddings()
    return load_index(safe_id)


def _clamp(top_k: int) -> int:
    return min(max(int(top_k), 1), 20)


def main() -> None:
    transport = os.getenv("INSIGHT_MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
