from __future__ import annotations

import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from functions import mcp_server as server


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_mcp_lists_two_tools_and_collections_resource(monkeypatch):
    monkeypatch.setattr(server, "load_registry", lambda: {"demo": {"documents": []}})

    async with create_connected_server_and_client_session(server.mcp) as session:
        tools = await session.list_tools()
        resources = await session.list_resources()
        result = await session.read_resource("collections://all")

    assert {tool.name for tool in tools.tools} == {"search_corpus", "answer_question"}
    assert [str(resource.uri) for resource in resources.resources] == ["collections://all"]
    assert json.loads(result.contents[0].text)["indexes"][0]["index_id"] == "demo"


@pytest.mark.anyio
async def test_search_corpus_runs_in_process(monkeypatch):
    monkeypatch.setattr(server, "load_registry", lambda: {"demo": {"documents": []}})
    monkeypatch.setattr(server, "load_index", lambda _index_id: object())
    monkeypatch.setattr(
        server,
        "retrieve_sources",
        lambda _index, query, top_k: [{"filename": "demo.docx", "text": query, "score": 0.9, "type": "docx", "top_k": top_k}],
    )

    async with create_connected_server_and_client_session(server.mcp) as session:
        result = await session.call_tool("search_corpus", {"index_id": "demo", "query": "FR-006", "top_k": 5})

    assert result.isError is False
    assert "demo.docx" in result.content[0].text
