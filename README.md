# Insight AI — Agentic Document Analysis

[![CI](https://github.com/faisalcn24/insight-ai-2/actions/workflows/ci.yml/badge.svg)](https://github.com/faisalcn24/insight-ai-2/actions/workflows/ci.yml)

A local-first document assistant that uses bounded agentic retrieval to answer grounded questions over PDF, DOCX, and XLSX collections.

**Runtime scope:** trusted local API use. A public live demo, bundled client, and GIF were deliberately removed from this iteration; this README does not invent a URL or recording.

## What reviewers can verify

- Upload public PDF, DOCX, and XLSX files into persistent named collections.
- Compare single-shot RAG with a bounded LangGraph agent through `/chat` and `/agent`.
- Inspect source snippets and retrieval scores in API responses.
- Call retrieval, agent answers, and raw retrieval inspection through MCP.
- Run a fixed 60-question golden set and 20 adversarial cases locally.
- Inspect query latency and retrieval iterations in local JSONL logs.

The agent checks evidence sufficiency, reformulates weak queries, and re-retrieves. It returns after at most four retrieval iterations, the 30-second response deadline, or the configured token ceiling. Repeated queries terminate, and budget exhaustion produces an explicit low-confidence answer. The deadline cannot forcibly stop a non-cooperative provider call already running in the worker; production calls rely on their HTTP timeout.

> Privacy: the default configuration uses local Ollama `llama3.2:3b`, so answer generation stays on the machine. If `INSIGHT_LLM_PROVIDER=groq` is enabled, retrieved excerpts are sent to Groq; use public/demo documents only.

## Results and limitations

The earlier 100% evaluation snapshot was deleted. See [the evaluation methodology and retained smoke results](docs/EVALUATION.md) and [three regression-tested agent failures](docs/POSTMORTEM.md).

A full judged scorecard is outside the current local-only scope. The retained smoke results use the same `llama3.2:3b` model for answers and judging, so judge-based values are provisional. The corpus is self-authored; hand-written questions reduce question leakage, not corpus authorship bias.

## Architecture

```text
FastAPI
    +-- POST /chat  --------> retrieve 20 -> bge rerank -> top 3 -> synthesize
    +-- POST /agent --------> plan -> raw top-8 retrieve -> sufficiency -> reformulate loop
    +-- POST /retrieve -----> raw vector retrieval, no answer LLM
    +-- GET  /metrics ------> JSONL-derived latency/iteration summary
    |
    +-- local parsing: pypdf, python-docx, openpyxl
    +-- local embeddings: BAAI/bge-small-en-v1.5
    +-- local persisted LlamaIndex storage
    +-- local Ollama llama3.2:3b by default

MCP server
    +-- search_corpus
    +-- answer_question
    +-- inspect_retrieval
    +-- collections://all
```

The agent wraps the existing retriever; it does not replace document parsing, embeddings, persistence, or `/chat`.

## Quick start (Linux/macOS)

Requirements: Python 3.13 and [Ollama](https://ollama.com/).

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
cp .env.example .env
ollama pull llama3.2:3b
bash scripts/run_dev.sh
```

Open `http://localhost:8000/docs`. The launcher runs the FastAPI service on `127.0.0.1:8000`; Ollama must also be running.

### Windows

```powershell
py -3.13 -m venv venv
venv\Scripts\Activate.ps1
py -3.13 -m pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull llama3.2:3b
.\run.bat
```

The default environment values are:

```env
INSIGHT_LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
OLLAMA_PLANNER_MODEL=llama3.2:3b
OLLAMA_JUDGE_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CONTEXT_WINDOW=8192
INSIGHT_STORAGE_DIR=.insight_data
```

## API

- `GET /health`
- `GET /indexes`
- `POST /indexes`
- `DELETE /indexes/{index_id}`
- `POST /chat`
- `POST /agent`
- `POST /retrieve`
- `GET /metrics`

`POST /chat` and `POST /agent` accept:

```json
{
  "index_id": "demo-documents",
  "message": "What does FR-006 require?",
  "history": [{"role": "user", "content": "Earlier question"}]
}
```

## MCP

The project pins `mcp==1.28.1`, the stable Python v1 SDK available on 21 July 2026, and targets MCP specification revision `2025-11-25`. The future `2026-07-28` revision and Python SDK v2 stable release are not claimed.

Run the stdio server:

```bash
python -m functions.mcp_server
```

Run Streamable HTTP when a host requires it:

```bash
INSIGHT_MCP_TRANSPORT=streamable-http python -m functions.mcp_server
```

The test suite connects a client directly to the server instance in memory, with no subprocess or port. External MCP-host verification was removed from this scope and is not claimed.

## Evaluation

The stable datasets are:

- `evals/datasets/golden.jsonl`: 60 questions — 20 single-hop, 15 multi-hop, 15 spreadsheet, 10 unanswerable.
- `evals/datasets/adversarial.jsonl`: 20 false-premise, out-of-scope, and retrieved prompt-injection cases.

Optionally run the full local comparison (not a current completion requirement):

```bash
python -m pip install -r requirements-dev.txt
python evals/run_evals.py --mode compare
```

Run a fast pipeline smoke test without LLM judges:

```bash
python evals/run_evals.py --mode agent --limit 2 --skip-judges
```

RAGAS supplies context precision, context recall, and faithfulness. DeepEval supplies answer correctness, hallucination rate, and citation accuracy. Comparisons use a ±3-point tolerance band; smaller differences are noise.

## Telemetry

Each completed query appends its mode, latency, and retrieval iterations to `<INSIGHT_STORAGE_DIR>/metrics/queries.jsonl`. The summary reports:

- p50/p95 latency
- mean retrieval iterations

Summarize locally:

```bash
python -m functions.telemetry
```

Token usage, provider details, prompts, answers, and cost are not recorded.

## Upload validation

- PDF, DOCX, and XLSX extension/MIME allowlist; legacy `.xls` is rejected.
- 10 MB combined request upload limit in FastAPI and Nginx.
- Secrets stay in environment variables or `.env`, never Git.

This intentionally minimal API has no authentication, rate limiting, collection cap, spend cap, or automatic data retention. Add appropriate controls before exposing private documents or an untrusted public endpoint.

## AWS deployment

The retained EC2 + Nginx + systemd topology runs the API only:

```text
public :80 -> Nginx -> FastAPI 127.0.0.1:8000
```

```bash
bash deploy/setup_ec2.sh     # fresh host
bash deploy/update_app.sh    # existing host
```

See [README_AWS_HYBRID.md](README_AWS_HYBRID.md). The current local-only configuration should be tested for available RAM before using a small EC2 instance.

## Development checks

```bash
python -m compileall -q functions tests evals scripts
python -m pytest -q
python -m pip check
ruff check functions tests evals scripts
```

The included CI workflow runs those checks on Python 3.13. Design rationale and rejected alternatives are in [docs/DECISIONS.md](docs/DECISIONS.md).

## Repository map

- `functions/rag.py` — parsing, storage, indexing, retrieval, reranking, and providers.
- `functions/agent.py` — typed LangGraph state and bounded agent loop.
- `functions/mcp_server.py` — MCP tools and collections resource.
- `functions/api.py` — FastAPI endpoints and upload lifecycle.
- `functions/telemetry.py` — JSONL query events and summaries.
- `evals/` — fixed datasets, runner, and raw results.
- `deploy/` — EC2, Nginx, and API systemd templates.
