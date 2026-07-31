# Insight AI — Agentic Document Analysis

[![CI](https://github.com/faisalcn24/insight-ai-2/actions/workflows/ci.yml/badge.svg)](https://github.com/faisalcn24/insight-ai-2/actions/workflows/ci.yml)

A local-first document intelligence application that answers grounded questions over
PDF, DOCX, and XLSX collections. It combines semantic retrieval, exact fact
handling, deterministic spreadsheet analysis, and bounded multi-document
reasoning while returning the supporting source excerpts.

**Current delivery:** a recruiter-ready local demo with a bundled browser chat
page and documented API for trusted use. It does not claim a hosted demo.

## What this project demonstrates

- A complete ingestion-to-answer RAG pipeline with persistent named collections.
- Hybrid handling of prose and tables: LlamaIndex retrieval for documents and
  validated DuckDB operations for spreadsheet calculations.
- Bounded agent behavior: one retrieval for ordinary questions and one
  non-recursive, two-to-four-query pass for multi-hop questions.
- Lazy runtime initialization: health and UI routes stay lightweight while
  embeddings, provider configurations, and unchanged indexes are reused.
- Grounded answers with inspectable citations, explicit abstention, and
  prompt-injection regression coverage.
- Reproducible evaluation using fixed datasets, retained raw outputs, and direct
  answer review instead of an automated model judge.

**Stack:** Python 3.13, FastAPI, vanilla HTML/CSS/JavaScript, LlamaIndex,
DuckDB, Ollama, optional Groq, MCP, pytest, and Ruff.

## What reviewers can verify

- Upload public PDF, DOCX, and XLSX files into persistent named collections.
- Compare direct RAG with a bounded retrieval agent from the browser chat page.
- Inspect expandable source snippets and retrieval scores under each answer.
- Call retrieval and agent answers through MCP.
- Run a fixed 60-question golden set and 20 adversarial cases locally.
- Inspect query latency and retrieval iterations in local JSONL logs.

Ordinary agent questions retrieve exactly once. Recognized multi-hop questions
use one non-recursive decomposition pass with two to four focused retrievals.
Spreadsheet analysis uses validated fixed operations compiled to in-memory
DuckDB SQL. The 30-second response deadline and configured token ceiling remain
enforced. The deadline cannot forcibly stop a non-cooperative provider call
already running in the worker; production calls rely on their HTTP timeout.

> Privacy: the default configuration uses local Ollama `llama3.2:3b`, so answer generation stays on the machine. If `INSIGHT_LLM_PROVIDER=groq` is enabled, retrieved excerpts are sent to Groq; use public/demo documents only.

## Verified results

| Check | Result |
| --- | ---: |
| Normal-RAG golden answers | 60/60 |
| Agent golden answers | 60/60 |
| Adversarial cases | 20/20 |
| Expected-source recall | 100% on both paths |
| Spreadsheet subset | 15/15 |
| Agent multi-hop subset | 15/15 |
| Answer-not-present behavior | 10/10 |
| Automated code tests | 123 passed, 2 opt-in live tests skipped |

See [the complete methodology and review history](docs/EVALUATION.md) and the
retained raw comparison in `evals/results/eval-20260730T223145Z.json`. Answer
quality is reviewed directly against expected facts and retrieved source text;
there is no automated model judge. The corpus is self-authored, so these
results do not establish performance on arbitrary production documents. The
latest review used live local Ollama generation; Groq has request-contract tests
but was not run live without an API key.

## Short demo walkthrough

After completing the quick start below, open the chat page at
[`http://localhost:8000`](http://localhost:8000):

1. Create a collection named `recruiter-demo` and upload the six sample files
   in `documents/`.
2. Select **Direct RAG** and ask an exact lookup: `What does FR-006 require?`
3. In the same mode, ask a spreadsheet calculation:
   `How much faster was tiny-smoke retrieval than large-policy-pack retrieval?`
4. Select **Agent** and ask a multi-document question:
   `Contrast the recommended local Windows and EC2 storage locations.`
5. Ask an unsupported question:
   `What was the company payroll for 2026?`

The first three answers demonstrate exact retrieval, deterministic table
calculation, and bounded evidence joining. The last demonstrates grounded
abstention instead of fabrication. Expand **supporting sources** beneath an
answer to inspect its evidence. Swagger remains available at
[`http://localhost:8000/docs`](http://localhost:8000/docs).

## Architecture

```text
FastAPI
    +-- GET  /      --------> bundled browser chat and collection upload page
    +-- POST /chat  --------> route -> exact/table/join or retrieve -> synthesize
    +-- POST /agent --------> route -> one retrieval or bounded 2-4-way decomposition
    +-- POST /retrieve -----> reranked source inspection, no answer LLM
    +-- GET  /metrics ------> JSONL-derived latency/iteration summary
    |
    +-- local parsing: pypdf, python-docx, openpyxl
    +-- local embeddings: BAAI/bge-small-en-v1.5
    +-- local persisted LlamaIndex storage
    +-- in-memory DuckDB for validated spreadsheet operations
    +-- local Ollama llama3.2:3b by default

MCP server
    +-- search_corpus
    +-- answer_question
    +-- collections://all
```

The agent wraps the existing retriever; it does not replace document parsing,
embeddings, persistence, or `/chat`. Runtime initialization belongs to the RAG
boundary: index build/load initializes embeddings, answering initializes the
selected LLM, and unchanged persisted indexes are reused. This keeps `/health`
and the browser page free of eager model-loading work.

## Quick start (Linux/macOS)

Requirements: Python 3.13 and either [Ollama](https://ollama.com/) for the
default local provider or a Groq API key for the optional hosted provider.

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
cp .env.example .env
ollama pull llama3.2:3b
bash scripts/run_dev.sh
```

When using Groq, set `INSIGHT_LLM_PROVIDER=groq` and `GROQ_API_KEY` in `.env`
and skip the `ollama pull` command.

Open `http://localhost:8000` for the graphical chat page or
`http://localhost:8000/docs` for Swagger. The launcher runs FastAPI on
`127.0.0.1:8000`, prefers the repository virtual environment, and respects the
provider configured in `.env`. When Ollama is selected, its service must also
be running.

### Windows

```powershell
py -3.13 -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull llama3.2:3b
.\run.bat
```

The default environment values are:

```env
INSIGHT_LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
OLLAMA_PLANNER_MODEL=llama3.2:3b
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

The project pins `mcp==1.28.1` and exposes retrieval, answering, and collection
discovery through an MCP server. `search_corpus` accepts `top_k` from 1 to 20,
so a second retrieval-inspection alias is unnecessary.

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

Generate the full local answer set for direct review:

```bash
python -m pip install -r requirements-dev.txt
python evals/run_evals.py --mode compare
```

Generate a two-question smoke sample:

```bash
python evals/run_evals.py --mode agent --limit 2
```

The runner records the expected answer, generated answer, retrieved sources, exact source recall, and agent termination. Answer correctness, abstention behavior, and citation support are reviewed directly from the saved JSON output. Automated answer scoring is deferred as a possible future addition.

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

The included CI workflow is configured to run those checks on Python 3.13.
Generated environments, caches, runtime data, coverage, build output, editor
state, and temporary files are excluded by `.gitignore`; source, tests,
datasets, documentation, and the latest reviewed evaluation remain visible.
Design rationale and rejected alternatives are in
[docs/DECISIONS.md](docs/DECISIONS.md).

## Repository map

- `functions/rag.py` — parsing, storage, retrieval, exact facts, reranking, and providers.
- `functions/agent.py` — typed bounded retrieval-and-answer workflow.
- `functions/spreadsheet.py` — spreadsheet parsing, exact lookups, validated
  operations, and DuckDB execution.
- `functions/multihop.py` — bounded decomposition and evidence-backed joins.
- `functions/mcp_server.py` — MCP tools and collections resource.
- `functions/api.py` — FastAPI endpoints, chat-page route, and upload lifecycle.
- `functions/static/chat.html` — dependency-free browser chat interface.
- `functions/telemetry.py` — JSONL query events and summaries.
- `evals/` — fixed datasets, runner, and raw reviewed results.
- `deploy/` — EC2, Nginx, and API systemd templates.
