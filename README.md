# Agentic CRAG

[![CI](https://github.com/faisalcn24/agentic-crag/actions/workflows/ci.yml/badge.svg)](https://github.com/faisalcn24/agentic-crag/actions/workflows/ci.yml)

Local-first document question answering with citations, OCR, hybrid search, and
one bounded corrective retrieval pass.

Agentic CRAG turns PDF, DOCX, XLSX, and image files into persistent collections
that can be queried through a browser, REST API, or MCP server. Parsing,
embeddings, retrieval, and storage remain local. Generation uses local Ollama by
default, with Groq available as an explicit hosted option.

## What stands out

- **Grounded answers:** unsupported claims are removed and missing answers abstain.
- **Hybrid retrieval:** BM25 and vector results are fused and reranked, improving
  both semantic search and exact-identifier lookup.
- **Bounded correction:** multi-part questions are decomposed, checked against
  exact source quotes, and receive at most one corrective retrieval per missing
  part—never an open-ended agent loop.
- **Document support:** local OCR handles images and scanned PDFs; validated
  operations compiled to DuckDB handle spreadsheet analysis.
- **Three interfaces:** a dependency-free web UI, FastAPI endpoints, and MCP tools
  for Claude Code or another compatible host.

**Stack:** Python 3.13, FastAPI, LlamaIndex, BM25, BGE embeddings/reranking,
RapidOCR, DuckDB, Ollama, optional Groq, MCP, pytest, and Ruff.

## Architecture

```text
documents -> parse / OCR -> chunk -> persistent vector index
                                      |
question -> BM25 + vector -> fusion -> rerank -> grounded answer
                                      |
                     multi-part -> quote coverage check
                                      |
                         one correction -> answer or abstain
```

## Run locally

Install [Python 3.13](https://www.python.org/) and
[Ollama](https://ollama.com/), then:

```powershell
py -3.13 -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull llama3.2:3b
.\run.bat
```

Open `http://localhost:8000`, create a collection, and upload files from
`documents/`. API documentation is at `http://localhost:8000/docs`.

Try:

- `What does FR-006 require?`
- `How much faster was tiny-smoke retrieval than large-policy-pack retrieval?`
- `Contrast the recommended local Windows and EC2 storage locations.`
- `What was the company payroll for 2026?` (demonstrates abstention)

Run the MCP server with `python -m functions.mcp_server`. It exposes collection
discovery, corpus search, and grounded answering tools over stdio.

## Verification

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
ruff check functions tests evals scripts
python -m compileall -q functions tests evals scripts
```

The current generalized runtime passes **101 default tests plus two opt-in live
model/reranker checks**. A retained 60-question corpus review is available
as [historical evaluation evidence](docs/EVALUATION.md), but it predates the
removal of corpus-specific shortcuts and is not presented as a current
general-data accuracy score.

## Boundaries

This is a trusted local demo, not a public multi-tenant service. Add
authentication, HTTPS, rate limiting, and a retention policy before accepting
private uploads through a public endpoint. OCR is tested on printed text, not
handwriting or arbitrary layouts.

See [engineering decisions](docs/DECISIONS.md),
[evaluation methodology](docs/EVALUATION.md), and
[deployment notes](docs/DEPLOYMENT.md).
