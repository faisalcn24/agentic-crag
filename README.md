# Insight AI

[![CI](https://github.com/faisalcn24/insight-ai-2/actions/workflows/ci.yml/badge.svg)](https://github.com/faisalcn24/insight-ai-2/actions/workflows/ci.yml)

**A local-first document intelligence system that answers questions with cited evidence.**

Insight AI turns PDF, DOCX, XLSX, and image files into persistent searchable
collections. It combines OCR, hybrid search, spreadsheet analysis, and bounded
multi-document reasoning in a browser application, REST API, and MCP server.

## Why it stands out

- **Grounded answers:** every factual response includes inspectable source excerpts;
  unsupported questions are explicitly declined.
- **Strong retrieval:** BM25 keyword search and vector similarity are fused, then
  cross-encoder reranked. This improves both semantic questions and exact identifiers.
- **Practical document support:** embedded PDF text uses fast extraction, while
  scanned PDFs and images use local OCR.
- **Controlled agent behavior:** ordinary questions retrieve once; multi-hop questions
  use one bounded, non-recursive pass of two to four focused retrievals.
- **Reliable spreadsheet analysis:** validated operations run through DuckDB instead
  of model-generated SQL.

**Stack:** Python 3.13, FastAPI, LlamaIndex, BM25, BGE embeddings/reranking,
RapidOCR, DuckDB, Ollama, optional Groq, MCP, pytest, and Ruff.

## Verified results

The complete post-OCR and hybrid-search review produced:

| Evaluation | Result |
| --- | ---: |
| Direct RAG | 60/60 |
| Agentic RAG | 60/60 |
| Adversarial cases | 20/20 |
| Expected-source recall | 100% |
| Spreadsheet questions | 15/15 |
| Multi-hop questions | 15/15 |
| Answer-not-present behavior | 10/10 |
| Automated tests | 128 passed, 2 optional live tests skipped |

Answers and citations were reviewed directly; no model judge was used. See the
[evaluation methodology](docs/EVALUATION.md) and
[reviewed raw output](evals/results/eval-20260731T114721Z.json).

These results measure the bundled, self-authored corpus. They do not claim the same
performance on arbitrary production documents. OCR tests cover generated printed-text
images and scanned PDFs, not every scan quality, layout, or handwriting style.

## How it works

```text
Documents -> parse/OCR -> persistent collection
                              |
Question -> BM25 + vector search -> rerank -> exact/table/multi-hop handling
                                                |
                                   grounded answer + citations
```

Parsing, OCR, embeddings, retrieval, and storage run locally. Ollama keeps answer
generation local as well. If Groq is selected, questions and retrieved excerpts are
sent to Groq for generation and planning.

## Run locally

Requirements: Python 3.13 and [Ollama](https://ollama.com/) with
`llama3.2:3b`.

### Windows

```powershell
py -3.13 -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull llama3.2:3b
.\run.bat
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
cp .env.example .env
ollama pull llama3.2:3b
bash scripts/run_dev.sh
```

Open `http://localhost:8000`, create a collection, and upload the sample files from
`documents/`. Useful demo questions:

- `What does FR-006 require?`
- `How much faster was tiny-smoke retrieval than large-policy-pack retrieval?`
- `Contrast the recommended local Windows and EC2 storage locations.`
- `What was the company payroll for 2026?` (demonstrates abstention)

Supporting evidence is expandable beneath every answer. Swagger is available at
`http://localhost:8000/docs`.

## Interfaces and scope

- Browser chat and collection management
- REST endpoints for collections, direct RAG, agentic RAG, retrieval, and metrics
- MCP tools for corpus search, grounded answers, and collection discovery
- PDF, DOCX, XLSX, PNG, JPEG, TIFF, BMP, and WebP input
- Local JSONL latency and retrieval-iteration telemetry

This is a trusted local demo, not a public SaaS product. It intentionally has no user
authentication, rate limiting, retention policy, or multi-tenant isolation. Add those
controls before exposing private documents or an untrusted public endpoint.

## Verify the project

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
ruff check functions tests evals scripts
python evals/run_evals.py --mode compare
```

For implementation details, see [architecture decisions](docs/DECISIONS.md),
[evaluation details](docs/EVALUATION.md), and the optional
[AWS deployment guide](docs/DEPLOYMENT.md).
