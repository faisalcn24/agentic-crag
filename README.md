# Agentic CRAG

[![CI](https://github.com/faisalcn24/insight-ai-2/actions/workflows/ci.yml/badge.svg)](https://github.com/faisalcn24/insight-ai-2/actions/workflows/ci.yml)

**A local-first RAG system that answers questions over documents with cited evidence.**

Agentic CRAG turns PDF, DOCX, XLSX, and image files into persistent searchable
collections. It combines OCR, hybrid retrieval, spreadsheet analysis, and bounded
multi-document reasoning behind a browser UI, REST API, and MCP server.

## What it demonstrates

- **Grounded answers:** factual claims include inspectable citations; unsupported
  questions are declined.
- **Hybrid retrieval:** BM25 and vector results are fused, then cross-encoder
  reranked for semantic questions and exact identifiers.
- **Bounded reasoning:** multi-hop questions are decomposed once, checked for
  verbatim evidence, and receive at most one corrective retrieval per missing leg.
- **Document intelligence:** local OCR handles images and scanned PDFs; validated
  DuckDB operations handle spreadsheet questions.
- **Local-first execution:** parsing, OCR, embeddings, retrieval, and storage stay
  local. Ollama also keeps generation local; optional Groq sends only the question
  and retrieved excerpts.

**Stack:** Python 3.13, FastAPI, LlamaIndex, BM25, BGE embeddings/reranking,
RapidOCR, DuckDB, Ollama, optional Groq, MCP, pytest, and Ruff.

## Verified results

The retained full-corpus review records:

| Evaluation | Result |
| --- | ---: |
| Direct RAG | 60/60 |
| Bounded agent RAG | 60/60 |
| Adversarial cases | 20/20 |
| Expected-source recall | 100% |
| Spreadsheet / multi-hop subsets | 15/15 each |
| Answer-not-present behavior | 10/10 |
| Automated tests | 135 passed |
| Quality integration suite | 24/24 with live checks enabled |

Answers and citations were reviewed directly without a model judge. After the latest
coverage-gate change, all 15 retained multi-hop answers preserved deterministic
parity, and the four affected normal corpus cases reproduced their answers
byte-for-byte in live runs.

A controlled live forced-miss test hid required evidence from eight reviewed
multi-document questions. The correction-disabled path completed 0/8; bounded
correction completed 7/8 byte-for-byte and 8/8 under direct review, restoring all
required citations without creating a retrieval loop.

See the [evaluation methodology](docs/EVALUATION.md) and
[reviewed output](evals/results/eval-20260803T202217Z.json). These measurements use
the bundled, self-authored corpus; they do not establish general production accuracy.
OCR tests cover generated printed-text scans, not arbitrary layouts or handwriting.

## How it works

```text
Documents -> parse/OCR -> persistent collection
                              |
Question -> BM25 + vector search -> rerank -> exact/table handling -> cited answer
                                      |
                     multi-hop -> quote check -> one optional correction
                                      |
                              answer or abstain
```

## Run locally

Requirements: Python 3.13 and [Ollama](https://ollama.com/) with
`llama3.2:3b`.

```powershell
py -3.13 -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull llama3.2:3b
.\run.bat
```

Open `http://localhost:8000`, create a collection, and upload files from
`documents/`. On Linux or macOS, use `bash scripts/run_dev.sh`.

Useful demo questions:

- `What does FR-006 require?`
- `How much faster was tiny-smoke retrieval than large-policy-pack retrieval?`
- `Contrast the recommended local Windows and EC2 storage locations.`
- `What was the company payroll for 2026?` - demonstrates abstention

Supporting evidence is expandable below each answer. API documentation is available
at `http://localhost:8000/docs`.

## Interfaces and boundaries

- Browser chat and collection management
- REST endpoints for direct RAG, bounded agent RAG, retrieval, metrics, and collections
- MCP tools for corpus search, grounded answers, and collection discovery
- PDF, DOCX, XLSX, PNG, JPEG, TIFF, BMP, and WebP input

This is a trusted local demo, not a public multi-tenant service. Add authentication,
HTTPS, rate limiting, and a retention policy before exposing private documents or an
untrusted public endpoint.

## Verify

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
ruff check functions tests evals scripts
python evals/run_evals.py --mode compare
```

More detail: [architecture decisions](docs/DECISIONS.md),
[evaluation record](docs/EVALUATION.md), and
[AWS deployment guide](docs/DEPLOYMENT.md).
