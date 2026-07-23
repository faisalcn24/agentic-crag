# Engineering Decisions

Decisions are recorded before measurements so later results are not tuned to a preferred narrative.

## D-001: Extend the existing LlamaIndex pipeline

- **Decision:** Keep LlamaIndex for parsing, chunking, indexing, retrieval, and synthesis.
- **Why:** The deployed application already has a working persisted-index boundary and `/retrieve` seam.
- **Rejected:** A LangChain rewrite. It would add migration risk without improving the measured retrieval baseline.

## D-002: Keep `BAAI/bge-small-en-v1.5` embeddings local

- **Decision:** Continue using the existing local embedding model.
- **Why:** Full documents remain on the host and existing indexes stay compatible.
- **Rejected:** Hosted embeddings and a new vector database. Neither is needed for this corpus.

## D-003: Retain format-aware chunking

- **Decision:** Keep 500-token overlapping prose chunks and 4,000-token non-overlapping spreadsheet chunks.
- **Why:** Spreadsheet headers and rows need to remain together; prose benefits from smaller retrieval units.
- **Rejected:** One chunk size for every format.

## D-004: Local Ollama is the current execution mode

- **Decision:** Use `llama3.2:3b` through Ollama for planning, sufficiency, answer synthesis, and the provisional judge.
- **Why:** The user explicitly requested a fully local implementation for the current iteration. The default path makes no hosted model calls.
- **Rejected for this scope:** Groq-hosted generation, a hosted stronger judge, the full judged scorecard, and model-routing cost optimization. The provider and evaluation hooks remain available, but only local smoke results are retained and no hosted or cost claims are made.
- **Limitation:** The judge is not stronger than the answer model. Judge-based scores are provisional and may be noisy or flattering.

## D-005: Pin evaluation inputs and tolerate judge noise

- **Decision:** Keep versioned JSONL datasets, temperature 0 where supported, one fixed judge model, and a ±3 point comparison band.
- **Why:** Exact pass thresholds make stochastic evaluations flaky.
- **Rejected:** Regenerating questions per run or treating one-point changes as real.

## D-006: Use a bounded LangGraph around retrieval

- **Decision:** Add a graph with at most four retrieval iterations, a 30-second response deadline, token accounting, and repeated-query detection.
- **Why:** It adds agent behavior without replacing retrieval.
- **Rejected:** An open-ended autonomous loop.
- **Limitation:** Python cannot safely kill a running synchronous provider call. The request returns at the deadline, but a non-cooperative call already running in the worker relies on its configured HTTP timeout.

## D-007: Pin MCP to the current stable SDK

- **Decision:** Pin `mcp==1.28.1`, implement the stable v1 FastMCP API against MCP specification revision `2025-11-25`, and verify it in process.
- **Why:** On 21 July 2026, Python SDK v2 is still beta and its targeted stable/spec date is in the future.
- **Rejected for this scope:** Pinning `2.0.0b2`, claiming compatibility with the not-yet-published `2026-07-28` specification, or claiming verification from an external MCP host.

## D-008: Keep the runtime surface API-only

- **Decision:** Retain FastAPI, upload type/size validation, and latency/iteration JSONL telemetry. Remove the Streamlit client, public demo/GIF, rate limiting, collection/spend caps, and automatic retention.
- **Why:** The user selected an API-only minimal setup and explicitly kept only upload validation and performance telemetry.
- **Rejected for this scope:** A bundled UI and operational controls that are not needed for the trusted local workflow. The deployment guide calls out the consequences of public exposure.

## D-009: Preserve honest project history

- **Decision:** Create meaningful commits only when the worktree can be staged without absorbing unrelated user changes; never backdate commits.
- **Why:** Six weeks of history cannot be truthfully produced in one implementation session.
- **Rejected:** Fabricated timestamps or filler commits.

## D-010: Isolate the RAGAS 0.4.3 legacy import

- **Decision:** The eval runner installs a narrow runtime alias for `langchain_community.chat_models.vertexai` before importing RAGAS.
- **Why:** RAGAS 0.4.3 imports that removed legacy module even though the selected local OpenAI-compatible Ollama adapter does not use Vertex AI. Current LangGraph requires the newer LangChain dependency line, so downgrading the transitive stack would break the agent pin.
- **Rejected:** Adding another top-level dependency, downgrading LangGraph, or silently omitting RAGAS metrics.
