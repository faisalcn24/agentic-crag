# Engineering Decisions

Decisions are recorded before measurements so later results are not tuned to a preferred narrative.

## D-001: Keep LlamaIndex at the index boundary

- **Decision:** Keep LlamaIndex for chunking, persisted vector indexing, and
  vector retrieval. Use the provider-neutral generation boundary in `rag.py` for
  both direct and agent answers.
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

## D-004: Local Ollama is the default execution mode

- **Decision:** Use `llama3.2:3b` through Ollama for query planning and answer synthesis. Groq remains selectable by the tester/operator.
- **Why:** The default path keeps generation local while preserving the requested provider-testing option.
- **Rejected for this scope:** End-user provider switching and model-routing cost optimization.

## D-005: Pin evaluation inputs and review answers directly

- **Decision:** Keep versioned JSONL datasets, save raw generated answers and retrieved sources, and inspect every answer against the expected answer and source text.
- **Why:** Direct review makes failures understandable and avoids delegating acceptance to another model.
- **Rejected:** Regenerating questions per run or using automated answer scores as current acceptance criteria. Automated scoring may be reconsidered later.

## D-006: Bound the retrieval workflow

- **Decision:** Retrieve exactly once for an ordinary standalone question. For a
  recognized multi-hop question, perform one non-recursive decomposition into
  two to four focused retrievals, require an exact source quote for each leg,
  and allow one corrective retrieval for each missing leg. Revalidate once, then
  answer or terminally abstain. Retain the 30-second deadline and token accounting.
- **Why:** The earlier open sufficiency/reformulation loop terminated 58 of 60
  agent cases through loop safeguards instead of normal answers. A single
  corrective pass addresses incomplete retrieval without recreating that loop.
- **Rejected:** Model-controlled repeated retrieval and an open-ended autonomous loop.
- **Implementation guard:** Generic exact evidence matches run before model
  coverage checks. Invalid corrective terms are rejected and receive one bounded
  query derived from the full missing sub-question, never another planning call.
- **Limitation:** Python cannot safely kill a running synchronous provider call. The request returns at the deadline, but a non-cooperative call already running in the worker relies on its configured HTTP timeout.
- **Implementation update:** D-013 removes the unnecessary graph framework while
  retaining these bounds.

## D-007: Pin MCP to the current stable SDK

- **Decision:** Pin `mcp==1.28.1`, implement the stable v1 FastMCP API against MCP specification revision `2025-11-25`, and verify it in process.
- **Why:** On 21 July 2026, Python SDK v2 is still beta and its targeted stable/spec date is in the future.
- **Rejected for this scope:** Pinning `2.0.0b2`, claiming compatibility with the not-yet-published `2026-07-28` specification, or claiming verification from an external MCP host.

## D-008: Add a minimal bundled local UI

- **Decision:** Serve one dependency-free HTML/CSS/JavaScript chat page from
  FastAPI. It uses the existing collection, `/chat`, and `/agent` endpoints;
  retain upload validation and latency/iteration JSONL telemetry.
- **Why:** The project now needs a graphical recruiter demo, but not a second
  frontend service or build toolchain.
- **Rejected for this scope:** Restoring the old Streamlit client, adding a
  JavaScript framework, or bundling unrelated public-deployment controls.

## D-009: Preserve honest project history

- **Decision:** Create meaningful commits only when the worktree can be staged without absorbing unrelated user changes; never backdate commits.
- **Why:** Six weeks of history cannot be truthfully produced in one implementation session.
- **Rejected:** Fabricated timestamps or filler commits.

## D-010: Constrain every model-generated control object

- **Decision:** Request provider-native JSON schemas for conversational query
  planning, spreadsheet plans, multi-hop decomposition, evidence coverage, and
  corrective queries. Validate every response; control failures use a small,
  domain-neutral fallback where one is safe or terminate.
- **Why:** Small local models were unreliable when JSON structure was requested
  only in prose.
- **Rejected:** Trusting unrestricted JSON, adding repair loops, or introducing
  another constrained-generation dependency before native schemas fail.

## D-011: Compile spreadsheet operations instead of executing model SQL

- **Decision:** Represent spreadsheet questions with a small validated operation
  plan and compile supported operations to parameterized in-memory DuckDB SQL.
- **Why:** Counts, filters, comparisons, grouping, and numeric conversions need
  deterministic table semantics rather than prose synthesis.
- **Rejected:** Executing arbitrary SQL generated by a model or replacing
  semantic workbook discovery with a separate permanent database.

## D-012: Use bounded decomposition for multi-document questions

- **Decision:** Decompose recognized complex questions once into two to four
  independent evidence queries. Verify each claimed supporting quote against the
  named source, correct missing legs once, and synthesize only after terminal
  revalidation succeeds.
- **Why:** One broad query can retrieve the right documents without exposing all
  required facts to the answer step.
- **Rejected:** Recursive agents, open-ended planning, and an indexing migration
  while source recall remains 100%.

## D-013: Keep startup lazy and orchestration direct

- **Decision:** Cache embeddings, provider configurations, and loaded indexes at
  their owning RAG boundaries. Run the bounded retrieval workflow directly and
  expose one MCP retrieval tool instead of equivalent search/inspection aliases.
- **Why:** The workflow is linear, non-recursive, and has no durable graph state.
  Recreating models and reopening an unchanged index for every request added
  latency without changing results, while duplicate MCP tools exposed the same
  reranked retrieval behavior under different names.
- **Rejected:** Eager model loading that delays the health/UI endpoints, a cache
  registry abstraction for three small caches, and merging retrieval,
  grounding, spreadsheet, and multi-hop responsibilities into one module.

## D-014: Add local fallback OCR and hybrid retrieval

- **Decision:** Keep pypdf for normal PDF extraction, invoke RapidOCR through
  ONNX Runtime only for direct image inputs or PDF pages with insufficient
  embedded text, and render those PDF pages with PDFium. Rank persisted chunks
  independently with BM25 and vector similarity, combine them with reciprocal
  rank fusion, then apply the existing cross-encoder reranker.
- **Why:** Text PDFs retain their fast path, OCR remains local and lazy, and
  exact identifiers can enter the candidate set without replacing the current
  persisted index or embedding model.
- **Rejected:** Requiring a system Tesseract installation, sending images to a
  hosted vision model, adding a second search service, or rebuilding every
  existing index solely to enable keyword retrieval.

## D-015: Keep evaluation knowledge out of production logic

- **Decision:** Production retrieval and answering must remain corpus-agnostic.
  Exact identifiers, versions, rows, and quotes are handled by generic parsing and
  grounding rather than question-specific branches or expected-answer tables.
- **Why:** Corpus-specific shortcuts inflated fixed-dataset results, expanded the
  runtime, and did not transfer to new documents.
- **Removed:** The duplicate LlamaIndex synthesis stack, hard-coded document facts,
  fixed spreadsheet column semantics, and deterministic decompositions tied to the
  bundled evaluation corpus.
- **Guard:** Keep those scenarios as black-box regression tests. A historical
  evaluation result may be retained, but it must be labeled historical after a
  behavior-changing refactor until a new live review is completed.
