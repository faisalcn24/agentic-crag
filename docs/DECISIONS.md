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

- **Decision:** Keep versioned JSONL datasets, save raw generated answers and the
  evidence returned with each answer, and inspect every answer against the expected
  answer and source text.
- **Why:** Direct review makes failures understandable and avoids delegating acceptance to another model.
- **Rejected:** Regenerating questions per run or using automated answer scores as current acceptance criteria. Automated scoring may be reconsidered later.

## D-006: Bound the retrieval workflow

- **Decision:** Retrieve exactly once for an ordinary standalone question. For a
  recognized multi-hop question, perform one non-recursive decomposition into
  two to four evidence obligations, require an exact source quote for each leg,
  and allow one corrective retrieval for each missing leg. Explicit requirements
  may share one broad initial retrieval; independent retrieval legs run
  concurrently. Revalidate once, then answer or terminally abstain. Retain the
  30-second deadline and token accounting.
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
  planning, multi-hop decomposition, evidence coverage, and corrective queries.
  Validate every response; control failures use a small, domain-neutral fallback
  where one is safe or terminate.
- **Why:** Small local models were unreliable when JSON structure was requested
  only in prose.
- **Rejected:** Trusting unrestricted JSON, adding repair loops, or introducing
  another constrained-generation dependency before native schemas fail.

## D-011: Index XLSX sheets as document evidence

- **Decision:** Parse XLSX sheets locally with openpyxl, preserve workbook and
  sheet provenance, convert rows to labeled text, and index them through the same
  Chroma and BM25 path as other documents. Retain deterministic unambiguous row
  lookup in the grounding layer.
- **Why:** One retrieval path supports spreadsheet search and cross-file evidence
  while keeping provenance and answer behavior consistent across formats.

## D-012: Use bounded decomposition for multi-document questions

- **Decision:** Decompose recognized complex questions once into two to four
  evidence obligations. Preserve explicit sentence and clause scope, reuse one
  initial result set when the original query retrieves evidence for every
  obligation, and run genuinely independent retrieval legs concurrently. Verify
  each claimed supporting quote against the named source, correct only missing
  legs once, and synthesize only after terminal revalidation succeeds.
- **Why:** Explicit multi-part prompts often retrieve both required documents in
  one well-formed query. Replanning and sequentially retrieving every clause spent
  the wall-clock budget without improving evidence, while focused retrieval is
  still necessary when the shared result set has a missing leg.
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
  registry abstraction for three small caches, and merging retrieval, grounding,
  and multi-hop responsibilities into one module.

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
  and deterministic decompositions tied to the bundled evaluation corpus.
- **Guard:** Keep those scenarios as black-box regression tests. A historical
  evaluation result may be retained, but it must be labeled historical after a
  behavior-changing refactor until a new live review is completed.

## D-016: Separate same-record answers from cross-source decomposition

- **Decision:** Route multi-field questions about one identifiable record through
  one retrieval and grounded structured-field presentation. Reserve decomposition
  for questions that compare, connect, or otherwise require evidence from distinct
  records or sources. Present one citation per source-backed answer group and use
  conversational prose for source overviews.
- **Why:** Treating every multi-part sentence as multi-hop added model calls until
  the 30-second deadline even when one retrieved passage contained every answer.
  Repeating the same citation on every field also obscured the answer without
  adding provenance.
- **Guard:** Field selection and presentation remain corpus-agnostic; exact values
  come from parsed source labels, and unfamiliar structures continue through the
  existing grounded synthesis path.

## D-017: Preserve conversational intent and source referents

- **Decision:** Have the schema-constrained conversation planner return both a
  standalone retrieval query and an answer operation (`answer` or `overview`),
  attach supporting filenames to assistant history turns, and use the same
  overview and structured-answer presentation in browser Agent and Direct RAG
  modes. Phrase matching is only a planner-failure fallback. Overview generation
  is schema-constrained and claim-grounded; structured sources additionally need
  every available core field represented or fall back to a generic grounded
  renderer.
- **Why:** Query rewriting correctly found a referenced document while losing the
  user's overview intent. A phrase-specific shortcut then worked only when the
  user explicitly wrote "image" or "document," and the extractive grounder
  rejected otherwise useful conversational paraphrases.
- **Guard:** Exact identifiers, numbers, paths, and quoted field values retain
  strict grounding. Conversational formatting may reorganize supported facts but
  cannot add definitions, causes, or implications absent from retrieved text.
  The fallback operates on generic labeled-field roles and contains no collection,
  filename, identifier, or expected-answer table.

## D-018: Verify answer coverage by obligation, not verbatim quote reproduction

- **Decision:** Keep literal source verification for evidence selection, then
  validate the grounded final answer against each original evidence obligation.
  Do not require synthesized prose to reproduce an entire verified quote. For
  plural requests, prefer the shortest passage that contains at least two matching
  items, and reject evidence whose public/private polarity conflicts with the
  question.
- **Why:** Requiring the final answer to repeat every word of a supporting passage
  replaced concise correct synthesis with unrelated neighboring security details.
  Obligation-level validation retains the requested paths and ports while source
  quote verification continues to prevent unsupported claims.
- **Guard:** This changes presentation coverage only. Evidence must still be an
  exact substring of a retrieved named source, and generated claims remain subject
  to the existing grounding pass.

## D-019: Separate collection overviews from single-source summaries

- **Decision:** Treat plural questions about the selected documents or collection
  as collection overviews. Retrieve up to ten passages, retain at most one
  representative passage per filename, and present up to six human-readable file
  topics with one citation per distinct source. Keep single-document and referential
  follow-up summaries on the model-generated, claim-grounded path.
- **Why:** A vague collection query returned duplicate passages from the highest
  scoring file. The small generation model then described that file as the entire
  collection or invented an unsupported umbrella description, which the grounding
  layer correctly rejected as answer-not-present.
- **Guard:** Collection topic labels are derived generically from retrieved source
  filenames and preserve acronym casing found in source text. The runtime contains
  no collection names, expected topics, or bundled-corpus answer table.

## D-020: Scope conversational claims to verified obligations

- **Decision:** When every leg of a multi-part question explicitly targets an
  identifier-bearing value such as a path, storage location, port, address, or ID,
  and every verified leg contains exact identifiers, retain only generated claims
  whose identifiers are contained in one verified leg, then order retained claims
  by the user's obligations. Treat loopback
  addresses as private presentation context and translate localhost-only binding
  into plain private-access wording when that is what the user asked. For a
  referential overview of a prior multi-source answer, use only the filenames
  actually cited in that answer and describe the prior question's topic directly.
- **Why:** A grounded but unrequested HTTP/CIDR caveat survived claim verification,
  while a vague follow-up retrieved unrelated passages and repeatedly abstained.
  Source support alone does not establish that a sentence is necessary for the
  requested answer.
- **Guard:** Identifier scoping activates only when every evidence obligation asks
  for an identifier-bearing value and has exact identifiers; semantic comparisons
  and other question types retain the existing grounded synthesis path. Referential
  topic wording is derived from conversation text and cited filenames, not
  collection-specific facts.

## D-021: Structure multi-part answers after grounding

- **Decision:** Once multi-part synthesized claims have passed evidence grounding,
  group them by the question's verified evidence obligations under semantic section
  headings, with one claim per bullet. Keep existing structured-field answers intact
  and leave simple answers as paragraphs. Present referential overviews of
  multi-part conversations as a short topic list. In the browser, parse only this
  small heading/list convention into native DOM nodes whose content is assigned with
  `textContent`.
- **Why:** Correct grounded facts were difficult to scan when rendered as one dense
  paragraph, especially when paths, public ports, and private ports appeared
  together.
- **Guard:** Formatting runs after grounding and cannot create, remove, or rewrite
  factual claims or citations. Headings are derived from obligation intent, not
  collection names, filenames, or expected facts. The UI never passes answer text
  through `innerHTML` or another executable HTML sink.

## D-022: Keep cached Hugging Face model loading offline at runtime

- **Decision:** Set Hugging Face and Transformers offline mode before importing
  their runtime libraries, while respecting an operator's explicit environment
  override.
- **Why:** A fresh API worker attempted a remote model metadata request even though
  the embedding model was cached locally. After the blocked request, Hugging Face
  retried with a closed HTTP client and surfaced `RuntimeError` before retrieval.
- **Guard:** This changes only runtime cache lookup. It does not change model names,
  embeddings, index compatibility, retrieval ranking, or answer content.

## D-023: Separate supporting evidence from retrieval diagnostics

- **Decision:** Return raw ranked passages and reranker scores only from retrieval
  diagnostics (`/retrieve` and MCP `search_corpus`) or with an abstention. For a
  completed answer, return only cited evidence, grouped by document and ordered by
  first citation appearance. Multi-part agent answers expose the exact passages
  verified for their evidence obligations; ordinary grounded answers expose the
  best cited passage from each named document.
- **Why:** The answer APIs previously returned every top-k candidate as a
  "supporting source," including unused documents and duplicate chunks. Cross-
  encoder outputs are useful for ordering candidates but are not calibrated
  confidence values: one required release-note passage ranked below unrelated
  candidates and still contained the exact answer.
- **Guard:** Do not use a fixed reranker-score cutoff to decide support. A passage
  supports an answer because the answer cites it and the grounding or obligation
  verifier confirmed its text. The browser hides scores for supporting evidence,
  numbers references by first citation appearance, and labels abstention results as
  closest retrieved passages rather than evidence.
- **Evaluation contract:** New runs report expected supporting-source recall,
  exact citation/source alignment, and grouped-evidence shape. These are mechanical
  provenance checks, not automated answer-correctness scores. The retained August 3
  `expected_source_recall` value measured the older top-k response and is not
  directly comparable.

## D-024: Persist vector indexes in Chroma without changing retrieval semantics

- **Decision:** Replace LlamaIndex's default JSON-backed `SimpleVectorStore` with
  embedded persistent ChromaDB. Use one isolated Chroma collection per named
  document collection, configure cosine HNSW search, and retain node copies in the
  LlamaIndex docstore so the existing BM25 corpus and reciprocal-rank fusion remain
  unchanged. Load Chroma lazily and disable its anonymous telemetry.
- **Why:** The application now exposes an explicit vector-database boundary with
  durable collection management, while preserving local execution and the existing
  multi-file retrieval, reranking, grounding, and citation contracts.
- **Rejected for this scope:** Keeping `SimpleVectorStore`, which remains adequate
  for experiments but does not demonstrate a database-backed index, and adding a
  separately managed Chroma server to the current single-process local demo.
- **Migration:** Existing `SimpleVectorStore` indexes are not read through the new
  boundary. Recreate each collection from its uploaded files after this storage
  layout change.
