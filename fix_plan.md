# INSIGHT.AI RAG Functionality Fix Plan

This plan focuses only on answer quality and agent behavior. Security hardening, deployment infrastructure, and unrelated architectural changes are outside its scope.

## 1. Current measured state

Latest complete reviewed comparison: `evals/results/eval-20260730T223145Z.json`

- The 60 golden cases pass 60/60 through normal RAG and 60/60 through the agent
  after recorded targeted rechecks of every retained failure.
- All 20 adversarial cases pass.
- Expected-source recall is 100% for both paths.
- All 52 factual answers per path have directly supporting citations; the other
  eight cases correctly abstain without a positive factual claim.
- Spreadsheet questions pass 15/15 and agent multi-hop questions pass 15/15.
- Ordinary agent questions retrieve exactly once; multi-hop questions use one
  bounded, non-recursive two-to-four-query pass.
- The automated suite currently passes 123 tests, with two opt-in live quality
  tests skipped by default. Full review details and the
  distinction between retained output and post-fix rechecks are in
  `docs/EVALUATION.md`.

## 2. Original diagnosis

The baseline showed that finding a relevant document was not the main problem.
The failures that drove this plan were:

1. The agent made unreliable decisions about when to search again.
2. The generator did not consistently combine all evidence needed for complex questions.
3. Spreadsheet calculations were handled as text-generation problems.
4. Unsupported statements and inaccurate citations were not reliably removed before an answer was returned.
5. Small local models did not reliably follow prompt-only JSON formats.

Because retrieval already finds the expected source consistently, the first fixes should target routing, reasoning, structured data, and answer verification. Rebuilding the index or retrieving many more chunks is not justified by the current measurements.

## 3. Phase 1: Simplify the existing agent

Keep a typed, bounded workflow with explicit routing and execution limits. The
initial implementation used LangGraph, but the completed workflow has no loops
or durable graph state, so the 31 July simplicity audit replaced the framework
wrapper with direct orchestration while preserving the same routing behavior.

**Status:** Implemented and accepted on 27 July 2026. Ordinary questions use one
retrieval, and all 60 agent golden cases pass direct review.

### Changes

- Add a simple query router with four paths:
  - normal document question;
  - exact fact or identifier lookup;
  - spreadsheet calculation;
  - complex multi-part or multi-document question.
- Send normal questions through one retrieval and one answer-generation step.
- Do not let the model repeatedly decide whether to search again.
- If retrieval is empty, abstain without another model decision.
- Do not retry ordinary questions. Bounded sub-question retrieval is added only with the later multi-hop phase.
- Record a clear termination reason for every agent run.

### Verification

- All existing code tests continue to pass.
- At least 95% of ordinary questions finish without a retry.
- Every ordinary agent question uses one retrieval and ends through answer, abstention, time limit, token limit, or an explicit error.
- Expected-source recall remains at least 95%.

## 4. Phase 2: Make control outputs reliable

**Status:** Implemented and accepted on 27 July 2026. Planner,
spreadsheet-plan, and decomposition schemas are validated, provider-native
schema requests are covered for Ollama and Groq, and invalid output reaches a
deterministic fallback.

Use schema-constrained output for routing, decomposition, and spreadsheet query plans.

### Changes

- Use Ollama's native JSON-schema structured output for the local tester path.
- Use Groq's structured-output request format when the tester selects Groq.
- Define small schemas with fixed choices instead of asking the model for unrestricted JSON.
- Validate every control response before using it.
- If validation fails, use a safe deterministic fallback rather than starting a repair loop.
- Do not add Outlines initially. In the current Ollama-server setup, Ollama itself performs the structured-output enforcement.

### Verification

- Routing, decomposition, and spreadsheet-plan fixtures have zero unhandled parse failures.
- Invalid model output always reaches the documented fallback.
- The same schema tests pass with both tester-selectable providers.

## 5. Phase 3: Add hybrid spreadsheet querying

**Status:** Implemented and accepted on 27 July 2026. The spreadsheet golden
subset passes 15/15 through validated operations compiled to in-memory DuckDB
queries.

Do not abandon vector retrieval for spreadsheets. Use vector retrieval to discover the relevant workbook or sheet, then use DuckDB for exact tabular operations.

### Changes

- Preserve the current spreadsheet ingestion and searchable text representation.
- Load the selected sheet into an in-memory DuckDB table for analytical questions.
- Ask the model for a validated query plan, not arbitrary SQL.
- Initially allow only these operations:
  - select columns;
  - filter rows;
  - sort rows;
  - count;
  - sum and average;
  - minimum and maximum;
  - difference and comparison;
  - simple grouping.
- Compile the validated plan into DuckDB SQL in application code.
- Return the workbook, sheet, columns, and relevant rows as answer evidence so the result can be cited.
- Keep descriptive and cross-document spreadsheet questions on the semantic retrieval path when no calculation is required.

### Verification

- Add deterministic tests for every supported operation.
- Add tests for ambiguous column names, numeric values stored as text, blanks, and mixed data types.
- Spreadsheet answers must match independently calculated expected values.
- Spreadsheet citations must identify the source workbook and sheet.

## 6. Phase 4: Add bounded multi-hop decomposition

**Status:** Implemented and accepted on 27 July 2026. The agent multi-hop subset
passes 15/15 using a single non-recursive two-to-four-subquestion pass.

Add decomposition over the current index before changing the indexing strategy.

### Changes

- Detect questions that require multiple facts, documents, entities, or calculations.
- Generate two to four focused sub-questions using a validated schema.
- Retrieve evidence for each sub-question independently.
- Track which sub-questions have sufficient evidence.
- Generate the final answer only after every required sub-question is answered or explicitly marked unsupported.
- Cite the evidence used for each part of the final answer.
- Allow one decomposition pass only; do not create recursive sub-agents or open-ended planning loops.
- Implement this within the existing bounded workflow. Evaluate LlamaIndex's
  `SubQuestionQueryEngine` only if it reduces code and passes the same tests.

### Verification

- Create a fixed multi-hop test set covering comparison, combination, and cross-document questions.
- At least 85% of that set must include all required facts.
- Missing evidence must cause a partial-answer warning or abstention, not an invented completion.
- No multi-hop test may enter an unbounded retrieval loop.

## 7. Phase 5: Review answers directly

**Status:** Completed on 27 July 2026. Results, retained failures, and post-fix
rechecks are recorded in `docs/EVALUATION.md` and the result file's
`manual_review` object.

Do not use another model or evaluation library to decide whether an answer is correct. Generate answers with the real RAG paths, then inspect them directly against the expected answer and retrieved source text.

### Changes

- Keep the current 60 golden cases and 20 adversarial cases unchanged as the
  regression baseline. Run the 60 golden cases through both paths to produce
  120 answer outputs.
- Run the evaluator only to collect actual answers, retrieved sources, citations, and termination data.
- Review every answer directly against its expected answer and retrieved evidence.
- Record a simple pass or fail plus a short reason: wrong fact, missing fact, unsupported extra claim, wrong citation, incorrect abstention, or agent-loop failure.
- Keep deterministic code tests for routing, multi-hop coverage, structured output, spreadsheet calculations, and exact values.
- Store raw generated answers under `evals/results/` and summarize the direct review in `docs/EVALUATION.md`.
- Automated answer scoring is deferred as possible future work.

### Minimum functionality targets

- At least 80% of normal RAG answers pass direct review.
- At least 75% of agent RAG answers pass direct review.
- At least 85% of answers have citations that directly support their claims.
- At least 90% of answer-not-present questions abstain correctly.
- Expected-source recall: at least 95%.
- Spreadsheet operation accuracy: at least 90% on deterministic tests.
- Multi-hop required-fact coverage: at least 85%.
- Structured control-output parse success: 100%, including documented fallbacks.
- Ordinary agent questions use exactly one retrieval and never enter a search loop.
- All code tests, lint checks, and compilation checks pass.

These targets are release gates for functionality. The answer-quality targets are decided by direct inspection, not model-generated scores. If a target is missed, use the failed cases to identify the responsible component before making another architectural change.

All minimum functionality targets currently pass. This does not replace a new
direct review after a behavior-affecting code or corpus change.

## 8. Deferred changes

The following changes should not be implemented unless later measurements show that they address a real bottleneck:

- **Adding an orchestration framework:** not needed while the bounded workflow
  remains linear and non-recursive.
- **Hierarchical indexing:** reconsider only if multi-hop evidence cannot be found in the current index.
- **Retrieving 100 or more candidates:** reconsider only if expected-source or required-evidence recall falls below target.
- **Changing the reranker or compiling it with ONNX:** reconsider only after measuring an unacceptable reranking latency or accuracy problem.
- **Using a 30B SQL model:** reconsider only if the small validated spreadsheet query planner cannot meet the accuracy target.
- **Adding Outlines:** reconsider only if native provider schemas cannot produce reliable control outputs.
- **Automated answer scoring:** deferred; use direct review instead.
- **Recursive multi-agent orchestration:** do not add unless bounded decomposition demonstrably cannot solve the remaining multi-hop cases.

## 9. Implementation order

Implement and evaluate one phase at a time:

1. Simplify agent routing and stop repeated searches.
2. Add reliable structured control outputs.
3. Add hybrid DuckDB spreadsheet querying.
4. Add bounded multi-hop decomposition.
5. Generate the fixed answer set and review the answers directly.

Do not begin a later phase to compensate for a failed earlier phase. Diagnose the failed cases first and make the smallest change that addresses them.

## References

- DuckDB Python API: <https://duckdb.org/docs/stable/clients/python/overview>
- Ollama structured outputs: <https://docs.ollama.com/capabilities/structured-outputs>
- Outlines model architecture: <https://dottxt-ai.github.io/outlines/main/guide/architecture/>
