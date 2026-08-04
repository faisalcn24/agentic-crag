# Evaluation

## Current status

The current generalized runtime passes **101 default tests**. Both optional live
checks also pass; they load the real reranker and Ollama and run only when
`AGENTIC_CRAG_RUN_LIVE_QUALITY_TESTS=1`.

The retained file
`evals/results/eval-20260803T202217Z.json` records a directly reviewed historical
run on the bundled corpus:

| Historical measurement | Result |
| --- | ---: |
| Direct RAG | 60/60 |
| Bounded agent RAG | 60/60 |
| Adversarial cases | 20/20 |
| Expected-source recall | 100% |

That run predates the August cleanup that removed corpus-specific exact-answer
branches and the duplicate LlamaIndex generation layer. It remains useful as an
audit artifact, but its accuracy totals are **not current release claims**. A new
full live run and direct answer review are required before publishing replacement
quality numbers.

## Fixed corpus

`golden.jsonl` contains 60 questions: 20 single-hop, 15 multi-hop, 15
spreadsheet, and 10 unanswerable. `adversarial.jsonl` contains 20 false-premise,
out-of-scope, and retrieved prompt-injection cases.

The corpus is self-authored. It is designed to expose retrieval, citation,
abstention, table, and multi-document failures—not to estimate production accuracy
on arbitrary documents.

## Review method

1. Run both direct and agent modes against the unchanged dataset.
2. Retain raw answers, retrieved sources, citations, termination metadata, and
   provider errors.
3. Compare every factual claim with its cited source text.
4. Reject wrong facts, omitted required parts, unsupported additions, incorrect
   abstentions, and citations that do not support the claim.
5. Record results without an automated model judge.

```powershell
python evals\run_evals.py --mode compare
```

The selected provider and indexed demo collection must be running. Groq currently
has provider-contract coverage only; provider parity requires its own live run.

## What automated tests cover

- local OCR for generated image and scanned-PDF text;
- hybrid BM25/vector fusion, reranking, and exact identifiers;
- claim grounding, citation rebuilding, and abstention;
- bounded decomposition, quote verification, one correction, and terminal failure;
- validated spreadsheet plans compiled to parameterized DuckDB SQL;
- API upload rules, MCP tools, telemetry, and provider request schemas.

## Known limits

- No current general-corpus accuracy measurement exists.
- OCR coverage does not establish handwriting or arbitrary-layout quality.
- Clean Linux deployment, AWS deployment, and external MCP-host behavior are
  separate environment checks.
- Direct review must be repeated whenever retrieval, prompting, routing,
  extraction, or grounding behavior changes.
