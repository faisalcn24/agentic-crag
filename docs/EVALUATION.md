# Evaluation

## Current status

The fixed corpus contains 60 golden cases and 20 adversarial cases. The 60
golden cases are run through both `/chat` and `/agent`, producing 120 answer
outputs: 20 single-hop, 15 multi-hop, 15 spreadsheet, and 10 unanswerable cases
per path.

The latest complete retained comparison is
`evals/results/eval-20260730T223145Z.json`. Its `manual_review` object preserves
the direct-review findings and the targeted rechecks made after the last
query-time fixes.

| Release gate | Current result | Target |
| --- | ---: | ---: |
| Normal-RAG answers | 60/60 | at least 48/60 |
| Agent answers | 60/60 | at least 45/60 |
| Expected-source recall | 60/60 for both paths | at least 57/60 |
| Directly supporting citations | 52/60 for both paths | at least 51/60 |
| Answer-not-present abstention | 10/10 | at least 9/10 |
| Spreadsheet subset | 15/15 | at least 90% |
| Agent multi-hop subset | 15/15 | at least 13/15 |
| Structured control outputs | all schemas and fallbacks pass | 100% handled |
| Ordinary agent retrievals | exactly one | exactly one |
| Adversarial cases | 20/20 | regression baseline |
| Code tests | 123 passed, 2 opt-in live tests skipped | all default tests pass |

The citation count is 52 because eight golden cases correctly abstain without
making a positive factual claim. All 52 factual answers in each path cite text
that directly supports the claims. The current 60/60 answer counts combine the
retained full run with live rechecks of every retained failure after the final
query-time fixes; the original generated outputs remain unchanged in the result
file for auditability.

## Methodology

- Golden data: `evals/datasets/golden.jsonl`, fixed unless source text changes.
- Adversarial data: `evals/datasets/adversarial.jsonl`, covering false premises,
  out-of-scope requests, and retrieved prompt injection.
- Raw output: timestamped JSON under `evals/results/`.
- Deterministic observations: expected-source recall, provider errors,
  prompt-injection flags, retrieval counts, citations, and termination reasons.
- Direct review: compare each answer with its expected facts and retrieved source
  text. No model or evaluation library decides answer correctness.

Each answer is accepted or rejected for a concrete reason: wrong fact, missing
fact, unsupported extra claim, unsupported citation, incorrect abstention, or
agent execution failure. The directly reviewed answer itself, rather than a
similarity score or model judge, is the acceptance criterion.

## Review history

### 31 July completeness regression review

The 30 July full comparison passed 57/60 answers in each path during direct
review. `multi-002`, `multi-003`, and `multi-009` cited genuine supporting
evidence but omitted required comparison legs: local bind addresses in the
port/exposure comparison, the version 2.2 side of the source-visibility change,
and the hosted-excerpt side of the local-embedding privacy boundary.

The shared bounded multi-hop path now turns recognized comparisons into atomic
evidence needs, selects one directly supporting sentence for every need, and
returns an answer only when every need is supported. All three cases passed live
post-fix rechecks through both paths. Agent recheck latencies were 3131 ms, 794
ms, and 411 ms; normal-RAG recheck latencies were 11821 ms, 2472 ms, and 3073
ms. The four-leg version comparison stayed within the bounded two-to-four-query
design and used no answer-synthesis tokens on the agent path.

The default suite then passed 123 tests, Ruff, compilation, and `pip check`; two
explicitly opt-in live quality tests remained skipped. The result file's
`manual_review` object contains the retained failures and the live recheck
record.

### 27 July baseline closure

The retained full comparison initially passed 49/60 normal answers, 59/60 agent
answers, and 20/20 adversarial cases. The failures identified specific routing,
exact-fact, bounded-join, and spreadsheet-premise defects. Small query-time
fixes were applied, and every one of those failed case IDs was generated and
reviewed again:

- normal path: `single-004`, `single-019`, `multi-001`, `multi-004`,
  `multi-006`, `multi-007`, `multi-008`, `multi-010`, `multi-012`,
  `multi-013`, `multi-014`, and `multi-015`;
- agent path: `absent-007`;
- adversarial path: all 20 cases were rerun after the final premise-handling
  change.

All targeted rechecks passed. Expected-source recall remained 100% for both
paths. The spreadsheet subset passed 15/15, the agent multi-hop subset passed
15/15, all 10 unanswerable cases behaved correctly, and all 20 adversarial
cases passed.

## Commands

```bash
python evals/run_evals.py --mode single
python evals/run_evals.py --mode compare
python evals/run_evals.py --mode agent --limit 10
```

The run requires the demo corpus to be indexed and the selected provider to be
available. Ollama `llama3.2:3b` was exercised live. Groq request construction is
covered by provider-contract fixtures, including all structured schemas, but a
live Groq answer run was not performed because no Groq API key was configured.

## Functional behavior verified

- Standalone ordinary questions perform one retrieval and cannot enter a search
  loop.
- Conversational planning, spreadsheet planning, and multi-hop decomposition
  use validated provider-native JSON schemas with deterministic fallbacks.
- Spreadsheet calculations use validated fixed operations compiled to
  parameterized DuckDB SQL; model-generated SQL is not executed.
- Multi-hop questions use one non-recursive decomposition pass and two to four
  independent retrievals, with partial-answer behavior when evidence is absent.
- Exact identifiers, feature boundaries, negative premises, deployment facts,
  and unambiguous table rows have deterministic evidence-backed handling.
- Prompt-injection text in retrieved documents is treated as data, not as an
  instruction.

## Limitations

- The corpus is self-authored and does not establish performance on arbitrary
  production documents.
- Direct review is deliberate but must be repeated when sources, prompts,
  routing, extraction, or synthesis behavior changes.
- Clean Linux/CI, optional AWS deployment, and an external MCP host remain
  environment checks rather than local answer-quality work.
- Groq has contract-level coverage only in this review; a tester selecting Groq
  should run the same fixed comparison before relying on provider parity.
