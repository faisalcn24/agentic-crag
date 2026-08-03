# Evaluation

## Current status

The fixed corpus contains 60 golden cases and 20 adversarial cases. The 60
golden cases are run through both `/chat` and `/agent`, producing 120 answer
outputs: 20 single-hop, 15 multi-hop, 15 spreadsheet, and 10 unanswerable cases
per path.

The latest complete retained answer comparison is
`evals/results/eval-20260803T202217Z.json`. Its `manual_review` object records
the direct inspection of all 120 golden answers and all 20 adversarial answers
generated after the exact-quote corrective pass. Every answer is byte-for-byte
identical to the prior directly reviewed pass; retrieval and termination metadata
were reviewed separately.

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
| Code tests | 135 passed; quality integration suite 24/24 with live checks enabled | all pass |

The citation count is 52 because eight golden cases correctly abstain without
making a positive factual claim. All 52 factual answers in each path cite text
that directly supports the claims. The current 60/60 answer counts come from
the retained full run itself; no targeted recheck was needed.

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

### 4 August forced-miss coverage review

The earlier three-trial synthetic test proved that the corrective branch could
work, but a stronger ablation exposed two permissive shortcuts: one accepted a
single overlapping term as complete evidence, and another accepted any passage
containing the same identifier. With one required source hidden from the initial
retrievals of eight reviewed multi-document questions, correction activated in
only three cases and reproduced just two retained answers.

Those shortcuts were removed. Complete deterministic joins remain for facts that
can prove every required leg, including ports and exposure, versioned source
visibility, and the local/hosted privacy boundary. Other non-empty retrieval legs
must supply model-selected verbatim evidence. After correction, a complete
deterministic join may terminate successfully; otherwise the existing exact-quote
revalidation runs once and can only answer or abstain. Invalid corrective plans now
fall back to the full missing sub-question instead of an identifier-only query.

The real-corpus ablation then produced 0/8 complete baseline answers versus 7/8
byte-for-byte retained answers and 8/8 correct answers under direct review. The
eighth answer stated the same two facts with an alternate directly supporting
evaluation-plan citation. Required target citations improved from 0/8 to 8/8.
Every case stayed bounded to one corrective retrieval per missing leg; some
questions had two missing legs. A targeted rerun verified the final authentication
case after making that decomposition atomic and source-specific.

The four normal corpus questions affected by the refactor reproduced their retained
answers byte-for-byte with normal termination and no corrective pass. The standard
suite passed 135 tests with two optional skips; the opt-in OCR, reranker, and live
Ollama suite passed all 24 tests. This ablation measures controlled failures on the
bundled corpus, not general-corpus accuracy.

### 3 August bounded-correction review

The first live rollout exposed a latency regression rather than an accuracy
improvement: all 15 agent multi-hop cases failed because model-based coverage
checks consumed the 30-second request budget. Twelve ended at the wall-clock
limit and three ended during corrective planning. Those failed intermediate
runs were retained during diagnosis but are not release evidence.

The shared path was tightened without increasing the timeout. Existing
deterministic evidence joins now prove exact coverage before model review, empty
legs are marked missing immediately, and a corrective plan may not introduce
terms or identifiers absent from the missing sub-question. At that review point,
an invalid plan received one deterministic exact-ID fallback and still could not
loop; the 4 August review replaced it with a fuller grounded fallback.

The final real-model run passed all 60 Agentic answers, all 20 adversarial cases,
the 15/15 multi-hop subset, and 100% expected-source recall. All Agentic and
adversarial answers were byte-for-byte identical to the previous directly
reviewed pass. The separate Direct run also reproduced all 60 previous answers
with 100% expected-source recall. There were no generation errors or non-normal
terminations, and none of the fixed-corpus cases needed correction because their
initial evidence was already complete.

A controlled forced-miss A/B exercised the new branch with Ollama
`llama3.2:3b`. Both paths initially received the AX-101 evidence and missed
BX-202. Across three trials, the initial-only baseline completed 0/3 answers;
the bounded path completed 3/3, adding exactly one retrieval and both source
citations. Mean latency rose from 3.802 seconds to 6.562 seconds. The model added
unsupported catalyst language to each corrective plan; validation rejected it
and the exact-ID fallback recovered the correct BX-202 source. This demonstrates
bounded recovery for that controlled failure, not general-corpus improvement.

### 31 July OCR and hybrid-retrieval review

After adding local OCR and BM25/vector reciprocal-rank fusion, the complete
comparison was regenerated with local Ollama `llama3.2:3b` against the same
persisted demo corpus. Direct review passed all 60 normal-RAG answers, all 60
agent answers, and all 20 adversarial answers. Expected-source recall remained
100% for both golden paths.

All 52 factual answers in each path used directly supporting citations; the
other eight answers in each path were exact abstentions. All 10 unanswerable
cases behaved correctly, including two evidence-backed feature-boundary
corrections. The 15 spreadsheet cases and 15 agent multi-hop cases passed. All
45 ordinary agent cases retrieved exactly once; the multi-hop cases used two to
four retrievals. All eight retrieved prompt-injection cases were flagged and
answered from genuine corpus evidence. There were no provider, generation, or
termination errors.

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
- Conversational planning, spreadsheet planning, multi-hop decomposition,
  evidence coverage, and corrective queries use validated provider-native JSON
  schemas.
- Spreadsheet calculations use validated fixed operations compiled to
  parameterized DuckDB SQL; model-generated SQL is not executed.
- Multi-hop questions begin with one non-recursive decomposition and two to four
  retrievals. Each leg needs a verified exact quote; missing legs receive at most
  one corrective retrieval and one terminal revalidation before answer or abstention.
- Deterministically provable complete joins bypass model coverage checks. Corrective
  plans with ungrounded terms use one reported query derived from the full missing
  sub-question rather than a second model call.
- Exact identifiers, feature boundaries, negative premises, deployment facts,
  and unambiguous table rows have deterministic evidence-backed handling.
- Prompt-injection text in retrieved documents is treated as data, not as an
  instruction.

## Limitations

- The corpus is self-authored and does not establish performance on arbitrary
  production documents.
- Direct review is deliberate but must be repeated when sources, prompts,
  routing, extraction, or synthesis behavior changes.
- OCR quality is verified on generated printed text, not arbitrary scans,
  handwriting, unusual layouts, or every supported image format.
- Clean Linux/CI, optional AWS deployment, and an external MCP host remain
  environment checks rather than local answer-quality work.
- Groq has contract-level coverage only in this review; a tester selecting Groq
  should run the same fixed comparison before relying on provider parity.
