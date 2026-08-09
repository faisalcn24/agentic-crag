# Raw evaluation results

`evals/run_evals.py` writes timestamped JSON files here. Each file contains the
generated answer, expected answer, returned sources, citations, provenance metrics,
prompt-injection flags, and agent termination data. For completed answers, returned
sources are cited supporting evidence grouped by document. For abstentions, they
are closest retrieval diagnostics.

The retained complete directly reviewed comparison is
`eval-20260803T202217Z.json`. Its top-level `manual_review` object records the
direct inspection and acceptance summary after OCR, hybrid retrieval, and the
initial bounded-correction rollout. It predates removal of corpus-specific answer
shortcuts and is historical evidence, not a current accuracy claim. See
`docs/EVALUATION.md`.

That historical file uses the former `expected_source_recall` metric over the
top-k response. New runs use `expected_supporting_source_recall`,
`citation_source_alignment`, and `supporting_evidence_contract`; do not compare the
old source-recall percentage directly with these metrics.

Only this final reviewed comparison is retained; superseded smoke and
intermediate runs are intentionally excluded.

Generate a complete comparison with:

```bash
python evals/run_evals.py --mode compare
```

These files are evidence for direct review, not automated correctness scores.
The consolidated methodology and current verification status are in
`docs/EVALUATION.md`.
