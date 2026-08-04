# Raw evaluation results

`evals/run_evals.py` writes timestamped JSON files here. Each file contains the
generated answer, expected answer, retrieved sources, citations,
expected-source recall, prompt-injection flags, and agent termination data.

The retained complete directly reviewed comparison is
`eval-20260803T202217Z.json`. Its top-level `manual_review` object records the
direct inspection and acceptance summary after OCR, hybrid retrieval, and the
initial bounded-correction rollout. It predates removal of corpus-specific answer
shortcuts and is historical evidence, not a current accuracy claim. See
`docs/EVALUATION.md`.

Only this final reviewed comparison is retained; superseded smoke and
intermediate runs are intentionally excluded.

Generate a complete comparison with:

```bash
python evals/run_evals.py --mode compare
```

These files are evidence for direct review, not automated correctness scores.
The consolidated methodology and current verification status are in
`docs/EVALUATION.md`.
