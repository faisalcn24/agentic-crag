# Raw evaluation results

`evals/run_evals.py` writes timestamped JSON files here. Each file contains the
generated answer, expected answer, retrieved sources, citations,
expected-source recall, prompt-injection flags, and agent termination data.

The latest complete directly reviewed comparison is
`eval-20260731T114721Z.json`. Its top-level `manual_review` object records the
direct inspection and acceptance summary for the complete post-OCR and
post-hybrid-retrieval run.

Only this final reviewed comparison is retained; superseded smoke and
intermediate runs are intentionally excluded.

Generate a complete comparison with:

```bash
python evals/run_evals.py --mode compare
```

These files are evidence for direct review, not automated correctness scores.
The consolidated methodology and current results are in
`docs/EVALUATION.md`.
