# Evaluation result artifacts

The current valid scorecards are the independently graded, same-stack,
corrective-ON runs completed on 18 August 2026:

- Qwen3 8B: `eval-raw-20260818T162900Z.json` and
  `eval-scored-20260818T162900Z.json`
- Llama 3.2 3B: `eval-raw-20260818T170047Z.json` and
  `eval-scored-20260818T170047Z.json`
- Injection fast loop: `eval-scored-20260818T155543Z.json`

Each model ran all 80 cases three times. The resulting 480 trials use the same
code, prompts, dataset, index, temperature, corrective mode, deadline, order, and
deterministic grader. See `docs/EVALUATION.md` for the scorecard, baseline diff,
and interpretation.

The pre-independent-grading artifacts and manual-review files remain invalidated
audit history. The 14 August independently graded scorecards are now superseded by
the injection-continuation rerun, but remain useful as the disclosed before-state.

`evals/run_evals.py` writes an append-only checkpoint, a raw result JSON, and an
automatically scored JSON. Every trial contains the question, expected answer,
actual answer, ranked retrieval trace, control trace, termination metadata,
latency, token usage, and model fingerprint. Every grade records correctness,
faithfulness, hallucination, citation accuracy, abstention behavior, retrieval
metrics, and diagnostics.

Collect a three-repeat run:

```powershell
python evals\run_evals.py --corrective on --repeats 3 `
  --checkpoint evals\results\checkpoint-<model>-on-3x.jsonl
```

Resume a compatible checkpoint with `--resume`. Re-score a raw artifact with the
current grader:

```powershell
python evals\run_evals.py --score-results <raw-result.json>
```

No manual-judgment file or separate judge model is used.
