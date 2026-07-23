# Evaluation

## Status

The old five-row, 100% snapshot was removed because it was measured on a tiny self-authored set and overstated confidence. The replacement uses a fixed 60-question golden set plus 20 adversarial cases. Results are written here only by `python evals/run_evals.py`; no score is filled by hand.

Current smoke-test judge: local Ollama `llama3.2:3b`, temperature 0. This is the same model family and size used for answers, not a stronger independent judge. The resulting LLM-judged metrics are **provisional**. A full judged scorecard and local compute accounting were removed from the current scope.

A one-case integration validation completed on 21 July 2026. RAGAS context precision and recall plus all three DeepEval metrics returned values, while RAGAS faithfulness returned `null` after the 3B judge failed its structured-output schema on all four attempts. The raw failure is retained in `evals/results/eval-20260721T154540Z.json`. This smoke result is not presented as the project scorecard.

## Methodology

- Golden data: `evals/datasets/golden.jsonl`, fixed and reviewed as source text changes.
- Adversarial data: `evals/datasets/adversarial.jsonl`, covering false premises, out-of-scope requests, and prompt injection.
- RAGAS: context precision, context recall, and faithfulness.
- DeepEval: answer correctness, hallucination rate, and citation accuracy.
- Deterministic checks: expected-source recall and adversarial refusal/flagging.
- The adversarial aggregate is behavior-only. Free-form false-premise corrections remain raw with a `null` automated result rather than being guessed correct.
- Comparison tolerance: ±3 percentage points. Smaller movements are reported as noise, not improvements.
- Generation and judge temperature: 0 where the provider supports it.
- No metric is rounded up. Raw per-case output is retained under `evals/results/`.

Run the single-shot baseline:

```bash
python evals/run_evals.py --mode single
```

Run single-shot and agentic modes side by side:

```bash
python evals/run_evals.py --mode compare
```

Use `--limit N` for a smoke run. A full run requires the demo corpus to be indexed and Ollama to be running with `llama3.2:3b`.

## Scorecard

<!-- EVAL_RESULTS_START -->
No full scorecard is claimed or required in the current local-only scope. The runner can insert a dated result table if a future operator deliberately completes a full judged run.
<!-- EVAL_RESULTS_END -->

## Remaining limitation

The corpus itself is still self-authored. Hand-written questions reduce direct question leakage, but do not make the underlying documents independent or representative of arbitrary production data.
