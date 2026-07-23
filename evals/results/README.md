# Raw evaluation results

- `eval-20260721T153730Z.json` is a one-case agent/retrieval smoke run with judges skipped.
- `eval-20260721T154540Z.json` is a one-case framework integration run. It proves the RAGAS and DeepEval paths execute against local Ollama, and retains the `null` RAGAS faithfulness result caused by `llama3.2:3b` structured-output failures.
- `eval-20260721T155603Z.json` is a one-case single-shot smoke run after capping the Ollama context allocation; it retrieved the expected source and completed successfully.
- `eval-20260721T155815Z.json` is a one-case comparison plus one adversarial smoke run with judges skipped.

None of these files is a full scorecard, and a full judged scorecard was removed from the current local-only scope. If a future operator elects to publish one, it must contain all 60 golden cases in both modes and all 20 adversarial cases, and must be produced by:

```bash
python evals/run_evals.py --mode compare
```
