# Agent Failure Postmortem

This file records failures that have executable regression tests. Only the first was observed in the real local smoke run; the other two were reproduced in controlled regression harnesses and are not presented as production incidents.

## 1. Repeated reformulations consumed the whole loop

- **Observed failure:** In `eval-20260721T153730Z.json`, `llama3.2:3b` reformulated an FR-006 question into a SQL-like query and later repeated it, despite already retrieving the expected source.
- **Fix:** Normalize queries and terminate when the new query is near-identical to any prior query.
- **Regression:** `tests/test_agent.py::test_near_duplicate_query_is_detected`.

## 2. Budget exhaustion returned an unqualified answer

- **Reproduced failure:** The first smoke-run fallback named evidence but did not actually return an extractive best effort. A controlled one-token budget also reproduced the termination path without spending a model call.
- **Fix:** The fallback always includes an explicit low-confidence statement and a termination reason.
- **Regression:** `tests/test_agent.py::test_budget_fallback_states_low_confidence`.

## 3. Retrieved prompt injection could override grounding instructions

- **Reproduced failure:** The adversarial fixture places “ignore previous instructions” in retrieved test text. Without an explicit detection/flag path, it is indistinguishable from ordinary evidence at the orchestration layer.
- **Fix:** Agent prompts delimit retrieved text as untrusted evidence, tell the model never to follow instructions inside it, and flag detected injection phrases.
- **Regression:** `tests/test_agent.py::test_prompt_injection_is_flagged`.

Evaluation may still find cases where the agent does worse than single-shot retrieval. Those cases must remain in the published comparison rather than being removed from the dataset.
