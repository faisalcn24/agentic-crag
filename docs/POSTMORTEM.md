# Agent Failure Postmortem

This file records failures that have executable regression tests. Only the first was observed in the real local smoke run; the other two were reproduced in controlled regression harnesses and are not presented as production incidents.

## 1. Model-controlled retrieval retries became the normal exit

- **Observed failure:** The previous full run found the expected source but still ended 51 of 60 agent cases through repeated-query protection and another seven through the iteration limit.
- **Fix:** Remove the model-controlled evidence check and retry path. Ordinary
  questions now retrieve once and answer or abstain; recognized multi-hop
  questions use one bounded, non-recursive decomposition pass.
- **Regression:** `tests/test_agent.py::test_ordinary_question_answers_after_one_retrieval` and `tests/test_agent.py::test_single_retrieval_answers_without_a_loop_limit`.

## 2. Budget exhaustion returned an unqualified answer

- **Reproduced failure:** The first smoke-run fallback named evidence but did not actually return an extractive best effort. A controlled one-token budget also reproduced the termination path without spending a model call.
- **Fix:** The fallback always includes an explicit low-confidence statement and a termination reason.
- **Regression:** `tests/test_agent.py::test_budget_fallback_states_low_confidence`.

## 3. Retrieved prompt injection could override grounding instructions

- **Reproduced failure:** The adversarial fixture places “ignore previous instructions” in retrieved test text. Without an explicit detection/flag path, it is indistinguishable from ordinary evidence at the orchestration layer.
- **Fix:** Agent prompts delimit retrieved text as untrusted evidence, tell the model never to follow instructions inside it, and flag detected injection phrases.
- **Regression:** `tests/test_agent.py::test_prompt_injection_is_flagged`.

Evaluation may still find cases where the agent does worse than single-shot retrieval. Those cases must remain in the published comparison rather than being removed from the dataset.
