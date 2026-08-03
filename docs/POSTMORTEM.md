# Agent Failure Postmortem

This file records failures that have executable regression tests. Only the first was observed in the real local smoke run; the other two were reproduced in controlled regression harnesses and are not presented as production incidents.

## 1. Model-controlled retrieval retries became the normal exit

- **Observed failure:** The previous full run found the expected source but still ended 51 of 60 agent cases through repeated-query protection and another seven through the iteration limit.
- **Fix:** Remove the model-controlled evidence check and retry path. Ordinary
  questions now retrieve once and answer or abstain; recognized multi-hop
  questions use one bounded, non-recursive decomposition pass.
- **Current guard:** The later multi-hop exact-quote check permits only one
  corrective pass; failed revalidation goes directly to abstention and cannot
  route back to planning or retrieval.
- **Regression:** `tests/test_agent.py::test_ordinary_question_answers_after_one_retrieval`, `tests/test_agent.py::test_single_retrieval_uses_extractable_evidence_without_synthesis`, and `tests/test_agent.py::test_failed_revalidation_abstains_without_second_loop`.

## 2. Budget exhaustion returned an unqualified answer

- **Reproduced failure:** The first smoke-run fallback named evidence but did not actually return an extractive best effort. A controlled one-token budget also reproduced the termination path without spending a model call.
- **Fix:** The fallback always includes an explicit low-confidence statement and a termination reason.
- **Regression:** `tests/test_agent.py::test_budget_fallback_states_low_confidence`.

## 3. Retrieved prompt injection could override grounding instructions

- **Reproduced failure:** The adversarial fixture places “ignore previous instructions” in retrieved test text. Without an explicit detection/flag path, it is indistinguishable from ordinary evidence at the orchestration layer.
- **Fix:** Agent prompts delimit retrieved text as untrusted evidence, tell the model never to follow instructions inside it, and flag detected injection phrases.
- **Regression:** `tests/test_agent.py::test_prompt_injection_is_flagged`.

Evaluation may still find cases where the agent does worse than single-shot retrieval. Those cases must remain in the published comparison rather than being removed from the dataset.

## 4. Exact-quote checks exhausted the multi-hop latency budget

- **Observed failure:** The first real-model rollout run failed all 15 multi-hop
  cases: 12 hit the 30-second wall-clock limit and three ended during corrective
  planning. The quote check was unnecessarily calling the model even when the
  existing deterministic join had already proven every evidence leg.
- **Fix:** Accept deterministic verbatim evidence matches before model review,
  mark empty legs missing without a model call, reject ungrounded corrective terms,
  and use at most one full-missing-sub-question fallback before terminal revalidation.
- **Regression:** The final live run restored 15/15 multi-hop answers and 100%
  source recall. A controlled real-model forced-miss A/B first completed 0/3 baseline
  answers versus 3/3 bounded answers. A later eight-case corpus ablation completed
  0/8 with correction disabled versus 7/8 byte-exact and 8/8 directly reviewed
  answers with bounded correction.
