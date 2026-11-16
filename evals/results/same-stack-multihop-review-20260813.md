# Same-stack multi-hop model ablation

This direct review compares local Ollama `qwen3:8b` and `llama3.2:3b` on the same
13 August 2026 code, persistent Chroma index, prompts, 8,192-token context window,
30-second deadline, corrective-ON configuration, 15 fixed multi-hop questions, and
three repetitions. Only `OLLAMA_MODEL` and `OLLAMA_PLANNER_MODEL` changed. No judge
model was used.

Raw artifacts:

- Qwen: `eval-raw-20260813T155714Z.json`
- Llama: `eval-raw-20260813T190415Z.json`
- Shared index fingerprint:
  `1b863ba9ffcbc57da76633d465647e05a7063c5572f897b19863a75195107ecc`

Answer correctness is the fraction of required reference facts stated correctly,
with contradictions and unsupported additions penalized. Scores were assigned by
directly comparing every distinct answer with its expected answer and retrieved
evidence. Repeated byte-identical answers retain the same judgment.

| Case | Qwen runs 1/2/3 | Llama runs 1/2/3 | Direct-review basis |
| --- | ---: | ---: | --- |
| multi-001 | 0 / 0 / 0 | 0 / 0 / 0 | Both models omitted the Windows and EC2 locations. |
| multi-002 | .75 / 0 / 1 | 0 / 0 / 0 | Qwen run 1 gave all ports and exposure but omitted explicit loopback addresses; run 3's cited fallback contained the complete deployment statement. |
| multi-003 | .25 / .25 / .25 | .25 / .25 / .25 | Qwen stated structured reporting but omitted the full 2.2-to-2.4 visibility contrast; Llama's fallback contained only part of 2.2. |
| multi-004 | .75 / .75 / .75 | 0 / 0 / 0 | Qwen gave FR-006 and Recall@5 correctly but left their relationship implicit. |
| multi-005 | 0 / 0 / 0 | 0 / 0 / 0 | Neither model supplied the two requested security reasons. |
| multi-006 | 1 / 1 / 1 | 1 / 1 / 1 | Both correctly joined R-002 to I-004 and its Nginx fix. |
| multi-007 | .5 / .5 / .5 | .5 / .5 / .5 | Both gave R-003 and its mitigation but omitted I-001 and added irrelevant R-006 detail. |
| multi-008 | .5 / .5 / .5 | .5 / .5 / .5 | Both gave the R-006 mitigation but omitted the evaluation plan's exact-value/no-rounding requirement. |
| multi-009 | 0 / 1 / 0 | 0 / 0 / 0 | One Qwen run stated the complete local-document/hosted-excerpt boundary. |
| multi-010 | 0 / 0 / 0 | 0 / 0 / 0 | Both abstained despite the runbook and R-001 evidence. |
| multi-011 | .5 / .5 / .5 | 0 / 0 / 0 | Qwen gave the hybrid-search improvement but did not explicitly explain the pure-vector limitation. |
| multi-012 | .25 / .5 / .5 | 0 / 0 / 0 | Qwen partially covered the version progression but omitted POST `/retrieve`; Llama abstained. |
| multi-013 | 1 / 1 / 1 | 0 / 0 / 0 | Qwen gave both health checks and the diagnostic 404 condition. |
| multi-014 | 1 / 1 / 1 | 0 / 0 / 0 | Qwen correctly connected FR-005 with risk R-004; Llama failed revalidation. |
| multi-015 | .8 / .8 / .8 | 0 / 0 / 0 | Qwen's cited fallback covered retrieval, faithfulness, citation support, and refusal but did not explicitly state correctness. |

## Aggregate result

| Measure | Qwen3 8B | Llama 3.2 3B | Qwen - Llama |
| --- | ---: | ---: | ---: |
| Direct-reviewed answer correctness | 50.9% | 15.0% | +35.9 pp |
| Normal `answered` completion rate | 66.7% | 20.0% | +46.7 pp |
| Non-abstaining response rate, including deadline fallback | 84.4% | 33.3% | +51.1 pp |
| Correctness conditional on those responses | 60.3% | 45.0% | +15.3 pp |
| Context precision | 85.7% | 84.1% | +1.6 pp |
| Context recall | 96.7% | 93.3% | +3.3 pp |
| Mean latency | 24.96 s | 23.69 s | +1.27 s (+5.4%) |
| P95 latency | 29.96 s | 29.98 s | -0.02 s |
| Mean recorded tokens | 1,123.2 | 1,519.4 | -396.2 (-26.1%) |
| Corrective trigger rate | 13.3% | 80.0% | -66.7 pp |
| Corrective recovery rate | 0.0% | 0.0% | 0.0 pp |
| Generation errors | 0 | 0 | 0 |

The fixed Qwen score is 14.5 percentage points above its pre-fix 36.4% multi-hop
score. All three targeted failure classes also cleared in the one-repeat diagnostic
`eval-raw-20260813T152705Z.json`: the variable-length version quote was accepted,
the lower-ranked matching incident I-004 was selected, and exact R-004 evidence was
used instead of NFR-004.

This establishes that the measured Qwen regression was primarily an integration
regression. It does not show that the workflow is finished: 50.9% multi-hop
correctness remains below a reasonable release target, and neither model completed
a corrective recovery. On the current fixed stack, however, Qwen is the better
operational choice. Llama's smaller model did not provide a meaningful latency
advantage because failed coverage and corrective calls consumed the deadline.
