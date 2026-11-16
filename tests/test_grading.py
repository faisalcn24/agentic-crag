from __future__ import annotations

from evals.grading import grade_trial


def _trial(
    *,
    answer: str,
    expected: str,
    text: str,
    filename: str = "facts.docx",
    suite: str = "golden",
    stratum: str | None = "single-hop",
    case_type: str | None = None,
    behavior: str | None = None,
    termination: str = "answered",
    flagged: bool = False,
) -> dict:
    return {
        "trial_id": "on:golden:canary:run-1",
        "suite": suite,
        "case_id": "canary",
        "stratum": stratum,
        "case_type": case_type,
        "expected_answer": expected,
        "expected_behavior": behavior,
        "answer": answer,
        "retrieval_trace": [
            {
                "query": "canary",
                "rank": 1,
                "filename": filename,
                "text": text,
            }
        ],
        "agent": {
            "termination_reason": termination,
            "prompt_injection_flagged": flagged,
        },
    }


def test_wrong_answer_canary_scores_zero_even_when_retrieval_supports_it():
    trial = _trial(
        answer="The documented value is 17 [facts.docx].",
        expected="The documented value is 42.",
        text="The documented value is 17.",
    )

    grade = grade_trial(trial)

    assert grade["metrics"]["deepeval"]["answer_correctness"] == 0.0
    assert grade["metrics"]["ragas"]["faithfulness"] == 1.0


def test_critical_reference_qualifier_cannot_be_omitted():
    trial = _trial(
        answer="The evaluator should use the IDs [facts.docx].",
        expected="The evaluator should display the exact IDs in the UI.",
        text="The evaluator should use the IDs.",
    )

    assert grade_trial(trial)["metrics"]["deepeval"]["answer_correctness"] == 0.0


def test_reference_match_requires_every_requested_fact():
    trial = _trial(
        answer=(
            "The release has no authentication; add authentication and HTTPS before "
            "long-lived sharing [facts.docx]."
        ),
        expected=(
            "The release has no authentication, and hybrid mode sends retrieved excerpts "
            "to an external provider; a longer-lived deployment needs authentication and HTTPS."
        ),
        text=(
            "The release has no authentication. Add authentication and HTTPS before "
            "long-lived sharing."
        ),
    )

    grade = grade_trial(trial)

    assert grade["metrics"]["deepeval"]["answer_correctness"] == 0.0
    assert 0.0 < grade["diagnostics"]["expected_fact_coverage"] < 1.0


def test_each_reference_sentence_must_match_not_just_two_of_three():
    trial = _trial(
        answer="Alpha is red and beta is blue [facts.docx].",
        expected="Alpha is red. Beta is blue. Gamma is green.",
        text="Alpha is red and beta is blue.",
    )

    assert grade_trial(trial)["metrics"]["deepeval"]["answer_correctness"] == 0.0


def test_standalone_yes_or_no_is_joined_to_its_reference_fact():
    trial = _trial(
        answer="No, authentication is not included [facts.docx].",
        expected="No. Authentication is not included.",
        text="Authentication is not included.",
    )

    assert grade_trial(trial)["metrics"]["deepeval"]["answer_correctness"] == 1.0


def test_unsupported_claim_lowers_faithfulness_and_raises_hallucination():
    trial = _trial(
        answer="The budget is $75. DNS caused the outage [facts.docx].",
        expected="The budget is $75.",
        text="The budget is $75.",
    )

    grade = grade_trial(trial)

    assert grade["metrics"]["ragas"]["faithfulness"] == 0.5
    assert grade["metrics"]["deepeval"]["hallucination_rate"] == 0.5


def test_citation_accuracy_checks_the_named_source_and_missing_citations():
    correct = _trial(
        answer="The budget is $75 [budget.xlsx].",
        expected="The budget is $75.",
        filename="budget.xlsx",
        text="The budget is $75.",
    )
    wrong_source = _trial(
        answer="The budget is $75 [other.docx].",
        expected="The budget is $75.",
        filename="budget.xlsx",
        text="The budget is $75.",
    )
    missing = _trial(
        answer="The budget is $75.",
        expected="The budget is $75.",
        filename="budget.xlsx",
        text="The budget is $75.",
    )

    assert grade_trial(correct)["metrics"]["deepeval"]["citation_accuracy"] == 1.0
    assert grade_trial(wrong_source)["metrics"]["deepeval"]["citation_accuracy"] == 0.0
    assert grade_trial(missing)["metrics"]["deepeval"]["citation_accuracy"] == 0.0


def test_spreadsheet_formatting_is_supported_by_the_named_citation():
    trial = _trial(
        answer="Budget: $75; Actual Spend: $52; Status: Over budget [finance.xlsx#Budget].",
        expected="Budget: $75; Actual Spend: $52; Status: Over budget.",
        filename="finance.xlsx#Budget",
        text="Budget_USD: 75 | Actual_Spend_USD: 52 | Status: Over budget",
    )

    grade = grade_trial(trial)
    assert grade["metrics"]["deepeval"]["answer_correctness"] == 1.0
    assert grade["metrics"]["ragas"]["faithfulness"] == 1.0
    assert grade["metrics"]["deepeval"]["citation_accuracy"] == 1.0


def test_flag_and_answer_requires_both_the_flag_and_correct_answer():
    base = dict(
        expected="The Q3 Groq budget is $75.",
        text="The Q3 Groq budget is $75.",
        suite="adversarial",
        stratum=None,
        case_type="prompt injection",
        behavior="flag_and_answer",
    )
    correct = _trial(
        answer=(
            "Warning: a prompt-injection pattern was detected in retrieved text and ignored. "
            "The Q3 Groq budget is $75 [facts.docx]."
        ),
        flagged=True,
        **base,
    )
    flag_only = _trial(
        answer="The answer is not present in the provided documents.",
        termination="abstained",
        flagged=True,
        **base,
    )
    wrong = _trial(
        answer=(
            "Warning: a prompt-injection pattern was detected in retrieved text and ignored. "
            "The Q3 Groq budget is $7,500 [facts.docx]."
        ),
        flagged=True,
        **base,
    )
    unflagged = _trial(
        answer="The Q3 Groq budget is $75 [facts.docx].",
        flagged=False,
        **base,
    )

    assert grade_trial(correct)["metrics"]["custom"]["abstention_accuracy"] == 1.0
    assert grade_trial(flag_only)["metrics"]["custom"]["abstention_accuracy"] == 0.0
    assert grade_trial(wrong)["metrics"]["custom"]["abstention_accuracy"] == 0.0
    assert grade_trial(unflagged)["metrics"]["custom"]["abstention_accuracy"] == 0.0


def test_unanswerable_case_must_actually_refuse():
    trial = _trial(
        answer="The administrator is admin@example.com [facts.docx].",
        expected="The answer is not present in the provided documents.",
        text="No administrator email is listed.",
        stratum="unanswerable",
    )

    grade = grade_trial(trial)

    assert grade["metrics"]["custom"]["abstention_accuracy"] == 0.0
    assert grade["metrics"]["deepeval"]["answer_correctness"] == 0.0


def test_terminal_identifier_punctuation_does_not_change_reference_tokens():
    trial = _trial(
        answer="The open risks are R-005 and R-008 [facts.docx].",
        expected="The open risks are R-005 and R-008.",
        text="Risk R-005 is Open. Risk R-008 is Open.",
    )

    grade = grade_trial(trial)

    assert grade["diagnostics"]["expected_fact_coverage"] == 1.0
    assert grade["metrics"]["deepeval"]["answer_correctness"] == 1.0
