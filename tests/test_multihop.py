from functions.multihop import (
    decomposition_prompt,
    decomposition_is_useful,
    deterministic_decomposition,
    fallback_decomposition,
    is_multi_hop_question,
    requires_cross_leg_join,
    sentence_decomposition,
    validate_decomposition,
)


def test_router_recognizes_general_multi_part_questions():
    assert is_multi_hop_question(
        "Compare the onboarding policy with the support policy."
    )
    assert is_multi_hop_question("How do AB-123 and CD-456 relate?")
    assert is_multi_hop_question(
        "What does the first source require, and how does the second source implement it?"
    )
    assert is_multi_hop_question(
        "How does control AB-123 support the evaluation plan's retrieval metric?"
    )
    assert is_multi_hop_question(
        "How do the risk register and evaluation plan address omitted rows?"
    )
    assert is_multi_hop_question(
        "What evidence across the corpus supports adding swap on a small instance?"
    )
    assert is_multi_hop_question(
        "Why might exact identifiers be difficult, and what improvement is suggested?"
    )
    assert not is_multi_hop_question("What does AB-123 require?")
    assert not is_multi_hop_question(
        "For incident OCR-417, which service was affected, what caused it, "
        "and how was it resolved?"
    )
    assert requires_cross_leg_join(
        "Connect risk R-002 with the matching recorded incident and fix."
    )
    assert not requires_cross_leg_join(
        "Compare Windows and EC2 storage locations."
    )


def test_decomposition_prompt_preserves_question():
    question = "Compare policy A with policy B."
    prompt = decomposition_prompt(question)

    assert question in prompt
    assert "two to four" in prompt
    assert "identifiers and qualifiers" in prompt
    assert "do not invent or answer" in prompt


def test_validation_accepts_two_to_three_unique_atomic_questions():
    original = "Compare policy A with policy B."
    result = validate_decomposition(
        {
            "subquestions": [
                "What does policy A require?",
                "What does policy B require?",
            ]
        },
        original,
    )

    assert result == ["What does policy A require?", "What does policy B require?"]


def test_validation_rejects_invalid_control_output():
    original = "Compare policy A with policy B."

    assert validate_decomposition({"subquestions": ["Only one?"]}, original) is None
    assert (
        validate_decomposition(
            {"subquestions": ["What does A require?", "What does A require?"]},
            original,
        )
        is None
    )
    assert (
        validate_decomposition(
            {"subquestions": ["What does A require", "What does B require?"]},
            original,
        )
        is None
    )


def test_validation_rejects_output_that_drops_required_identifiers():
    original = "How do FR-005 and R-004 describe the same trust control?"

    assert (
        validate_decomposition(
            {
                "subquestions": [
                    "What is the control name?",
                    "Where is the control documented?",
                ]
            },
            original,
        )
        is None
    )


def test_validation_rejects_non_atomic_questions_that_repeat_every_identifier():
    original = "How do FR-005 and R-004 describe the same trust control?"

    assert (
        validate_decomposition(
            {
                "subquestions": [
                    "What is the name described by FR-005 and R-004?",
                    "Where is the control described by FR-005 and R-004 located?",
                ]
            },
            original,
        )
        is None
    )


def test_fallback_preserves_multiple_identifiers():
    result = fallback_decomposition("How do AB-123 and CD-456 relate?")

    assert len(result) == 2
    assert "AB-123" in result[0]
    assert "CD-456" in result[1]


def test_fallback_expands_a_small_version_range():
    result = fallback_decomposition("What changed from versions 3.1 through 3.3?")

    assert result == [
        "What changed in version 3.1?",
        "What changed in version 3.2?",
        "What changed in version 3.3?",
    ]


def test_fallback_expands_repeated_version_labels_without_topic_special_cases():
    result = fallback_decomposition(
        "How did source visibility change from version 2.2 to version 2.4?"
    )

    assert result == [
        "What source visibility was documented for version 2.2?",
        "What source visibility was documented for version 2.4?",
    ]


def test_fallback_keeps_identifier_questions_corpus_agnostic():
    result = fallback_decomposition(
        "How do FR-005 and risk R-004 describe the same trust control?"
    )

    assert result == [
        "What does FR-005 state that is relevant to the question?",
        "What does R-004 state that is relevant to the question?",
    ]


def test_deterministic_decomposition_is_limited_to_reliable_patterns():
    assert deterministic_decomposition(
        "How did source visibility change from version 2.2 to version 2.4?"
    )
    assert deterministic_decomposition(
        "Connect risk R-002 with the matching recorded incident and fix."
    ) == [
        "What does risk R-002 state?",
        "What evidence describes the matching recorded incident and fix?",
    ]
    assert deterministic_decomposition(
        "Compare local storage and hosted storage."
    ) == [
        "What do the documents say about local storage?",
        "What do the documents say about hosted storage?",
    ]


def test_decomposition_is_used_only_for_reliable_splits():
    assert decomposition_is_useful(
        "Contrast the recommended local Windows and EC2 storage locations."
    )
    assert decomposition_is_useful(
        "Why might exact requirement IDs be difficult, and what improvement is repeatedly suggested?"
    )
    assert not decomposition_is_useful(
        "What two document-based reasons make a long-lived public deployment unsafe?"
    )
    assert not decomposition_is_useful(
        "What evidence across the corpus supports adding swap on a small EC2 instance?"
    )
    assert deterministic_decomposition(
        "Why might exact requirement IDs be difficult, and what improvement is repeatedly suggested?"
    ) == [
        "Why might exact requirement IDs be difficult?",
        "What improvement is repeatedly suggested for exact requirement IDs?",
    ]


def test_deterministic_decomposition_handles_explicit_subject_relations():
    assert deterministic_decomposition(
        "How does FR-006 support the evaluation plan's retrieval metric?"
    ) == [
        "What does FR-006 state?",
        "What does the evaluation plan's retrieval metric measure?",
    ]
    assert deterministic_decomposition(
        "How do the risk register and evaluation plan address omitted spreadsheet rows?"
    ) == [
        "What does the risk register say about omitted spreadsheet rows?",
        "What does evaluation plan say about omitted spreadsheet rows?",
    ]
    assert deterministic_decomposition(
        "How do Nginx, FastAPI, and Streamlit divide the demo's ports and exposure?"
    ) == [
        "What does Nginx say about the demo's ports and exposure?",
        "What does FastAPI say about the demo's ports and exposure?",
        "What does Streamlit say about the demo's ports and exposure?",
    ]


def test_fallback_splits_an_explicit_comparison():
    result = fallback_decomposition("Compare local storage and hosted storage.")

    assert result == [
        "What do the documents say about local storage?",
        "What do the documents say about hosted storage?",
    ]


def test_sentence_decomposition_preserves_explicit_requirement_scopes():
    question = (
        "Compare Windows and EC2 storage locations. "
        "Then state which port is public and which application ports remain private. "
        "Cite each source."
    )

    assert sentence_decomposition(question) == [
        "Compare Windows and EC2 storage locations?",
        "State which port is public?",
        "Which application ports remain private?",
    ]
