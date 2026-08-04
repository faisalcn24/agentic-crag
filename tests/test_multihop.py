from functions.multihop import (
    decomposition_prompt,
    fallback_decomposition,
    is_multi_hop_question,
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
    assert not is_multi_hop_question("What does AB-123 require?")


def test_decomposition_prompt_preserves_question():
    question = "Compare policy A with policy B."
    prompt = decomposition_prompt(question)

    assert question in prompt
    assert "two to four" in prompt
    assert "Do not answer" in prompt


def test_validation_accepts_two_to_four_unique_atomic_questions():
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


def test_fallback_splits_an_explicit_comparison():
    result = fallback_decomposition("Compare local storage and hosted storage.")

    assert result == [
        "What do the documents say about local storage?",
        "What do the documents say about hosted storage?",
    ]
