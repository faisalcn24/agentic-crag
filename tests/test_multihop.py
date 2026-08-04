from functions.multihop import (
    answer_bounded_multihop_fact,
    deterministic_decomposition,
    fallback_decomposition,
    is_multi_hop_question,
    validate_decomposition,
)


def test_multi_hop_router_recognizes_comparison_combination_and_cross_document():
    questions = [
        "Contrast the recommended local Windows and EC2 storage locations.",
        "Connect risk R-002 with the matching recorded incident and fix.",
        "What evidence across the corpus supports adding swap?",
        "How do FR-005 and risk R-004 describe the same trust control?",
        "Why might exact IDs be difficult, and what improvement is suggested?",
        "How did source visibility change from version 2.2 to version 2.4?",
        "How does FR-006 support the evaluation plan's retrieval metric?",
        "How do the risk register and evaluation plan address omitted rows?",
    ]

    assert all(is_multi_hop_question(question) for question in questions)
    assert not is_multi_hop_question("What is the canonical EC2 storage path?")
    assert not is_multi_hop_question("What was the Q1 budget and actual spend?")


def test_decomposition_validation_accepts_two_to_four_unique_questions():
    original = "Contrast Windows and EC2 storage."

    assert validate_decomposition(
        {"subquestions": ["What is the Windows path?", "What is the EC2 path?"]},
        original,
    ) == ["What is the Windows path?", "What is the EC2 path?"]
    assert validate_decomposition({"subquestions": ["Only one?"]}, original) is None
    assert validate_decomposition(
        {"subquestions": ["Same question?", "Same question?"]}, original
    ) is None
    assert validate_decomposition(
        {"subquestions": ["One?", "Two?", "Three?", "Four?", "Five?"]},
        original,
    ) is None


def test_decomposition_drops_redundant_synthesis_question():
    value = validate_decomposition(
        {
            "subquestions": [
                "What is the Windows storage path?",
                "What is the EC2 storage path?",
                "What are the differences between the Windows and EC2 paths?",
            ]
        },
        "Contrast Windows and EC2 storage paths.",
    )

    assert value == [
        "What is the Windows storage path?",
        "What is the EC2 storage path?",
    ]


def test_fallback_decomposition_is_bounded_and_preserves_identifiers():
    subquestions = fallback_decomposition(
        "How do FR-005 and risk R-004 describe the same trust control?"
    )

    assert 2 <= len(subquestions) <= 4
    assert any("FR-005" in question for question in subquestions)
    assert any("R-004" in question for question in subquestions)


def test_fallback_decomposition_expands_small_version_range():
    subquestions = fallback_decomposition(
        "Summarize the feature progression from versions 2.2 through 2.4."
    )

    assert subquestions == [
        "What changed in version 2.2?",
        "What changed in version 2.3?",
        "What changed in version 2.4?",
    ]


def test_deterministic_decomposition_targets_required_facts():
    cases = {
        "How does FR-006 support the evaluation plan's retrieval metric?": (
            "FR-006",
            "metric",
        ),
        "Connect risk R-002 with the matching recorded incident and fix.": (
            "R-002",
            "incident",
        ),
        "What privacy boundary is created by local embeddings plus hosted answer generation?": (
            "stays local",
            "sent to",
        ),
        "Which health checks distinguish an app failure from an Nginx routing failure?": (
            "direct app",
            "Nginx-routed",
        ),
    }

    for question, required_phrases in cases.items():
        subquestions = deterministic_decomposition(question)
        assert 2 <= len(subquestions) <= 4
        joined = " ".join(subquestions)
        assert all(phrase.casefold() in joined.casefold() for phrase in required_phrases)


def test_public_deployment_decomposition_asks_for_the_two_actual_risks():
    assert deterministic_decomposition(
        "What two document-based reasons make a long-lived public deployment unsafe as currently specified?"
    ) == [
        "What does the product requirements document state about authentication in the current release?",
        "What document data can hybrid hosted answer generation send to an external provider?",
    ]


def test_decomposition_rejects_non_question_control_fragments():
    assert validate_decomposition(
        {
            "subquestions": [
                '{ "type": "entity", "value": "Nginx" }',
                '{ "type": "function", "name": "port_range" }',
            ]
        },
        "How do Nginx and FastAPI divide ports?",
    ) is None


def test_bounded_answer_connects_risk_to_matching_incident():
    sources = [
        {
            "filename": "risks.xlsx-Risk Register",
            "type": "xlsx",
            "text": (
                "Risk_ID: R-003 | Risk: Wrong public IP in SSH security group | "
                "Probability: 0.4 | Impact: High | Owner: Deployment | "
                "Mitigation: Update inbound SSH source to current public IPv4 /32 | "
                "Status: Mitigated"
            ),
        },
        {
            "filename": "risks.xlsx-Incident Log",
            "type": "xlsx",
            "text": (
                "Incident_ID: I-001 | Symptom: SSH connection timed out | "
                "Root_Cause: Security group allowed old public IP address | "
                "Fix: Updated SSH inbound source to current public IPv4 /32\n"
                "Incident_ID: I-004 | Symptom: Nginx 404 | "
                "Root_Cause: Inactive site | Fix: Reloaded Nginx"
            ),
        },
    ]

    answer = answer_bounded_multihop_fact(
        "Connect risk R-003 with the incident that demonstrated it.", sources
    )

    assert "Wrong public IP" in answer
    assert "I-001" in answer
    assert "old public IP" in answer
    assert "I-004" not in answer
    assert "[risks.xlsx-Risk Register]" in answer
    assert "[risks.xlsx-Incident Log]" in answer


def test_bounded_answer_connects_requirement_to_recall_metric():
    sources = [
        {
            "filename": "requirements.docx",
            "type": "docx",
            "text": "FR-006: The system exposes a retrieval-only debugging endpoint.",
        },
        {
            "filename": "evaluation.docx",
            "type": "docx",
            "text": (
                "Recall@5 measures whether the expected source appears in the top five "
                "retrieved chunks."
            ),
        },
    ]

    answer = answer_bounded_multihop_fact(
        "How does FR-006 support the evaluation plan's retrieval metric?", sources
    )

    assert "retrieval-only" in answer
    assert "Recall@5" in answer
    assert "top five" in answer


def test_bounded_answer_covers_public_deployment_privacy_and_access():
    sources = [
        {
            "filename": "requirements.docx",
            "type": "docx",
            "text": (
                "The current release does not include authentication. Retrieved excerpts "
                "are sent to an external model provider."
            ),
        },
        {
            "filename": "runbook.docx",
            "type": "docx",
            "text": (
                "For a longer-lived deployment, add HTTPS, a domain name, and authentication."
            ),
        },
    ]

    answer = answer_bounded_multihop_fact(
        "What two document-based reasons make a long-lived public deployment unsafe?", sources
    )

    assert "does not include authentication" in answer
    assert "external provider" in answer
    assert "add HTTPS" in answer


def test_bounded_answer_connects_source_snippet_trust_controls():
    sources = [
        {
            "filename": "requirements.docx",
            "type": "docx",
            "text": "FR-005: The system returns source snippets with each answer.",
        },
        {
            "filename": "risks.xlsx-Risk Register",
            "type": "xlsx",
            "text": (
                "Risk_ID: R-004 | Risk: Missing source citations reduce trust | "
                "Mitigation: Return source snippets and retrieval scores | Status: Mitigated"
            ),
        },
    ]

    answer = answer_bounded_multihop_fact(
        "How do FR-005 and risk R-004 describe the same trust control?", sources
    )

    assert "FR-005" in answer
    assert "R-004" in answer
    assert "source snippets" in answer
    assert "retrieval scores" in answer


def test_bounded_answer_contrasts_windows_and_ec2_storage():
    sources = [
        {
            "filename": "releases.docx",
            "type": "docx",
            "text": "Windows development uses .agentic_crag_data.",
        },
        {
            "filename": "runbook.docx",
            "type": "docx",
            "text": "The canonical EC2 storage path is /opt/agentic-crag/data.",
        },
    ]

    answer = answer_bounded_multihop_fact(
        "Contrast the recommended local Windows and EC2 storage locations.", sources
    )

    assert ".agentic_crag_data [releases.docx]" in answer
    assert "/opt/agentic-crag/data [runbook.docx]" in answer


def test_bounded_answer_preserves_each_supported_port_and_exposure_leg():
    sources = [
        {
            "filename": "runbook.docx",
            "type": "docx",
            "text": (
                "FastAPI listens on 127.0.0.1:8000 and Streamlit listens on "
                "127.0.0.1:8501. Nginx listens publicly on port 80. The EC2 instance "
                "should not expose FastAPI port 8000 or Streamlit port 8501 directly "
                "to the public internet."
            ),
        }
    ]

    answer = answer_bounded_multihop_fact(
        "How do Nginx, FastAPI, and Streamlit divide the demo's ports and exposure?",
        sources,
    )

    assert "Nginx listens publicly on port 80" in answer
    assert "FastAPI listens on 127.0.0.1:8000" in answer
    assert "Streamlit listens on 127.0.0.1:8501" in answer
    assert answer.count("[runbook.docx]") == 2


def test_bounded_answer_preserves_each_supported_version_comparison_leg():
    sources = [
        {
            "filename": "releases.docx",
            "type": "docx",
            "text": (
                "Version 2.2 used simple chat responses that returned only answer text. "
                "Source snippets were not yet exposed to the UI. "
                "Version 2.4 added structured source reporting to chat responses. "
                "The chat response now includes an answer field and a sources list. "
                "Version 2.4 updated Streamlit to display source snippets in a Sources "
                "expander under assistant responses."
            ),
        }
    ]

    answer = answer_bounded_multihop_fact(
        "How did source visibility change from version 2.2 to version 2.4?",
        sources,
    )

    assert "Version 2.2" in answer
    assert "only answer text" in answer
    assert "not yet exposed" in answer
    assert "Version 2.4" in answer
    assert "sources list" in answer
    assert "display source snippets" in answer


def test_bounded_answer_preserves_each_supported_privacy_boundary_leg():
    sources = [
        {
            "filename": "requirements.docx",
            "type": "docx",
            "text": (
                "FR-004 uses local embeddings to avoid sending full documents to the "
                "answer generation model. AWS hybrid mode sends retrieved excerpts to "
                "Groq for final answer generation."
            ),
        }
    ]

    answer = answer_bounded_multihop_fact(
        "What privacy boundary is created by local embeddings plus hosted answer generation?",
        sources,
    )

    assert "local embeddings" in answer
    assert "full documents" in answer
    assert "retrieved excerpts" in answer
    assert "Groq" in answer


def test_decomposed_evidence_answer_requires_support_for_every_leg():
    sources = [
        {
            "filename": "requirements.docx",
            "type": "docx",
            "text": (
                "FR-004 uses local embeddings to avoid sending full documents to the "
                "answer generation model."
            ),
        }
    ]

    answer = answer_bounded_multihop_fact(
        "What privacy boundary is created by local embeddings plus hosted answer generation?",
        sources,
    )

    assert answer is None
