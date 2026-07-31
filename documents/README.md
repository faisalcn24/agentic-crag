# RAG Test Corpus

This folder contains synthetic public/demo documents for testing Document Analysis RAG. The files are intentionally overlapping and moderately long so retrieval, citations, numeric lookup, comparison, and abstention behavior can be tested.

The corpus describes a historical 2.4 demo release. Its statements about OCR,
reranking, and hybrid retrieval are evidence fixtures, not documentation of the
current application runtime.

## Files

- `01_product_requirements.docx`
  - Tests product facts, requirements lookup, scope boundaries, and absent-feature questions.

- `02_aws_deployment_runbook.docx`
  - Tests deployment facts, exact paths, ports, service names, and troubleshooting retrieval.

- `03_rag_evaluation_plan.docx`
  - Tests metric definitions, expected-source retrieval, and answer-not-present behavior.

- `04_release_notes.docx`
  - Tests version comparison, overlapping facts, and release-specific details.

- `05_budget_and_benchmarks.xlsx`
  - Tests spreadsheet retrieval, exact numeric values, budget questions, and benchmark lookup.

- `06_risk_register_and_incidents.xlsx`
  - Tests tabular risk lookup, highest/lowest value questions, mitigation facts, and incident root-cause questions.

## Suggested Questions

- What does FR-006 require?
- What path stores data on EC2?
- Which port should be public for the deployed app?
- What is the Q3 Groq API budget?
- Which risk has the highest probability score?
- Why did `/api/health` return a 404 during deployment?
- What changed between version 2.2 and 2.4?
- Is authentication included in the current release?
- What is the company payroll for 2026?

For the last question, the correct behavior is to say that the answer is not present in the documents.
