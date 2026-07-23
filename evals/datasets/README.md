# Evaluation datasets

`golden.jsonl` contains exactly 60 fixed questions: 20 single-hop, 15 multi-hop, 15 spreadsheet lookups, and 10 unanswerable cases. `adversarial.jsonl` contains 20 fixed false-premise, out-of-scope, and retrieved-context prompt-injection cases.

Do not regenerate these files during evaluation. Update an expected answer or source only when the underlying corpus changes, and review that change like code.
