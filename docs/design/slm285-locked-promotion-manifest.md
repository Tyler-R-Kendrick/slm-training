# SLM-285 locked promotion manifest

`abstract_planning_locked_v1.jsonl` is the immutable local promotion-holdout
manifest. Its content digest is
`b4ad49cf1b73ad50528709daaad53dbf4846036c9dea787f1c2017c16e0a2d48`.

The local build used 1,000 sanitized RICO-cache candidates and every committed
training snapshot (6,337 records, excluding duplicate governance copies). The
all-view audit retained 286 records: 226 `locked_test`, and 20 each for
`dev`, `agentv_calibration`, and `human_audit`. The human-audit partition is
immutable but optional; no human score or rating is a promotion gate.

Hard admission views are exact/normalized leakage, canonical AST, canonical
semantic plan, and lexical prompt-similarity (`0.95`). The manifest source and
candidate digests, verifier configuration, complexity strata, and partition
assignment are content-addressed in the file. Promotion/ship campaign claims
must carry this manifest digest, and promoted-anchor metadata persists it.

This is corpus wiring and decontamination evidence, not a model-quality or
ship claim. `legal_action_entropy` is deliberately recorded as requiring a
decode trace; no model-selection or promotion result was run here.
