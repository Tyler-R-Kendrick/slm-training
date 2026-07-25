# SLM-343 / CAP1-GEN-01: CAP1 gate closeout (slm343_gate_closeout)

Matrix set: `cap1-gen-01-grounding-contract` · Version: `slm343-v1` · Status: **closeout**
Decision: **not_authorized** — no production code.

This issue is blocked on `CERT_CAP0` evidence from SLM-362. The matched CAP0
curriculum experiment ran on 2026-07-25 and **rejected** CERT_CAP0 on four
stop rules (`prediction_identical`, `copy_dominated`,
`structurally_regressive`, `underpowered`); the DSH1-09 gate reports
`insufficient_n` for every arm ([results](iter-slm362-cap0-experiment-20260725.md)).

Per the stop rule, CAP1/CAP2 training stays closed: the generic NL↔DSL
grounding contract and capability certificate must not be implemented.
SLM-344 and the DSH2 line stay closed until a powered CAP0 rerun issues
CERT_CAP0.
