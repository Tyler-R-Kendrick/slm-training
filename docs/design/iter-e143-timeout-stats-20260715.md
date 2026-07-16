# E143 — Preserve timeout decode telemetry (2026-07-15)

## Question

Do diagnostic timeouts lose the constrained-decoder evidence needed to guide the next iteration?

## Change

`generate_with_stats()` now attaches its live `DecodeStats` collector to an exception before re-raising. The evaluator consumes that attached collector when converting a `TimeoutError` into a failed prediction. Normal successful generation is unchanged.

## Evidence

The normalized one-record smoke replay used the E135 HF-context checkpoint with the E141 policy: CPU, local-only HF context, LTR primary and repair, constrained slot decoding, exact-stream probe skipped, three attempts, and a 20-second timeout.

| Metric | Result |
| --- | ---: |
| n | 1 |
| parse rate | 0.0 |
| structural similarity | 0.0 |
| placeholder validity | 0.0 |
| timeout count | 1 |
| emitted tokens captured | 304 |
| DFA syncs captured | 6,724 |
| probes captured | 1 |
| total decode time | 20,001.5 ms |
| bounded selection events | 64 |

The first captured divergence remains at position 5 after `root=CheckBoxItem(`, where the repair path selects a malformed multi-line token while the model argmax is `=`. The run is therefore a truthful quality failure, but its partial decoder telemetry is now durable in `eval_smoke.json` and the AgentEvals bundle.

## Next iteration

Fix the selection telemetry's stale `legal_candidates=0` value on early picker return paths, then use the corrected trace to test the singleton-token bypass hypothesis. Do not claim a structural-adherence improvement until a broader suite replay passes the parse and ship gates.
