# E144 — Correct constrained candidate-count telemetry (2026-07-15)

## Question

Can the selection trace tell us whether a malformed choice came from a true singleton legal set, or from an early picker return whose full legal set was never counted?

## Change

The picker now resets `constrained_last_legal_candidates` at the start of every pick. A proven singleton accept set reports `1`; early paths that do not enumerate the full set report `-1`; a completed search with no legal candidates reports `0`. The trace serializer now preserves `-1` instead of clamping it to `0`.

## Evidence

The normalized one-record smoke replay used the E135 HF-context checkpoint and E141 constrained-decoding policy on CPU. It produced 64 bounded selection events with parse rate `0.0`, structural similarity `0.0538`, and no timeout. The first malformed decision is unchanged: after `root=CheckBoxItem(`, the repair path chooses `")\ncta = Button("` while the model argmax is `=`.

Observed `legal_candidates` values in the trace were `-1` and `0`; no captured choice was a proven singleton (`1`). Therefore the current evidence does not justify bypassing probing on the malformed choice. The next experiment needs explicit singleton-event coverage or a broader trace before changing selection semantics.

The AgentEvals JSONL and full evaluator output are persisted under `outputs/runs/iter-e144-candidate-count-20260715/e144-candidate-count-one-rerun/`.
