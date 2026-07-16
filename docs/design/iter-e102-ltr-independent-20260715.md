# E102 independent LTR objective (2026-07-15)

E102 disabled fused LTR masking (`--no-fuse-ltr`) while holding the visible
contract corpus and model recipe constant. The run completed 128 CPU steps
with loss `8.42000`, 43,146 target tokens, and persisted training telemetry.

Strict smoke evaluation remained invalid: parse/raw syntax `0.0`, structural
similarity `0.15`, component recall `0.0`, contract precision/recall
`1.0/1.0`, placeholder fidelity `1.0`, and latency `7284.73 ms`. AgentV had
no execution errors but all five checks failed.

Decision: reject independent LTR. Fused masking is not the primary cause of
the repair failure; return focus to repair-state data and decode-state
alignment.
