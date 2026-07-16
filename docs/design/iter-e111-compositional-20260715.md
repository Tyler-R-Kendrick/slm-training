# E111 compositional-tokenizer control (2026-07-15)

E111 retrained the base visible-contract recipe with the compositional output
tokenizer instead of lexer-native output. The run completed 128 CPU steps at
loss `8.28341`, saw 63,599 target tokens, and persisted training telemetry.

Strict smoke evaluation remained invalid: parse/raw syntax `0.0/0.0`,
structural similarity `0.1917`, contract precision/recall `0.0/0.0`,
placeholder fidelity `0.0`, component recall `0.0`, and latency `8119.06 ms`.
AgentV failed all five checks.

Decision: reject. Lexer output is not the sole source of the structural
failure, and it remains preferable because it preserves contract fidelity.
