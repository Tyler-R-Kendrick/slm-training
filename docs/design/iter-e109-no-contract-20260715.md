# E109 no-contract conditioning control (2026-07-15)

E109 trained the same visible-contract corpus and base recipe without slot
contract context or constrained slot decoding. This isolates whether contract
conditioning caused the Stack-list corruption.

Training completed 128 CPU steps at loss `7.63404`, with 43,146 target tokens
and persisted telemetry. Strict smoke evaluation remained invalid: parse/raw
syntax `0.0/0.0`, structural similarity `0.3417`, contract precision/recall
`0.0/0.0`, placeholder fidelity `0.0`, component recall `0.0`, and latency
`11804.24 ms`. AgentV remained non-ship with 5 failed checks.

Decision: reject. Contract conditioning is necessary for content fidelity but
does not explain the structural failure; retain it for subsequent runs.
