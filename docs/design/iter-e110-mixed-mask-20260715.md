# E110 mixed-mask recipe control (2026-07-15)

E110 tested the lineage-recommended `mask_pattern=mixed` against the CLI
default `random`, holding the 1,417-record visible-contract corpus and model
recipe constant. Training completed 128 CPU steps at loss `6.61512`, with
43,146 target tokens and persisted telemetry.

Strict smoke evaluation remained invalid: parse/raw syntax `0.0/0.0`,
structural similarity `0.2333`, contract precision/recall `1.0/0.5`,
placeholder fidelity `0.5`, component recall `0.25`, and latency `13785.03 ms`.
AgentV failed all five checks.

Decision: reject. The mask-pattern mismatch is not the primary cause of the
Stack-list failure; retain the explicit recipe choice in future experiment
metadata but do not promote this checkpoint.
