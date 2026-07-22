# E741 — lexer slot-coverage close

**Date:** 2026-07-22  
**Decision:** retain generalized close behavior; reject checkpoint promotion  
**Evidence:** [`iter-e741-lexer-slot-coverage-close-20260722.json`](iter-e741-lexer-slot-coverage-close-20260722.json)

E741 extends the existing `slot_coverage_close_decode_weight` lever to lexer
compiler-tree and restricted decoding. Once every declared template symbol has
appeared, the model raises the immediately legal closing path above the current
candidate maximum. It abstains before coverage, adds no new lever, and emits no
free-form text.

The accepted arms reuse the unchanged local E735 checkpoint and three frozen
smoke records. They ran locally on CPU with `strict_compiler_tree`, honest slot
contracts, visible semantic roles, semantic-role and schema-role weights 2,
semantic-plan weight 4 / margin 2, a 160-symbol canvas, an eight-second
per-record guard, and the two-minute command cap. The complete 235-field
effective model configurations are persisted in both eval artifacts and differ
only in coverage-close weight 0 versus 2.

| Arm | Close apps / changes | Tokens | Parse | Meaning-v1 | Strict-v2 | Fidelity | Validity | Structure | Recall | Reward | p50 / p95 | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v207 close 0 | 0 / 0 | 174 | 1.0000 | 0.6667 | 0.0000 | 1.0000 | 1.0000 | 0.1653 | 0.4167 | 0.9370 | 5242 / 5757 ms | 0/1 |
| v207 close 2 | 3 / 3 | 61 | 1.0000 | 0.6667 | 0.0000 | 1.0000 | 1.0000 | 0.5464 | 0.4167 | 0.9370 | 1749 / 2715 ms | 0/1 |

Coverage-close fires on all three records. It removes the repeated nested tail,
cuts emitted tokens by 64.9%, reduces median decode latency by 66.6%, and raises
structural similarity by 0.3811. Parse, meaning-v1, fidelity, validity,
component recall, reward, strict-v2, and AgentV do not regress. This is a useful
bounded decoder correction, not a ship result: strict-v2 remains zero, AgentV
remains 0/1, and one smoke record still fails component recall.

Retain the generalized default-off implementation and its shared telemetry.
Reject checkpoint promotion and ship claims; no checkpoint was created, synced,
or promoted. The next intervention should target the remaining declared-family
recall failure without reopening unbounded container generation.

## Harness provenance remediation

An exact-policy replay of the historical E740 artifact at its parent v206
commit produced the same long predictions and metrics as the v207 default-off
control, rather than the shorter predictions reported by E740. This proves the
older artifact did not persist enough of the effective model configuration to
reconstruct its recipe; it is retained as historical diagnostic evidence but
must not be treated as a reproducible baseline.

Eval harness v42 fixes the cause rather than filtering the mismatch from the
dashboard. Every `evaluation_policy` now recursively serializes the complete
effective model configuration, including defaulted and dynamically supplied
fields. Cache identity already hashes `evaluation_policy`, so omitted settings
can no longer alias distinct runs. Regression tests verify both complete
serialization and the real policy shape. The parent v206 replay and v207
default-off control match in predictions and aggregate metrics, also proving
that the v207 default-off model change itself is neutral.
