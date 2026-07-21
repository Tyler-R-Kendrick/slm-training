# E694 — open-vocabulary enum finalization

Date: 2026-07-21
Status: completed positive; retained; not ship

E694 extends the retained post-decode enum finalizer to enum values without a
single fixed vocabulary token. It substitutes the first public-schema enum as
a framed literal only when the generated row has capacity; otherwise it keeps
the original row. The replacement is replayed through a cloned grammar state,
so invalid substitutions fail closed and later choices remain unchanged. The
focused invariant and all 134 compiler-decode tests passed.

The independently capped full Held-out replay completed with exit 0, no timeout
or fallback, and emitted AgentEvals JSONL plus an AgentV SDK bundle.

| Held-out `n=5` | E693 v149 | E694 v150 |
| --- | ---: | ---: |
| syntax / meaningful v1 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| strict v2 / coverage | 0.6000 / 1.0000 | 0.8000 / 1.0000 |
| fidelity / validity | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| structure / component recall | 0.6624 / 0.7933 | 0.6624 / 0.7933 |
| reward | 0.9634 | 0.9634 |
| AST node / edge F1 | 0.7754 / 0.5901 | 0.7754 / 0.5901 |
| latency p50 / p95 | 3864.72 / 6155.73 ms | 3414.50 / 5951.17 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

Only settings changes: Slider variant `"tet"` becomes `"continuous"`. The
Slider, SwitchGroup subtree, root references, and every later choice remain
byte-identical. Settings now passes strict v2 with no reason codes; the other
four predictions are byte-identical. All continuous quality metrics are flat.
The latency movement on this tiny replay is not a performance claim.

Retain v150 as a generalized schema-validity correction. This is one reused
scratch checkpoint and Held-out `n=5`, not a powered result or ship claim. No
checkpoint was created, synced, or promoted. Form remains the sole strict
failure (`placeholder_semantic_role_mismatch`, `placeholder_spam`).

Evidence: [JSON](iter-e694-open-enum-finalize-20260721.json).
