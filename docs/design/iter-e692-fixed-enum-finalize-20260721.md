# E692 — fixed enum finalization

Date: 2026-07-21
Status: completed positive; retained; not ship

E692 extends E666's post-decode schema-enum normalization from framed dynamic
literals to fixed literal tokens. Invalid enum values are replaced only after
generation, so later model choices cannot change. Already-valid canonical and
compact direction spellings remain byte-identical. The focused invariant and
all 132 compiler-decode tests passed.

The first replay (`e692-fixed-enum-finalize-r1`, v147) is excluded as
implementation-confounded. It compared compact direction tokens only by token
ID, rewriting every valid Stack `"column"` direction to `"row"`. V148 treats
those compact tokens as semantic spellings of the same enum values.

The corrected independently capped full Held-out r2 completed with exit 0, no
timeout or fallback, and emitted AgentEvals JSONL plus an AgentV SDK bundle.

| Held-out `n=5` | E691 v146 | E692 r2 v148 |
| --- | ---: | ---: |
| syntax / meaningful v1 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| strict v2 / coverage | 0.4000 / 1.0000 | 0.6000 / 1.0000 |
| fidelity / validity | 0.8667 / 0.9200 | 0.8667 / 0.9200 |
| structure / component recall | 0.6658 / 0.6933 | 0.6658 / 0.6933 |
| reward | 0.9210 | 0.9210 |
| AST node / edge F1 | 0.7640 / 0.6434 | 0.7640 / 0.6434 |
| latency p50 / p95 | 3714.48 / 6075.57 ms | 3489.19 / 6560.11 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

Only `held_out_tabs_01` changes versus E691: the schema-invalid
`Callout("column", title, body)` becomes `Callout("info", title, body)`.
Its corrected TabItem/TextContent structure stays byte-identical and the record
now passes strict v2 with no reason codes. The other four predictions are
byte-identical; all continuous quality metrics remain exactly flat. The small
p95 difference is not treated as a performance claim.

Retain v148 as a generalized schema-validity correction. This is one reused
scratch checkpoint and Held-out `n=5`, not a powered result or ship claim. No
checkpoint was created, synced, or promoted. The remaining two strict failures
should be diagnosed from their binding-aware reason codes before another lever.

Evidence: [JSON](iter-e692-fixed-enum-finalize-20260721.json).
