# E47 Silver+ surface-contract retrain — 2026-07-15

After repairing constrained decoding to admit lexer-native BYTE, literal, and
newline transitions, E47 was retrained for 256 CPU steps on the judged Silver+
corpus (218 records). Strict constrained LTR was then evaluated on the smoke
suite (n=3), with no unconstrained fallback.

| metric | result |
| --- | ---: |
| parse / language validity | 0/3 |
| structural similarity | 0.000 |
| placeholder fidelity | 0.000 |
| reward | 0.000 |
| constrained fallback | 0.000 |
| latency p50 | 8,846 ms |
| AgentV | 0/5, 0 execution errors |

The target-transition probe admits all 37 tokens in a Silver+ target, so the
decoder's language surface is now internally aligned. The retrained model
still produces no complete root element. Therefore parse is measuring the
end-to-end requirement of producing valid OpenUI, while structural adherence
must also be measured independently with teacher-forced or prefix-level
metrics. This checkpoint fails both: it has no parseable prediction and no
eligible structural evidence.

Decision: reject this checkpoint. Investigate native-surface learning and add
a separate teacher-forced structural-adherence scoreboard before further
decode heuristics or ship claims.

This is a scratch smoke result, not a ship claim.
