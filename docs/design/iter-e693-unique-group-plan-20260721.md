# E693 — unique public Group planning

Date: 2026-07-21
Status: completed positive tradeoff; retained; not ship

E693 plans a public `*Group` component from an authored base noun only when the
library has no standalone component with that base name. This general rule
covers SwitchGroup, RadioGroup, and CheckBoxGroup without fixture literals.
The focused invariant and all 133 compiler-decode tests passed.

The independently capped full Held-out replay completed with exit 0, no timeout
or fallback, and emitted AgentEvals JSONL plus an AgentV SDK bundle.

| Held-out `n=5` | E692 v148 | E693 v149 |
| --- | ---: | ---: |
| syntax / meaningful v1 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| strict v2 / coverage | 0.6000 / 1.0000 | 0.6000 / 1.0000 |
| fidelity / validity | 0.8667 / 0.9200 | 1.0000 / 1.0000 |
| structure / component recall | 0.6658 / 0.6933 | 0.6624 / 0.7933 |
| reward | 0.9210 | 0.9634 |
| AST node / edge F1 | 0.7640 / 0.6434 | 0.7754 / 0.5901 |
| latency p50 / p95 | 3489.19 / 6560.11 ms | 3864.72 / 6155.73 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

Only settings changes. It now emits the planned Slider plus
`SwitchGroup(..., [SwitchItem(notify, notify.desc, volume)])`, retains both in
the root, and covers all three visible slots. The missing-placeholder failure
disappears. Strict stays 3/5 because Slider still carries the role-invalid
non-enum variant `"tet"`. The four other predictions are byte-identical.

Retain v149 as a positive coverage/quality tradeoff: fidelity, validity,
recall, reward, node F1, and p95 improve, while structure and edge F1 slip.
This is one reused scratch checkpoint and Held-out `n=5`, not a powered result
or ship claim. No checkpoint was created, synced, or promoted. Next diagnose
Slider.variant without disturbing the new SwitchGroup subtree.

Evidence: [JSON](iter-e693-unique-group-plan-20260721.json).
