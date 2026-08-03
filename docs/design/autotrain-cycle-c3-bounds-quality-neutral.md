# Autotrain c3: frozen `--grammar-completion-bounds` replay, quality-neutral

**Verdict:** reject, quality-neutral. This is the first cycle in this loop to
complete end to end (train, `--ship-gates` eval with AgentV, honest gates)
after the infrastructure repairs in c1/c2 — a real, scoreable measurement
rather than an infra failure.

The frozen `-bounds` and `-control` arms reused the c2 checkpoints
(`f2fe8f5a...b81b2` control / `2c1749c6...41742f64` bounds; 1,608,962 params,
21 steps) and ran evaluation only. Both arms are quality-identical on smoke
`n=3`: structure `.0575`, binder F1 `.6333`, parse `1.0`, meaningful/recall/AST
gates all `0`, reward `0`. Bounds is `42.91` ms slower p50 (`3107.70` vs
`3064.79`) with no quality offset, so this is a straightforward reject —
`--grammar-completion-bounds` is quality-neutral at this size/step budget.

AgentV bundles are complete and gates fail honestly on fixture volume
(`insufficient_n actual=3 need>=20`) plus the quality thresholds
(`meaningful_program_rate`, `structural_similarity`, `component_type_recall`,
`ast_beq_rate`, `canonical_beq_rate`, `reward_score` all below floor) — this
was an expected gate rejection on a 3-sample smoke fixture, not a failure.

No new checkpoint was produced (frozen replay reused c2's). Lean is
`not_applicable:screening`.

Next: test the distinct size-matched `component-plan` hypothesis
(`c20260803-continuous-openui-local-8c0b60dd-c3-component-plan`), per the
ranked successor priority.

Machine evidence:
[`autotrain-cycle-c3-bounds-quality-neutral.json`](autotrain-cycle-c3-bounds-quality-neutral.json).
