# Autotrain c2 (continuous-openui-local, 2026-08-04): frozen bounds/control replay, quality-neutral

**Verdict:** reject, quality-neutral. This is the `retry_measurement` replay of
the c1 arm pair (frozen manifest
`47de63eca7855f4451ff8f6cf5decf10a9eb0ebd0976fd6c1dfe9f3682747920`) now that
the AgentV SDK preflight and harness repair from
[c1](autotrain-cycle-20260804-c1-agentv-npm-ci-infra-failure.md) are in place
— a real, scoreable measurement rather than an infra failure.

The `-bounds` and `-control` arms reused c1's checkpoints
(`d2f2dc4b...c557e44b` control / `eb81529a...25b224a2` bounds; 1,608,962
params, 21 steps, training was already complete in c1) and ran evaluation
only. Both arms are quality-identical on smoke `n=3`: structure `.0575`,
binder F1 `.6333`, placeholder fidelity `.5278`, parse `1.0`,
meaningful/recall/AST/reward gates all `0`. Bounds is `125.75` ms faster p50
(`3951.35` vs `4077.10`) with no quality offset, so this is a straightforward
reject — the frozen `-bounds` lever is quality-neutral at this size/step
budget, matching the `c3` precedent from the prior session
(`docs/design/autotrain-cycle-c3-bounds-quality-neutral.md`).

AgentV bundles are complete and gates fail honestly on fixture volume
(`insufficient_n actual=3 need>=20`) plus the quality thresholds
(`meaningful_program_rate`, `structural_similarity`, `component_type_recall`,
`ast_beq_rate`, `canonical_beq_rate`, `reward_score` all below floor) — this
is an expected gate rejection on a 3-sample smoke fixture, not a failure.
SDLC Phase A classifies this **non-positive** (`fixture_insufficient_n_alone`,
null primary-metric delta); no stack layer opened.

No new checkpoint was produced (frozen replay reused c1's). Lean is
`not_applicable:retry_measurement`. `checkpoint_documentation_required` is
`false`, so `docs/MODEL_CARD.md` / the README model-card summary are
unchanged.

Next: test the distinct size-matched `component-plan` hypothesis
(`c20260804-continuous-openui-local-8c0b60dd-c2-component-plan`), per the
ranked successor priority, rather than re-selecting the now-exhausted
bounds/control pair.

Machine evidence:
[`autotrain-cycle-20260804-c2-bounds-quality-neutral-replay.json`](autotrain-cycle-20260804-c2-bounds-quality-neutral-replay.json).
