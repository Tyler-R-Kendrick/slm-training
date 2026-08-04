# Autotrain c3 (continuous-openui-local, 2026-08-04): component-plan regression, rejected

**Verdict:** reject, regression. Size-matched control vs `component-plan`
candidate, both 1,755,764 params, 21 steps, freshly trained (checkpoints
`07b25357...5362043e0` control / `8e25e2a0...810752c0` component-plan).

The candidate regresses on every guarded quality metric: structural
similarity `.2308→.1725` (`-0.0583`), binder reference F1 `.7333→.6333`
(`-0.10`), meaningful-program rate `.333→0`, and component-type recall
`.167→0`. It is also `152.09` ms slower p50 (`9533.62` vs `9381.53`). SDLC
Phase A classifies this **non-positive**
(`non_regression_fail:binder_reference_f1`,
`primary_metric_null_or_worse:smoke.structural_similarity`); no stack layer
opened.

AgentV bundles are complete and gates fail honestly on fixture volume
(`insufficient_n actual=3 need>=20`) plus quality thresholds — an expected
gate rejection on a 3-sample smoke fixture, not a failure.

Both checkpoints are local, explicit no-sync, and not reusable, promotable,
or ship evidence. Lean is `not_applicable:screening`.
`checkpoint_documentation_required` is `true` (new checkpoints created), so
this cycle's roster row is recorded in
[`docs/MODEL_CARD.md`](../MODEL_CARD.md) → Current checkpoint roster; no
promotion occurred, so the README model-card *summary* section is unchanged.

Next: test the distinct size-matched `component-edge` hypothesis
(`c20260804-continuous-openui-local-8c0b60dd-c3-component-edge`), per the
ranked successor priority, keeping the matched control as the size-matched
baseline.

Machine evidence:
[`autotrain-cycle-20260804-c3-component-plan-regression.json`](autotrain-cycle-20260804-c3-component-plan-regression.json).
