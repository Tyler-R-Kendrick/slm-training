# Autotrain c2 (continuous-openui-20260803): frozen replay, bounds quality-neutral

**Verdict:** rejected, quality-neutral. This is the `retry_measurement` frozen
replay of c1's control/bounds checkpoints (`f2fe8f5a...b81b2` /
`2c1749c6...41742f64`, identical checkpoints to the prior loop's c2/c3 — no
retrain). The `ensure_agentv_available()` preflight added in c1 worked as
intended: both arms ran train-then-eval end to end in this cold sandbox with
no infrastructure interruption.

Both arms tie exactly on every quality metric: parse `1.0`, meaningful
program rate `0`, structural similarity `.0575`, binder F1 `.6333`,
placeholder fidelity `.5278`, reward `0`. Bounds is `115.63` ms slower p50
(`1708.44` vs `1592.81`) with no offsetting quality gain. Fixture ship gates
fail as expected at `n=3` (need `>=20`); `held_out`/`adversarial`/`ood`/
`rico_held` suites are not published for this fixture recipe.

Not scoreable for promotion (fixture `n`), not reusable, not ship evidence.
Lean is `not_applicable:retry_measurement`.

Next: test the distinct size-matched `component-plan` quality hypothesis
(`c2-component-plan`, rank 1 in the driver's speculative priorities), keeping
the matched control every cycle.

Machine evidence:
[`continuous-openui-20260803-c2-bounds-quality-neutral.json`](continuous-openui-20260803-c2-bounds-quality-neutral.json).
