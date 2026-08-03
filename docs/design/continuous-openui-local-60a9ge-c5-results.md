# Autotrain c5 (continuous-openui-local, session 60a9ge): frozen replay, decode timeout confirmed on both arms

**Verdict:** infrastructure failure, not scoreable. This is the frozen replay
of cycle 4's `-component-edge` vs `-control` arm. Both time out identically
again: `control` compiler time 23622.2ms mean, `component-edge` 23184.0ms
mean, both past the wall-fitted `decode_timeout_seconds` (3 of 3 smoke docs
timing out on each arm).

Because **both** arms time out by essentially the same margin (as in cycle
4), this is not attributable to the `component-edge` knob. It confirms cycle
4's diagnosis: `_fit_screening_decode_timeout_seconds()` clamps the decode
budget tightly (train floor + overhead) to protect the repository's
`MAX_RUN_MINUTES=3` hard cap, and this recipe's per-document compile cost on
this host structurally exceeds that fitted budget for both arms. This is a
wall-budget/recipe-shape tension, not a code defect — widening
`decode_timeout_seconds` would only be a knob change for a future cycle, and
the clamp is deliberate (it exists to protect the hard run cap), so it is
not changed here.

## Frozen-replay exhaustion

`measurement.max_consecutive_frozen_replays` is `1`
(`src/slm_training/resources/experiments/autotrain_climb/policy.v1.json`).
Cycle 5 was the one permitted frozen replay of cycle 4's arm; both are now
exhausted at this eval-subset/recipe pairing. Per policy, the next cycle
rotates to a distinct knob from the driver's ranked priorities rather than a
third replay of the same `component-edge` arm.

No checkpoint reuse or new scoreboard here; nothing in this cycle is model
evidence. Lean is `not_applicable:retry_measurement`.

## SDLC Phase A

**Non-positive** (`harness_failure` / `measurement_incomplete` on both arms,
repeat of cycle 4; `primary_metric_unavailable`). Per `sdlc`
autotrain-iteration-delivery, no stacked PR layer is opened — local commit
and docs only.

Next: rotate to the next distinct size-matched knob hypothesis rather than
retry the exhausted `component-edge` arm again.

Machine evidence:
[`continuous-openui-local-60a9ge-c5-results.json`](continuous-openui-local-60a9ge-c5-results.json).
