# Autotrain continuous-openui-local c6: component-plan decode timeout confirmed reproducible, arm retired

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

**Verdict:** cycle 6 (`continuous-loop-20260802-continuous-openui-local-8c0b60dd-c6`)
is the frozen replay of cycle 5's `component-plan` candidate against a fresh
`control`. The control completed and honestly rejected ship gates on
fixture-scale `n=3` (as in every prior cycle). The `component-plan` arm
**reproduced the identical decode timeout** from cycle 5
(`decode_timeout_count=3`) — this time the driver's own frozen-replay logic
classified it as a completed runtime-reject rather than an infrastructure
crash, confirming cycle 5's diagnosis: this is a genuine, reproducible
wall-budget-vs-decode-cost mismatch (`compiler_decode_mode=tree` +
`component_plan_decode_weight=1.0` vs this CPU sandbox's ~29s wall budget),
not a transient fluke or a repairable harness defect.

## Outcome

The driver retired the `component-plan` arm and selected the next distinct
ranked hypothesis, `component-edge`, rather than looping on a knob
combination that cannot complete inside this host's wall cap — exactly the
self-heal behavior recommended in cycle 5's documentation ("prefer
wall-budget-compatible candidates for CPU-sandboxed screening cycles").

**Not positive** (fixture `insufficient_n` on control; reproduced timeout on
candidate, not a metric win or harness unblock). No new stack layer.

## Next priority

`component-edge` quality hypothesis, size-matched, per the driver's ranked
successor priorities.

Machine evidence:
[`autotrain-cycle-continuous-openui-local-c6-component-plan-retired.json`](autotrain-cycle-continuous-openui-local-c6-component-plan-retired.json).
