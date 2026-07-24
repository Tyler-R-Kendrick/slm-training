# Grammar leakage audit

SLM-284 adds an evaluation-only four-arm audit for a fixed checkpoint and
fixed request set. The primary quality metric is binding-aware meaningful
program v2; syntax parse is reported separately and is never a ship proxy.

## Arms and evidence

`scripts/evaluate_model.py --grammar-leakage-audit` runs these arms with the
same evaluator, checkpoint, corpus selection, and configured seed:

| Arm | Decode policy | Purpose |
| --- | --- | --- |
| `raw` | grammar constraint disabled | Unconstrained baseline |
| `constrained` | grammar constraint enabled | Measure grammar effect |
| `repaired` | constrained plus LTR repair | Isolate repair effect |
| `uniform_at_unforced` | constrained; deterministic seeded uniform legal choice when more than one legal compiler path exists | Control for forced singleton decisions |

Each scorecard retains meaningful-v2, parse, placeholder/binder-reference F1,
structural similarity, gates, per-record semantic-factor and complexity slices,
and raw-arm deltas. The bounded compiler decision trace records position, legal
set size, raw top-1, legal membership, selected token, override, and legal
probability mass. The legacy causal trace primitive separately preserves the
same decision facts for direct replay probes.

The control is statelessly derived from the configured seed, prefix, and legal
candidate paths. It therefore changes no singleton decision and does not
advance global RNG state. It is presently exercised by compiler LTR paths;
other decoder families report their normal scorecards rather than pretending
they received a uniform control.

## 2026-07-24 local diagnostic attempt

The timestamped record
[`iter-slm284-grammar-leakage-fixture-20260724.json`](iter-slm284-grammar-leakage-fixture-20260724.json)
records the local-only attempt. It intentionally ran one smoke example and no
training corpus. No four-arm scorecard completed: older local checkpoints were
rejected by the symbol-only/v2 contract, and the available v2 scratch
checkpoint was interrupted by the repository run cap before it emitted an
audit artifact. This is negative operational evidence only, not a model or
ship result. It did not relax a contract, gate, or corpus restriction.

The implementation-level regression suite passed (67 focused tests), including
the four-arm evaluator wiring, seeded uniform/singleton trace invariants,
raw/legal audit summary, and optional feature-flag preservation. A completed
local campaign must replace this entry with its full per-arm JSON and AgentV
bundle before any promotion or quality claim.
