# Grammar leakage audit

The audit compares deterministic constrained decoders for a fixed checkpoint
and request set. Binding-aware meaningful program v2 remains the primary
quality metric; syntax parse is separate and is never a ship proxy. Raw model
logits are retained only as constraint-shadow telemetry and are never emitted
as an unconstrained program.

## Arms and evidence

`scripts/evaluate_model.py --grammar-leakage-audit` runs these arms with the
same evaluator, checkpoint, corpus selection, and configured seed:

| Arm | Decode policy | Purpose |
| --- | --- | --- |
| `constrained_native` | mandatory constrained deterministic decode, native representation | Safe baseline |
| `constrained_compiler` | mandatory constrained compiler-tree decode | Compare exact forest ranking and speculative prefills |

Each scorecard retains meaningful-v2, parse, placeholder/binder-reference F1,
structural similarity, gates, per-record semantic-factor and complexity slices,
and baseline deltas. The bounded compiler decision trace records position, legal
set size, raw top-1, legal membership, selected token, override, and legal
probability mass. The legacy causal trace primitive separately preserves the
same decision facts for direct replay probes.

Complete singleton legal sets bypass inference and therefore do not fabricate a
raw-logit observation. Ambiguous decisions may record the model's raw winner,
but selection remains inside the exact legal domain. Unsafe
`grammar_constrained=False`, unconstrained fallback, sampling, and
uniform-at-unforced model-build flags are rejected.

## 2026-07-24 local diagnostic attempt

The timestamped record
[`iter-slm284-grammar-leakage-fixture-20260724.json`](iter-slm284-grammar-leakage-fixture-20260724.json)
records the historical pre-policy local-only attempt. Its four named arms are
retained as attempted provenance, but the raw and uniform emitted arms are now
superseded and cannot be rerun through the production evaluator. It
intentionally ran one smoke example and no
training corpus. No four-arm scorecard completed: older local checkpoints were
rejected by the symbol-only/v2 contract, and the available v2 scratch
checkpoint was interrupted by the repository run cap before it emitted an
audit artifact. This is negative operational evidence only, not a model or
ship result. It did not relax a contract, gate, or corpus restriction.

The current implementation-level suite covers the safe two-decoder evaluator,
singleton inference bypass, shadow-only raw diagnostics, and optional
feature-flag preservation. A completed
local campaign must replace this entry with its full per-arm JSON and AgentV
bundle before any promotion or quality claim.

The committed SLM-287 locked-power artifacts also predate this policy and keep
their original raw/constrained/repaired labels as immutable historical
provenance. The live SLM-287 runner now requests only
`constrained_native`/`constrained_compiler`; that changes its locked campaign
digest, so a fresh preregistered execution is required before comparing new
rows with the historical tables.
