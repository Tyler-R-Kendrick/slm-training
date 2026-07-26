# SLM-252 LOT2-01 — Downstream-gate disposition (causal-supervision-routing-gate-v1)

Verdict: **not_authorized**

LOT2-01 (SLM-252) activation requires: SLM-251 (LOT1-02) authorizes LOT2 and provides the selected parent, faithful treatment checkpoint/recipe, continued explicit control, curriculum, and fairness manifests. The LOT1-02 launch gate reports 'not_authorized' (LOT1-01 disposition 'not_authorized'): no faithful K x c model path, curriculum, Stage 0 parent, or continued-explicit control exists, and SLM-249's oracle ceiling is not positive. Closing not_authorized in plan-only/bounded-diagnostic mode; no training, factorial, intervention, or systems code is added by this disposition.

## Upstream chain

- LOT1-02 launch gate (SLM-251): `not_authorized`
- LOT1-01 activation gate (SLM-250): `not_authorized`
- Activation requirement: SLM-251 (LOT1-02) authorizes LOT2 and provides the selected parent, faithful treatment checkpoint/recipe, continued explicit control, curriculum, and fairness manifests.

## Non-goals honored

No training campaign, factorial, readout/decoder, intervention, systems measurement, or production default change. This disposition is itself the deliverable while the upstream activation gate is unmet.
