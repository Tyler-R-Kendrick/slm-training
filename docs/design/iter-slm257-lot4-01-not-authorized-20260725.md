# SLM-257 LOT4-01 — Downstream-gate disposition (lotus-openui-compute-frontier-v1)

Verdict: **not_authorized**

LOT4-01 (SLM-257) activation requires: SLM-251 supplies faithful treatment and continued explicit checkpoints with fairness manifests; SLM-252/253/256 select the objective/readout/workspace or explicitly close those branches; SLM-254 defines allowed causal-use language; SLM-255 defines accepted-equivalence evaluation. The LOT1-02 launch gate reports 'not_authorized' (LOT1-01 disposition 'not_authorized'): no faithful K x c model path, curriculum, Stage 0 parent, or continued-explicit control exists, and SLM-249's oracle ceiling is not positive. Closing not_authorized in plan-only/bounded-diagnostic mode; no training, factorial, intervention, or systems code is added by this disposition.

## Upstream chain

- LOT1-02 launch gate (SLM-251): `not_authorized`
- LOT1-01 activation gate (SLM-250): `not_authorized`
- Activation requirement: SLM-251 supplies faithful treatment and continued explicit checkpoints with fairness manifests; SLM-252/253/256 select the objective/readout/workspace or explicitly close those branches; SLM-254 defines allowed causal-use language; SLM-255 defines accepted-equivalence evaluation.

## Non-goals honored

No training campaign, factorial, readout/decoder, intervention, systems measurement, or production default change. This disposition is itself the deliverable while the upstream activation gate is unmet.
