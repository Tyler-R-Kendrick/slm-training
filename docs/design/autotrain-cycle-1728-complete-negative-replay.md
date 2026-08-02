# Autotrain c1728: complete negative batch-size replay

**Verdict:** the immutable token-kind cache cleared the frozen measurement
blocker. Both size-matched arms completed 3/3 records with zero decode timeouts,
zero fallbacks, and zero AgentV execution errors. The batch-size-1 candidate is a
complete negative model result: it ties the saturated binder metric, produces
substantially worse structure, and costs substantially more decode time. It is
rejected; no promotion is authorized.

## Result matrix

| Arm | Records | Params | Parse | Binder F1 | Meaningful | Structure | p50 | Timeout | Disposition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control, batch 2 | 3/3 | 1,608,962 | 1.0 | 1.0 | 0.3333 | 0.3656 | 3,098.46 ms | 0 | Complete fixture control; ship gates fail |
| candidate, batch 1 | 3/3 | 1,608,962 | 1.0 | 1.0 | 0.3333 | 0.1557 | 18,199.25 ms | 0 | Complete negative; reject batch-size arm |

Both arms retain grammar validity, binder fidelity, and fail-closed constrained
decode. The candidate loses 0.2099 absolute structural similarity (57.4% relative),
ties meaningful-program rate, and is 5.87 times slower at p50. Its AST edge F1 falls
from 0.2222 to 0.0417. This is fixture evidence (`n=3`), not a ship claim, but it is
complete enough to reject this matched arm.

## Runtime and repair matrix

| Signal | c1727 incomplete candidate | c1728 complete candidate | Interpretation |
| --- | ---: | ---: | --- |
| completed documents | 2 / 3 | 3 / 3 | Infrastructure blocker cleared |
| total decode work | 51.617 s | 49.659 s | -3.79% while finishing more output |
| compiler | 40.584 s | 38.656 s | -4.75%; token-kind cache effective |
| backbone | 10.535 s | 10.525 s | Stable |
| neural forwards | 115 | 117 | Two additional exact completed decisions |
| unique completion states | 268,764 | 270,456 | Same search scale; completed terminal path |
| witness states expanded | 13,875 | 13,907 | Same authority scale |
| parser forks | 282,566 | 284,299 | Same authority scale |

The cache changes no candidate ordering or grammar authority. The completed hero
output confirms that the previous timeout was an infrastructure artifact, while the
now-visible quality failure belongs to the batch-size-1 trained model.

## Harness feedback

c1728 exposed two end-to-end steering defects:

1. The screening primary was `smoke.binder_reference_f1`, but both arms score 1.0;
   it cannot distinguish the observed 0.2099 structural-quality gap.
2. A completed frozen replay retained a stale rank-1 infrastructure hypothesis to
   repair and replay the already-complete measurement.

Policy v3 makes `smoke.structural_similarity` the screening primary with a 0.01
minimum effect and requires both parse rate and binder F1 not to regress. The model
stays size matched. The screening bank gains established loss-only component-plan,
component-edge, component-inventory, binder-topology, and joint component-structure
arms; none grows the model or weakens constrained decode. Completed retry handoffs
now advance to a distinct quality arm instead of another infrastructure replay.

Structural similarity is already the promotion primary and has the Mathlib-free
LeverProof monotonicity theorem used by the promotion preflight. Screening remains
fixture-only; the Lean certificate, multi-seed promotion protocol, full AgentEvals
suites, parameter-efficiency gate, and ship gates remain mandatory.

## Next-run priorities

1. Run the new size-matched binder-topology screening arm selected for c1729; the
   only causal change is loss weight 0.25, not capacity.
2. Rank on structural similarity; require parse and binder F1 non-regression and
   report latency as cost, not as the quality primary.
3. If the arm is positive, confirm the same knobs under a new seed before promotion.
4. Keep Lean structural-metric preflight on promotion cadence and preserve typed
   assumption/theorem discrepancies.
5. If structural supervision remains null, use the resulting per-record AST edge/
   node residuals to choose component-edge or component-plan supervision next.

No checkpoint was created or promoted in c1728; both checkpoint hashes were reused
from the already documented c1716/c1717 scratch runs. No model-card or README update
is required. Machine-readable evidence is in
[`autotrain-cycle-1728-complete-negative-replay.json`](autotrain-cycle-1728-complete-negative-replay.json).
