# Preregistered experiment campaigns

`ExperimentCampaignV1` is the canonical preregistration contract for
autoresearch experiments. It separates the confirmatory decision from later
exploration and makes promotion depend on the exact plan locked before an
experiment starts.

## Contract

Every governed experiment declares:

- hypothesis and decision;
- one primary endpoint plus any secondary endpoints;
- paired control/candidate arms, seeds, budget, and stopping rules;
- quality and negative controls;
- prospectively declared Holm multiplicity families;
- promotion and rollback gates;
- typed, hashed artifact requirements; and
- one honest claim class: `wiring`, `fixture`, `diagnostic`, `screening`,
  `promotion_candidate`, or `ship_gate`.

The canonical owner is
`src/slm_training/autoresearch/experiment_campaign.py`. The autoresearch store
writes a content-addressed `CampaignLockV1` and records its digest in the
append-only campaign event chain before `experiment_started`. Execution plans,
start/finish events, and every outcome path carry that same digest. A missing,
changed, duplicated, or post-start lock fails closed.

Deviations are separate `CampaignDeviationV1` artifacts and events. They retain
the original manifest digest, are always labeled `exploratory`, and cannot
silently replace confirmatory endpoints, arms, seeds, stopping rules, or gates.

Promotion remains stronger than experiment completion. A promotion candidate
must publish every declared arm×seed row, paired example identities, endpoint
and Holm-family result, passed promotion/rollback gate, version stamp,
AgentEvals/AgentV bundle, and typed hashed artifact. Ship claims additionally
require the canonical full-suite ship gates. RL campaigns remain locked until
the readiness report is recomputed from its referenced evaluation bytes and
both digests match the campaign lock.

## Authoritative credit (promotion / ship)

Promotion_candidate and ship_gate results require content-addressed
`observation_table`, `analysis_plan`, and `credit_report` artifacts.
`credit_engine.compute_credit_report` recomputes paired effects, Holm rows,
promotion/rollback gate outcomes, and empirical promotability from observation
rows under a locked analysis plan. Recomputed endpoint values are keyed by
**manifest `endpoint_id`** (signed paired effect for the primary), not metric
name. Caller-supplied `endpoint_values` / Holm rows that disagree—or invent
values not recomputed—fail closed. Kind-only placeholder observation JSON is
rejected. Structural campaign governance **must not** clear a sole
`sufficient_evidence` failure (`promotion.py` / HTTP evaluate).

See `docs/design/authoritative-credit-one-shot-agent-prompt.md` and
`resources/experiments/authoritative_credit/defaults.v1.json`.

## Hill-climb evidence governance

Consistent autotrain progress is an evidence problem, not a smarter optimizer.
`src/slm_training/autoresearch/hillclimb.py` owns the pure predicates; campaign
and train entrypoints enforce them fail-closed.

| Gate | Rule |
| --- | --- |
| Climb vs non-climb | `validate_result_claim` **always** applies climb eligibility for `promotion_candidate` / `ship_gate` (locked held-out digest + multi-seed primary values for LCB). Seeds alone are insufficient (`primary_seed_values_missing`). Fixture / wiring / diagnostic / screening never count as climb progress; `label_as_climb=True` fails them closed. |
| Causal campaign shape | Promotion-class manifests require a control arm, positive matched control, destructive negative control, `mechanism_off_arm_ids`, and `executable_kill_criteria` (plus ≥2 seeds). |
| Exhausted knobs | `scripts/autoresearch.py` loads `exhausted_knob_ledger.json` into `validate_hypothesis_matrix` and auto-records null-measured signatures after feedback. Effects are **direction-signed** (`increase` → candidate−baseline, `decrease` → baseline−candidate) so continuous default `smoke.latency_ms_p50` (lower better) and quality metrics share one null rule: improvement ≤ minimum_effect. Absolute scores alone are never null. Same knob signature + claim class + data/eval identity is rejected until identity or claim class changes. |
| Synthesis → SFT | `scripts/train_model.py` refuses SFT when `synthesis_feedback.json` has open recommendations without a matching action/waiver in `synthesis_feedback_actions.json` (escape: `--allow-open-synthesis-feedback`, diagnostic only). |
| Capacity charge | `evaluate_promotion` / HTTP `/promotion/evaluate` pass trainable params + `EG_params` into `validate_result_claim`. Growth without `EG_params` LCB ≥ 1 fails closed; result objects may also carry the fields. |

See also `docs/design/autoresearch-autotraining.md` for the closed-loop harness
boundary.

## Endpoint transition

Binding-aware meaning-v2 is not the default until AP-001 supplies a
hash-verified certification artifact with disposition `certified`. Missing,
`revise`, `blocked`, or digest-invalid evidence selects
`binder_reference_f1`. This keeps SLM-337 compatible with the current
uncertified metric state while making the future transition explicit.

## Statistical boundary

Holm correction is applied to the prospectively declared family with stable
hypothesis-ID tie breaking. Raw p-values, rank-specific thresholds, monotone
adjusted p-values, and sequential decisions are retained for every member.
Power inputs are prospective; an underpowered nonsignificant outcome remains
inconclusive rather than becoming a supported negative claim.

This adapts:

- [OSF registrations](https://help.osf.io/article/330-welcome-to-registrations)
  and [registration amendments](https://help.osf.io/article/113-advanced-actions-registrations)
  for frozen, versioned plans;
- [Registered Reports](https://www.cos.io/initiatives/registered-reports) for
  the pre-result methods/results boundary;
- [Holm (1979)](https://doi.org/10.2307/4615733) for strong family-wise error
  control; and
- [Lakens (2022)](https://doi.org/10.1525/collabra.33267) and
  [Hoenig and Heisey (2001)](https://doi.org/10.1198/000313001300339897) for
  prospective sample-size justification and rejection of observed-power
  reasoning.

## Frozen retry successors

An infrastructure-incomplete measurement is never rerun by pretending its old
`source_commit` is current. Continuous autotrain creates a new locked successor
whose `replay_of_manifest_sha256` names the exact prior manifest. The successor
preserves the model/data recipe, endpoints, arms, seeds, budget, stopping rules,
controls, multiplicity family, and gates while binding execution to clean current
main. Both matched arms must complete before the retry action is acknowledged.
Formal obligations are not portable proof receipts: a cross-commit retry with such
obligations stops for a fresh Lean preflight.

## SLM-337 validation

The delivery is governance/fixture evidence, not a model-quality experiment
and not a checkpoint promotion. Focused tests cover canonical digest
round-trips, invalid and duplicate declarations, post-lock mutation, relocking,
event-chain tampering, append-only deviations, result completeness, AP-001
fallback, RL evidence forgery, legacy freeze integrity, Holm golden vectors,
and runner plan/outcome binding. The SLM-183 power-protocol fixture embeds the
canonical campaign as the AP-007 integration seam while retaining its honest
`wiring` claim.

The first fixture preflight rejected `max_wall_minutes=3` because the canonical
repository lever currently caps campaigns at two minutes. No simulation ran in
that failed preflight; the bridge was corrected to the stricter two-minute
budget before the documented fixture execution.

The final CPU fixture completed in 3.70 seconds:

```bash
PYTHONPATH=src python -m scripts.run_flow_power_protocol --mode fixture \
  --output-dir outputs/runs/slm337-campaign-governance-fixture \
  --n-targets 10 --paths-per-target 2 --n-seeds 5 \
  --seeds 0,1,2,3,4 --write-design-docs
```

It emitted canonical campaign digest
`ac0ceafb811a9cdf061973403eec80bf0016bfc0237b58612f4f835217632205`,
100 arm/seed/target cells, target variance `0.03833`, seed variance `0.015`,
paired-binary success delta `0.03`, and zero Holm rejections. The result remains
a no-go for promotion:
`wiring`, synthetic outcomes, no GPU, no trained weights, and no ship-gate
claim. Machine-readable and rendered evidence remain in
[`iter-slm183-power-protocol-20260720.json`](iter-slm183-power-protocol-20260720.json)
and
[`iter-slm183-power-protocol-20260720.md`](iter-slm183-power-protocol-20260720.md).
