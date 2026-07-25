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
