# SLM-180 / SDE4-02: Minimum controller capacity fixture (sde4_02_plan)

Matrix set: `sde4-02-min-controller`
Version: `sde4-02-v1`
Status: **plan_only**

Competence target (train accuracy): `0.66`

## Hypothesis

A minimum-width CPU controller can be identified for a deterministic synthetic decision task: as MLP hidden dimension increases, train accuracy crosses a fixed competence threshold and the smallest qualifying rung marks the minimum viable controller capacity.

## Falsifier

No rung in the ladder reaches the competence target on the training split, or a smaller-width rung consistently outperforms every larger rung, indicating that the ladder does not monotonically characterize controller capacity for this recipe.

## Frozen base recipe (SHA-256)

```
069beebc4af67de54ad3545e311ca1b0334ddeed5c8357e12786ab148f00738a
```

## Ladder

| Rung | Hidden dim |
| --- | --- |
| rung_001_h8 | 8 |
| rung_002_h16 | 16 |
| rung_003_h32 | 32 |
| rung_004_h64 | 64 |
| rung_005_h128 | 128 |

## Rows

| Rung | Hidden dim | Seed | Train acc | Val acc | Params | Active params | Meets target |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rung_001_h8 | 8 | 0 | 0.0000 | 0.0000 | 0 | 0 | False |
| rung_002_h16 | 16 | 0 | 0.0000 | 0.0000 | 0 | 0 | False |
| rung_003_h32 | 32 | 0 | 0.0000 | 0.0000 | 0 | 0 | False |
| rung_004_h64 | 64 | 0 | 0.0000 | 0.0000 | 0 | 0 | False |
| rung_005_h128 | 128 | 0 | 0.0000 | 0.0000 | 0 | 0 | False |

## Verdict

No rung met the competence target on all seeds; capacity_threshold_not_identifiable = True.

**Fixture caveat:** This is wiring-only evidence. The controllers are tiny CPU MLPs trained on a deterministic synthetic decision set with no production model, no GPU, no held-out eval suites, and no ship-gate claim. The competence threshold is a fixture probe, not a production readiness criterion.
