# SLM-481 (RSP-006): Quality-diversity corpus generation (EXP-SR-10)

**Claim class:** `fixture` only (catalogue `exp-sr-10`; not `promotion_candidate` / `ship_gate`)

**Catalogue:** `exp-sr-10`

**Primary metric (`corpus_topology_coverage`):** 0.722222

**QD descriptor version:** `rsp006_qd_descriptors/v1`

**Recommendation:** `reject_qd_not_better_than_random`

## Arm comparison

| Arm | Coverage | Validity | Accepted |
| --- | --- | --- | --- |
| `random_matched` | 0.722222 | 0.583333 | 21 |
| `frequency_balanced` | 0.666667 | 0.583333 | 21 |
| `quality_diversity` | 0.722222 | 0.583333 | 21 |

## Comparison snapshot

- qd_beats_random: `False`
- kill_gate_triggered: `False`
- automatic_adoption: `False`
- promotion: `False`

## Scope

Fixture-scale EXP-SR-10 evidence for MAP-Elites/novelty-style corpus generation over frozen pack-owned descriptors on the bounded OpenUI ProgramSpec candidate grid. Arms are size/exposure-matched by generation_budget; splits are assigned from canonical root families before derivatives; test-split roots are excluded from coverage numerators; eval corpora are never read. Equivalent canonical roots do not inflate diversity. No automatic adoption from fixture coverage — claim_class=fixture only.

Command: `python -m scripts.run_rsp006_quality_diversity --mode fixture`

Full detail: `docs/design/iter-slm481-rsp-006-quality-diversity-20260810.json`.
