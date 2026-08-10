# SLM-486 (RSP-007): Matched PySR/SRBench benchmark (EXP-SR-11)

**Claim class:** `diagnostic` only (catalogue `exp-sr-11`; not `promotion_candidate` / `ship_gate`)

**Evidence when blocked:** `external-blocked` — incomplete, not a loss.

**No SOTA claims.** Fixture-scale wiring only.

**Catalogue:** `exp-sr-11`

**Primary metric (`srbench_matched_score_gap`):** `None`

**Complete:** `False`

**Evidence:** `external-blocked`

## Environment probe

- PySR import available: `False`
- Live execution ready: `False`
- Manifest fingerprint: `bf60c4f6e8ec22aa…`
- Tool version (pinned): `0.19.4`

## Matched arms

| Arm | Status | Validation loss | Evidence |
| --- | --- | --- | --- |
| pack_enumerate_matched | complete | 0.0 | pack_enumerate |
| pysr_adapter_matched | incomplete | None | external-blocked |

## Gap analysis

- pack_validation_loss: `0.0`
- pysr_validation_loss: `None`
- srbench_matched_score_gap: `None`
- note: External PySR/SRBench arm incomplete — gap not scored as a benchmark loss.

- kill_gate_triggered: `False`
- promotion: `False`

## Scope

Fixture-scale EXP-SR-11 matched-budget wiring over SRP-010 enumerator and SRP-011 PySR adapter on a tiny SRBench-compatible problem. External PySR/Julia absence yields evidence='external-blocked' incomplete results — never benchmark losses or SOTA claims. claim_class=diagnostic; never promotion_candidate/ship_gate.

Command: `python -m scripts.run_rsp007_pysr_srbench --mode fixture`

Full detail: `docs/design/iter-slm486-rsp-007-pysr-srbench-20260810.json`.
