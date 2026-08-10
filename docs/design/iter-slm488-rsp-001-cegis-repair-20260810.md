# SLM-488 (RSP-001): Witness-guided CEGIS repair (EXP-SR-4)

**Claim class:** `fixture` only (catalogue identity claim_class=`screening`; execution is fixture — no promotion)

**Catalogue:** `exp-sr-4`

**Primary metric (`verified_repair_yield`, witness_cegis arm):** 1.0

**Effect vs regenerate:** `0.375` (minimum_effect=`0.05`)

**Recommendation:** `adopt-optional`

## Acceptance snapshot

| Check | Value |
| --- | --- |
| falsifier_holds | False |
| clears_regenerate_control | True |
| clears_minimum_effect | True |
| default_off | True |
| learned_scorer_status | available |
| seeds | [0, 1, 2] |
| promotion | False |

## Arms (`verified_repair_yield`)

| Arm | yield |
| --- | ---: |
| conflict_slice | 1.0 |
| conflict_slice_expanded | 1.0 |
| edit_distance | 1.0 |
| full_remask | 0.0 |
| learned_scorer | 1.0 |
| no_repair | 0.0 |
| regenerate | 0.625 |
| suffix_rollback | 0.0 |
| witness_cegis | 1.0 |

## Falsifier notes

- Witness CEGIS cleared regenerate control + minimum_effect on this fixture; still claim_class=fixture — adopt-optional only, never default-on or promotion.

## Scope

Fixture-scale EXP-SR-4 witness-guided CEGIS repair over SPV2-05 semantic repair records and one EFS2-03 topology conflict. Composes semantic_repair + conflict_slice_repair + RepairResidualV1 projection; no parallel CEGIS engine. Matched compute via shared max_cycles=3. claim_class=fixture; no promotion; default lever OFF.

Command: `python -m scripts.run_rsp001_cegis_repair --mode fixture`

Full detail: `docs/design/iter-slm488-rsp-001-cegis-repair-20260810.json`.
