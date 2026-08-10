# SLM-480 (RSP-005): Prospective certified macro/library learning (EXP-SR-9)

**Claim class:** `fixture` only (catalogue `exp-sr-9`; not `promotion_candidate` / `ship_gate`)

**Catalogue:** `exp-sr-9`

**Primary metric (`macro_library_size_reduction_rate`, learned_mdl arm):** 0.20073

**Recommendation:** `inconclusive_fixture`

## Acceptance snapshot

| Check | Value |
| --- | --- |
| semantics_preserved | True |
| minimum_effect | 0.02 |
| mdl_rate | 0.20073 |
| frequency_rate | 0.158759 |
| control_rate | 0.0 |
| library_sources | 21 |
| prospective_sources | 9 |
| max_macros | 8 |
| promotion | False |

## Arms (prospective `macro_library_size_reduction_rate`)

| Arm | rate | macros | pack |
| --- | ---: | ---: | --- |
| frequency_macros | 0.158759 | 8 | `0752e05e37d8…` |
| learned_mdl | 0.20073 | 8 | `978ece97db47…` |
| no_macros | 0.0 | 0 | `9c81aaf9a84b…` |

## Scope

Fixture-scale EXP-SR-9 evidence for certified macro/library learning. Library mined from train_seeds.jsonl only; prospective scoring on held_out+ood rows from test_seeds.jsonl. Retrospective compression on the training corpus is reported in induction_stats but is not the success criterion. claim_class=fixture; experimental until frontier-scale prospective evidence.

Command: `python -m scripts.run_rsp005_macro_library --mode fixture`

Full detail: `docs/design/iter-slm480-rsp-005-macro-library-20260810.json`.
