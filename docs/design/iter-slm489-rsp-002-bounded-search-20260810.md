# SLM-489 (RSP-002): Neural-guided bounded search + infilling order (EXP-SR-6)

**Claim class:** `fixture` only (catalogue `exp-sr-6` claim_class=`screening`; execution is fixture — no promotion)

**Catalogue:** `exp-sr-6`

**Primary metric (`bounded_search_hit_rate`, neural_prior arm):** 1.0

**Control (left_to_right):** 0.0

**Delta vs left_to_right:** 1.0 (minimum_effect=`0.05`)

**Reachability (generous bounds):** `True`

**Recommendation:** `adopt-optional`

## Acceptance snapshot

| Check | Value |
| --- | --- |
| falsifier_holds | False |
| clears_left_to_right_control | True |
| clears_minimum_effect | True |
| singleton_bypass | True |
| rankers_permute_only | True |
| energy_ranker_status | available |
| promotion | False |

## Arms (`bounded_search_hit_rate`)

| Arm | hit rate |
| --- | ---: |
| enumerative_best_first | 0.0 |
| left_to_right | 0.0 |
| neural_prior | 1.0 |
| neural_value_rerank | 0.5 |
| planned_hole_order | 0.0 |

## Falsifier notes

- Neural prior cleared left-to-right control on fixture; still claim_class=fixture — adopt-optional only, never default-on.

## Scope

Fixture-scale EXP-SR-6 bounded search over closed CSP fixtures and an optional OpenUI completion-forest reachability probe. Composes valid_state_search_generic + completion-forest adapter + advisory n-gram/energy rankers; no parallel search engine. Matched max_decisions=4. Avoids empty Card([]) / whole-corpus Lark oracle paths that hang under tight budgets. claim_class=fixture; no promotion; default lever OFF.

Command: `python -m scripts.run_rsp002_bounded_search --mode fixture`

Full detail: `docs/design/iter-slm489-rsp-002-bounded-search-20260810.json`.
