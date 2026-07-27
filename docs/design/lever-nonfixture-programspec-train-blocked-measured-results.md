# Non-fixture programspec / language_contract train — BLOCKED (NOT SHIP)

**Honesty:** build-only evidence; **no model scoreboard** (train load failed). Not ship.

## Hypothesis
Strict non-fixture corpora (`programspec`, `language_contract`) load under the
current `train_model` symbol-only / role-safe contracts and can lift
`meaningful_program_rate` vs fixture exposure12 on smoke n=3.

## Results

| version | source | n | train_load | error |
| --- | --- | ---: | --- | --- |
| lever_programspec_v1 | programspec (count=80, seed=47, cap=12) | 1086 | **FAIL** | non-canonical Harness serialization (`*_scope`) |
| lever_programspec_doc_v1 | programspec + `--target-kinds document` | 587 | **FAIL** | still emits `*_scope`; same serialization error |
| lever_langcontract_v1 | language_contract | 75 | **FAIL** | placeholder in non-content `TabItem.value` |

### Synthesis feedback (programspec v1)
- high_rejection_rate (~37% of candidates dropped)
- placeholder_contract_violations=333
- sanitize_fallbacks=184
- Gates held (no weakening).

## Decision
**BLOCKED** — builds are not train-loadable; model metrics unavailable.

This is a **harness/synthesis** gap, not a model-hyperparameter gap. Next action
belongs to `improve-openui-harnesses` + `synthesis-feedback`: emit canonical
Harness serialization for scope/document targets and enforce role-safe slots
before re-running the corpus lever.

## Contrast
Loadable corpora that already produced scoreboards: `wf_smoke_v2`,
`lever_exposure12_v1` (fixture), `e1291`, `e937` (e937 regressed).

Captured: 2026-07-27T17:20:22.706621+00:00
