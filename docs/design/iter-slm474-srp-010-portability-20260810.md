# SLM-474 (SRP-010): Symbolic-pack portability fixture (srp010_fixture)

Matrix set: `slm474_srp010_pack_portability`

Version: `srp010-v2`

Status: **fixture**

**Claim class:** `fixture` only. Related catalogue identity `exp-sr-12` is cited, not locked or executed as ExperimentCampaignV1 for promotion. No Julia/PySR/SymPy.

## Default isolation

- `SLM_GRAMMAR_DSL` unset: `True`
- `get_pack()` → `openui`
- `get_pack('auto')` → `openui`
- `get_pack('default')` → `openui`
- OpenUI is default: `True`

## Pack path

- Pack: `symbolic_regression` available=`True`
- Filled slots: pack_id, backend, placeholder_policy, reward_label, canonicalize, oracle
- Missing reference slots: corpus_generator, completion_artifact, scope_extractor, prop_order, incremental_engine
- Problem fingerprint: `e24faef56204128c…`
- VSP digest: `3d822ea95b3665c4…`
- Expression `v0` → canonical `v0`; oracle_ok=`True`
- SyGuS conformance: `inspired_interoperability`

## Forks

| Fork | Surface | Detail |
| --- | --- | --- |
| shadow_dsl_packs_registry | dsl/packs/ | Canonical registration is dsl/pack.py (list_packs includes 'symbolic_regression'); the legacy dsl/packs/ shadow registry still only ships openui/toy-layout/arith_sketch and does not list SR. |
| openui_pinned_language_contract | dsl/language_contract.py | LanguageContract / OUTPUT_CONTRACT_VERSION remain OpenUI-pinned (LANG_SPEC=openui-lang-0.2.x); SR uses its own problem schema rather than that contract identity. |
| sygus_inspired_not_conformance | data/progspec/synthesis_capability_report.py | SyGuS capability report conformance='inspired_interoperability' (inspired interoperability only, not SyGuS-IF standards conformance). |

## Unsupported hooks

| Hook | Status | Detail |
| --- | --- | --- |
| symbolic_expr_enumerate | available | slm_training.dsl.symbolic_expr_enumerate exercised (optimality_claim='exhaustive_under_declared_bounds', states_generated=10, unique=8); DslPack.corpus_generator remains unset |
| symbolic_expr_corpus | available | slm_training.dsl.symbolic_expr_corpus exercised (n=4, seed=474, rejections=5, leakage_clean=True); DslPack.corpus_generator remains unset |
| slot.corpus_generator | unsupported | DslPack slot 'corpus_generator' is None on symbolic_regression (honest partial pack) |
| slot.completion_artifact | unsupported | DslPack slot 'completion_artifact' is None on symbolic_regression (honest partial pack) |
| slot.scope_extractor | unsupported | DslPack slot 'scope_extractor' is None on symbolic_regression (honest partial pack) |
| slot.prop_order | unsupported | DslPack slot 'prop_order' is None on symbolic_regression (honest partial pack) |
| slot.incremental_engine | unsupported | DslPack slot 'incremental_engine' is None on symbolic_regression (honest partial pack) |

## Optional module exercise (SRP-008/009)

- Enumerate: exercised=`True` optimality_claim=`exhaustive_under_declared_bounds` states=`10` unique=`8`
- Corpus: exercised=`True` n=`4` rejections=`5` leakage_clean=`True`
- Pack slots `corpus_generator` / `completion_artifact` stay intentionally empty.

## Import audit (no OpenUI imports)

| Module | Offenders |
| --- | --- |
| `symbolic_regression.py` | — |
| `symbolic_expr_canonicalize.py` | — |
| `symbolic_expr_evaluator.py` | — |
| `symbolic_expr_fit.py` | — |
| `symbolic_expr_ir.py` | — |
| `symbolic_expr_report.py` | — |
| `symbolic_regression_pack.py` | — |

## Notes

- claim_class=fixture; no ExperimentCampaignV1 lock/execute for promotion
- related catalogue identity: exp-sr-12

## Verdict

Fixture wiring only. The symbolic-regression pack exercises parse/canonicalize/oracle/evaluate-fit/evidence, optional enumerate/corpus modules when importable, and the VSP/SyGuS capability export without changing OpenUI defaults. Real exp-sr-12 certification remains a separate campaign.
