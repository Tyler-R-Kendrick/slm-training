# SLM-487 (RSP-008): OpenUI + symbolic-regression second-pack portability (EXP-SR-12)

**Claim class:** `diagnostic` only (catalogue `exp-sr-12`; not `promotion_candidate` / `ship_gate`)

**Catalogue:** `exp-sr-12`

**Primary metric (`second_pack_portability_parity_rate`):** 0.9090909090909091

**Certified:** `False`

## Acceptance snapshot

| Check | Value |
| --- | --- |
| checks_passed / total | 10 / 11 |
| falsifier_holds | True |
| kill_gate_triggered | True |
| rate_stable_across_seeds | True |
| seeds | [0, 1, 2] |
| promotion | False |
| openui_default_isolation | True |

## Seam checks

| Seam | Pass |
| --- | --- |
| `default_isolation_openui` | True |
| `both_packs_registered` | True |
| `openui_shared_slots` | True |
| `sr_shared_slots` | True |
| `openui_readiness_core` | True |
| `sr_e2e_parse_canonicalize_oracle_evidence` | True |
| `sr_vsp_and_sygus` | True |
| `sr_import_audit_clean` | True |
| `cross_pack_mismatch_fails_closed` | True |
| `no_learned_legality` | True |
| `zero_required_shared_seam_forks` | False |

## Honest forks / unsupported hooks

- fork_ids: `['shadow_dsl_packs_registry', 'openui_pinned_language_contract', 'sygus_inspired_not_conformance']`
- falsifier_note: Pack-specific forks are required in the shared seam (fork_ids=['shadow_dsl_packs_registry', 'openui_pinned_language_contract', 'sygus_inspired_not_conformance']); catalogue falsifier holds — not certified.

Unsupported hooks (from SRP-010 inventory):

- `symbolic_expr_enumerate`: available — slm_training.dsl.symbolic_expr_enumerate exercised (optimality_claim='exhaustive_under_declared_bounds', states_generated=10, unique=8); DslPack.corpus_generator remains unset
- `symbolic_expr_corpus`: available — slm_training.dsl.symbolic_expr_corpus exercised (n=4, seed=474, rejections=5, leakage_clean=True); DslPack.corpus_generator remains unset
- `slot.corpus_generator`: unsupported — DslPack slot 'corpus_generator' is None on symbolic_regression (honest partial pack)
- `slot.completion_artifact`: unsupported — DslPack slot 'completion_artifact' is None on symbolic_regression (honest partial pack)
- `slot.scope_extractor`: unsupported — DslPack slot 'scope_extractor' is None on symbolic_regression (honest partial pack)
- `slot.prop_order`: unsupported — DslPack slot 'prop_order' is None on symbolic_regression (honest partial pack)
- `slot.incremental_engine`: unsupported — DslPack slot 'incremental_engine' is None on symbolic_regression (honest partial pack)

## Scope

Diagnostic EXP-SR-12 certification over SRP-010 seams on OpenUI (control) and symbolic_regression (candidate). Reuses assess_symbolic_pack_portability; inventored forks are reported honestly and count against zero_required_shared_seam_forks. claim_class=diagnostic; never promotion_candidate/ship_gate. No learned legality; OpenUI defaults unchanged.

Command: `python -m scripts.run_rsp008_second_pack_portability --mode diagnostic`

Full detail: `docs/design/iter-slm487-rsp-008-portability-20260810.json`.

Precursor fixture: [iter-slm474-srp-010-portability-20260810.md](iter-slm474-srp-010-portability-20260810.md).
