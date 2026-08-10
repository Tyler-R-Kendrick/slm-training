# SLM-485 (RSP-003): Extended static-artifact + packed semantic-summary (EXP-SR-7)

**Claim class:** `fixture` only (catalogue `exp-sr-7`; not `promotion_candidate` / `ship_gate`)

**Catalogue:** `exp-sr-7`

**Primary metric (`static_summary_domain_parity_rate`):** 1.0

**Certified:** `True`

**Recommendation:** `inconclusive_fixture`

## Acceptance snapshot

| Check | Value |
| --- | --- |
| checks_passed / total | 9 / 9 |
| kill_gate_triggered | False |
| cold_path_improved | True |
| best_candidate_arm | packed_semantic_summary |
| best_delta_ms | 82.514 |
| promotion | False |

## Parity checks

| Check | Pass |
| --- | --- |
| `differential_corpus_oracle_parity` | True |
| `base_artifact_domain_parity` | True |
| `control_summary_domain_parity` | True |
| `static_lalr_summary_domain_parity` | True |
| `packed_semantic_summary_domain_parity` | True |
| `provider_mediated_domain_parity` | True |
| `deliberate_miss_fallback_closed` | True |
| `packed_summary_request_independent` | True |
| `packed_summary_checker_fail_closed` | True |

## Cold-path summary-step deltas vs base_artifact

- baseline_summary_step_p50_ms: 182.894
- deltas: `{'base_artifact': 0.0, 'control_summary': 79.233, 'static_lalr_summary': -57.091, 'packed_semantic_summary': 82.514, 'provider_mediated': -5.209}`

## Scope

Fixture-scale EXP-SR-7 evidence for incremental request-independent summary layers on the certified completion artifact. Parity gate uses the fixed hard prefix plus whole-program/empty-prefix samples from STATIC_LALR_CORPUS; every intermediate prefix is owned by tests/test_dsl/test_static_control_domain.py (E1 suite). Measures process launch, tokenizer/pack construction, summary preload, and first completion-domain query — not model init, first-forward, or full first-verified-program latency. claim_class=fixture; never promotion_candidate/ship_gate. Request-local symbols/binders/placeholders remain overlay.

Command: `python -m scripts.run_rsp003_static_summary --mode fixture`

Full detail: `docs/design/iter-slm485-rsp-003-static-summary-20260810.json`.
