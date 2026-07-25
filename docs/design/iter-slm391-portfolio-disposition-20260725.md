# SLM-391 (DSH4-06): staged DSL harness portfolio disposition

**Claim class:** fixture (disposition audit at fixture scale; no ship claim)

**Schema:** `StagedDslHarnessDispositionV1`

**Evidence cutoff commit:** `4d93b0ab76f398b0f8ca25624fa54df5696b65b5`

**Generated at:** 2026-07-25T14:36:43.265252Z

Every claim names its compatible code/suite/config/hardware identities or is `unknown`. Missing or incomparable artifacts are classified `unknown` per the stop rule and are never normalized through unsupported assumptions. Negative, invalid, timeout, no-effect, contaminated, and prediction-identical outcomes remain first-class rows. Efficiency and semantic claims are separate; no fixture-only row supports a capability or ship claim.

## Claim dispositions

| Claim | Scope | Verdict | Supported bounds |
| --- | --- | --- | --- |
| cap0_grammar | model_capability | **unknown** | — |
| cap1_semantics | model_capability | **rejected** | — |
| cap2_operators | harness_contract | **supported** | fixture-scale harness-contract evidence only: exact bounded legal-set enumeration over a 4-candidate fixture with reserved serialization; not a model capability, NL transform, or ship claim |
| distillation | model_capability | **unknown** | — |
| efficiency | model_capability | **unknown** | — |
| trace_evals | harness_contract | **supported** | fixture-scale harness-contract evidence only: the lifecycle state machine, quarantine stop rules, freeze integrity, and one-touch confirmation all verified on synthetic traces; no runtime contamination claim |

### cap0_grammar — CAP0 exact grammar reproduction

- Verdict: **unknown** (model_capability)
- Rationale: required artifacts absent on this lineage: src/slm_training/harnesses/train_data/cap0_eval_freeze.py, src/slm_training/harnesses/experiments/slm362_cap0_experiment.py, docs/design/iter-slm361-*.json, docs/design/iter-slm362-*.json; classified unknown per the stop rule — never normalized through unsupported assumptions
- Identities: code `4d93b0ab76f398b0f8ca25624fa54df5696b65b5`; suite `n/a`; config `n/a`; hardware `{'platform': 'Linux-6.18.33.2-microsoft-standard-WSL2-aarch64-with-glibc2.39', 'machine': 'aarch64', 'python': '3.12.3', 'device': 'cpu'}`

| Suite | System | Outcome | Notes |
| --- | --- | --- | --- |
| — | — | — | no suite rows ran on this lineage |

Missing required artifacts (stop rule):
- `src/slm_training/harnesses/train_data/cap0_eval_freeze.py`
- `src/slm_training/harnesses/experiments/slm362_cap0_experiment.py`
- `docs/design/iter-slm361-*.json`
- `docs/design/iter-slm362-*.json`

### cap1_semantics — CAP1 held-out semantic grounding

- Verdict: **rejected** (model_capability)
- Rationale: CERT_CAP1 rejected with stop codes ['underpowered', 'ignores_schema', 'fails_hard_contrasts', 'cap0_regression']; the strongest retained configuration does not clear the preregistered paired-evidence bar at fixture scale, so held-out semantic and ship capability stay closed
- Identities: code `4d93b0ab76f398b0f8ca25624fa54df5696b65b5`; suite `2acc2014f634feca7baaca23b88ea4d74e5ce9b7344629fe810738ee192d4575`; config `5dbfaf66f3959c4ddf6d8ad7a165f9dca89167ae6882cc9ff7e90892be3f366c`; hardware `{'platform': 'Linux-6.18.33.2-microsoft-standard-WSL2-aarch64-with-glibc2.39', 'machine': 'aarch64', 'python': '3.12.3', 'device': 'cpu'}`

| Suite | System | Outcome | Notes |
| --- | --- | --- | --- |
| cap1_two_pack_freeze/v1 | oracle_ceiling | positive | schema-consuming exact-target replay ceiling; harness-contract evidence only (relevant schema perturbations abstain) |
| cap1_two_pack_freeze/v1 | abstain_control | negative | matched control: universal abstention must fail the gates |
| cap1_two_pack_freeze/v1 | constant_prediction_control | prediction_identical | prediction-identical control: one constant program for every row |
| cap1_two_pack_freeze/v1 | train_eval_overlap_audit | negative | eval/training overlap detection: no shared root families or targets |
| cap1_two_pack_freeze/v1 | arm:NL_NO_SCHEMA | negative | matched-exposure arm; fixture scale |
| cap1_two_pack_freeze/v1 | arm:SCHEMA_UNFILTERED | negative | matched-exposure arm; fixture scale |
| cap1_two_pack_freeze/v1 | arm:SCHEMA_FILTERED_SINGLE | negative | matched-exposure arm; fixture scale |
| cap1_two_pack_freeze/v1 | arm:SCHEMA_FILTERED_MULTI | negative | matched-exposure arm; fixture scale |
| cap1_two_pack_freeze/v1 | CERT_CAP1 | negative | certificate decision preserved verbatim; fixture n is honest |
| meaningful_v2_gaming | binding_aware_meaningful_v2 | positive | metric-only evidence; never a model capability claim |

### cap2_operators — CAP2 operator legal-set enumeration

- Verdict: **supported** (harness_contract)
- Rationale: operator legal-set enumeration reproduces the independent expectation exactly at fixture scale
- Identities: code `4d93b0ab76f398b0f8ca25624fa54df5696b65b5`; suite `dsl.operators.legal_set bounded fixture`; config `03723d88ad2e09a367c62cfa7c04775246df994c059d63930e6b2c725f0fcb94`; hardware `{'platform': 'Linux-6.18.33.2-microsoft-standard-WSL2-aarch64-with-glibc2.39', 'machine': 'aarch64', 'python': '3.12.3', 'device': 'cpu'}`

| Suite | System | Outcome | Notes |
| --- | --- | --- | --- |
| dsl.operators.legal_set | bounded_exact_fixture(n=4,allowed=2) | positive | exact enumeration matches the independent expectation (2 legal of 4) |
| dsl.operators.legal_set | truncated_budget_fixture | negative | budget truncation must classify UNKNOWN, never unsupported |

### distillation — DSH4 distillation (teacher -> student action programs)

- Verdict: **unknown** (model_capability)
- Rationale: required artifacts absent on this lineage: docs/design/iter-dsh4-01*.json, docs/design/iter-dsh4-02*.json; classified unknown per the stop rule — never normalized through unsupported assumptions
- Identities: code `4d93b0ab76f398b0f8ca25624fa54df5696b65b5`; suite `n/a`; config `n/a`; hardware `{'platform': 'Linux-6.18.33.2-microsoft-standard-WSL2-aarch64-with-glibc2.39', 'machine': 'aarch64', 'python': '3.12.3', 'device': 'cpu'}`

| Suite | System | Outcome | Notes |
| --- | --- | --- | --- |
| — | — | — | no suite rows ran on this lineage |

Missing required artifacts (stop rule):
- `docs/design/iter-dsh4-01*.json`
- `docs/design/iter-dsh4-02*.json`

### efficiency — Systems/efficiency (latency, work, calls)

- Verdict: **unknown** (model_capability)
- Rationale: no normalized comparative measurement (matched baseline, hardware, batch, model calls, node passes, compiler/verifier work, output length, fallback/timeout) exists on this lineage; classified unknown per the stop rule
- Identities: code `4d93b0ab76f398b0f8ca25624fa54df5696b65b5`; suite `n/a`; config `n/a`; hardware `{'platform': 'Linux-6.18.33.2-microsoft-standard-WSL2-aarch64-with-glibc2.39', 'machine': 'aarch64', 'python': '3.12.3', 'device': 'cpu'}`

| Suite | System | Outcome | Notes |
| --- | --- | --- | --- |
| cap1_two_pack_freeze/v1 | slm368_fixture_workload | no_effect | workload/hardware/batch/call identity recorded; no matched-baseline measurement exists, so no efficiency effect is claimed |

### trace_evals — Trace-to-eval candidate lifecycle (SLM-390)

- Verdict: **supported** (harness_contract)
- Rationale: trace-eval lifecycle verified end to end with the stop-rule trace held in quarantine
- Identities: code `4d93b0ab76f398b0f8ca25624fa54df5696b65b5`; suite `077867cde6b50fb4a02eb5b67a76a1982f10df7d09cc862964bbbc1b4767a942`; config `slm390_trace_evals lifecycle fixture`; hardware `{'platform': 'Linux-6.18.33.2-microsoft-standard-WSL2-aarch64-with-glibc2.39', 'machine': 'aarch64', 'python': '3.12.3', 'device': 'cpu'}`

| Suite | System | Outcome | Notes |
| --- | --- | --- | --- |
| slm390_trace_evals | clean_trace_lifecycle | positive | QUARANTINED->REVIEWED->FROZEN->CONFIRMATION_USED->PROMOTED explicit transitions only |
| slm390_trace_evals | undeidentifiable_trace | invalid | secret outside redactable fields: stop rule keeps it quarantined |

## Follow-up recommendations (deduplicated, evidence-grounded)

- **Scale CERT_CAP1 paired evidence beyond fixture n** (`cap1-scale-paired-n`): cap1_semantics is rejected: the frozen suite harness is intact and the matched experiment runs; only the preregistered paired-n bar is unmet at fixture scale
- **Port the SLM-361/SLM-362 frozen CAP0 suite onto this lineage** (`cap0-port-frozen-suite`): cap0_grammar is unknown: the CAP0 freeze and matched experiment harness are absent on this lineage (parallel unmerged branches); classification cannot proceed without them
- **Land DSH4-01/DSH4-02 distillation artifacts, then reclassify** (`dsh4-distillation-evidence`): distillation is unknown: the DSH4-01/02 evidence artifacts (in review as PRs #903/#905) are not present on this lineage
- **Run a normalized matched-baseline efficiency measurement** (`efficiency-normalized-baseline`): efficiency is unknown: workload identity is recorded but no matched-baseline latency/work comparison exists on this lineage

## Exact command

```bash
python -m scripts.run_slm391_portfolio_disposition --mode fixture --cap1-steps 4 --cap1-records-per-arm 8 --cap1-seeds 0
```

## Limitations

- Fixture scale only; a fixture certificate or gate pass is wiring evidence, never a held-out semantic or ship capability.
- CAP0 (SLM-361/362) and DSH4-01/02 distillation artifacts are absent on this lineage (parallel unmerged branches / in-review PRs); those claims are `unknown`, not negative.
- No promotion, checkpoint, model-card, or default change results.
