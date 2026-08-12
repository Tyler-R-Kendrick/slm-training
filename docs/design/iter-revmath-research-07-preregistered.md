# RESEARCH-07 — Alethe/SMT proof reconstruction (SLM-565)

**Status:** preregistered evidence (accept)  
**Experiment key:** `RESEARCH-07`  
**Linear:** [SLM-565](https://linear.app/quickdeploy-ai/issue/SLM-565)  
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

Alethe/SMT proof reconstruction for theory-rich VSS constraints preserves exact
agreement on a declared theory subset without widening unknown into refutation.

## Contract

| Arm | Role |
| --- | --- |
| Exhaustive finite-domain replay (+ SAT/CNF overlap) | matched control |
| VSS→SMT + checked/reconstructed Alethe via EVID-10 `alethe_pilot` | treatment |

| Gate | Result |
| --- | --- |
| Exact agreement rate (supported subset) | 1 |
| Witness disagreements | 0 |
| Mutation rejection rate | 1.0 |
| Reconstruction→fake refutation count | 0 |
| Decision | **accept** |

Decision rule: agreement rate == 1.0, mutation rejection == 1.0, and zero
fake refutations; else reject/retire. Unsupported theory / incomplete
reconstruction → `unknown` (never `refuted`/`unsat`).

## Declared theories + proof format

- Supported SMT theories: `QF_BOOL, QF_UF`
- Proof format version: `alethe_reconstructed_pilot/v1`
- Solver: fixture-backed stub when cvc5 unavailable (fail closed, default off)

## Campaign lock

- Manifest sha256: `9ca3a98359aeb757ff1b44e26006cead56952da73fa16797a22c7efa72c577df`
- Lock artifact: `src/slm_training/resources/formal/research_07_campaign_lock.v1.json`
- Claim class: `fixture` (research pilot; no promotion)

## Toolchain pin

```json
{
  "backend_id": "vss_alethe_smt_pilot",
  "certificate_format": "alethe_pilot",
  "certificate_schema": "vss_alethe_certificate/v1",
  "checker_backend": "python_alethe_reconstruct_stub",
  "cvc5_proof_format": "alethe",
  "cvc5_version_pin": "cvc5>=1.0.0 (optional; fixture stub when unavailable)",
  "encoder_family": "cnf_ref+smt_eq_stub",
  "encoder_hash": "0752e7e4570417794e8a3cbed145d659438cd055170e59d05ddcfe043bc4e270",
  "encoder_version": "cnf_ref_v1",
  "proof_format_version": "alethe_reconstructed_pilot/v1",
  "solver_mode": "fixture_stub",
  "supported_smt_theories": "QF_BOOL,QF_UF",
  "toolchain_id": "hermetic_python_alethe_stub_v1",
  "trust_domain": "encoding_bridge/cnf_ref+alethe_pilot"
}
```

## Artifacts

| Artifact | Path |
| --- | --- |
| Results JSON | [`iter-revmath-research-07-preregistered.json`](iter-revmath-research-07-preregistered.json) |
| Adapter | `src/slm_training/formal/vss_alethe_backend.py` |
| Experiment | `src/slm_training/harnesses/experiments/research_07_vss_alethe.py` |
| Encoding bridge | `src/slm_training/formal/encoding_adapter.py` (EVID-10) |

## Run

```bash
# default-off: refuses without enable
PYTHONPATH=src uv run python -m scripts.run_research_07_vss_alethe
SLM_ENABLE_RESEARCH_07=1 PYTHONPATH=src uv run python -m scripts.run_research_07_vss_alethe --write
```

## Authority note

Filing or compiling this pilot is not evidence of production readiness.
Alethe reconstruction remains research-only; production refutation authority
stays EVID-09 exhaustive replay / checked certificate policy.
