# RESEARCH-09 — parameterized / treewidth-aware VSS (SLM-541)

**Status:** preregistered evidence (reject)  
**Experiment key:** `RESEARCH-09`  
**Linear:** [SLM-541](https://linear.app/quickdeploy-ai/issue/SLM-541)  
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

Parameterized/treewidth-aware VSS complexity bounds predict practical solver
cost better than flat instance size on a frozen suite.

## Contract

| Arm | Role |
| --- | --- |
| Flat size / clause-count cost proxy + enumerative solver | matched control |
| Treewidth / binder / ambiguity / residual-class instrumentation + bag DP on supported subset | treatment |

| Gate | Result |
| --- | --- |
| Treewidth-proxy Spearman ρ (primary) | 0.792825 |
| Flat-size-proxy Spearman ρ | 0.991031 |
| Correlation improves vs flat | False |
| Witness disagreements | 0 |
| Timeout→fake refutation count | 0 |
| Parameter estimation failures | 0 |
| Mutation rejection rate | 1.0 |
| Decision | **reject** |

Decision rule: treewidth-proxy correlation **strictly exceeds** flat-size
proxy, with zero witness disagreements, zero timeout-as-refutation, and
mutation rejection == 1.0; else **reject** (honest null if params do not help).
Unsupported / incomplete → `unknown` (never `refuted`/`unsat`).

## Declared support + proof format

- Supported subset: exact treewidth ≤ 2 and n ≤ 8 (bool domains)
- Proof / instrumentation format: `treewidth_param_pilot/v1`
- Production enumerative solver unchanged (`encoding_adapter.exists_satisfying_assignment`)

## Campaign lock

- Manifest sha256: `fa5b8e97eb1f97f04e66e6fec9f5aeb725aa6b920bdcfd1b917adaa8d95267a0`
- Lock artifact: `src/slm_training/resources/formal/research_09_campaign_lock.v1.json`
- Claim class: `fixture` (research pilot; no promotion)

## Toolchain pin

```json
{
  "backend_id": "vss_treewidth_param_pilot",
  "enumerator": "encoding_adapter.exists_satisfying_assignment",
  "n_support_max": "8",
  "proof_format_version": "treewidth_param_pilot/v1",
  "toolchain_id": "hermetic_python_treewidth_proxy_v1",
  "treatment_solver": "bag_dp_on_nice_elimination",
  "treewidth_method": "exact_elimination_order_nle8",
  "trust_domain": "research_only/treewidth_proxy",
  "tw_support_max": "2"
}
```

## Artifacts

| Artifact | Path |
| --- | --- |
| Results JSON | [`iter-revmath-research-09-preregistered.json`](iter-revmath-research-09-preregistered.json) |
| Adapter | `src/slm_training/formal/vss_treewidth_backend.py` |
| Experiment | `src/slm_training/harnesses/experiments/research_09_vss_treewidth.py` |

## Run

```bash
# default-off: refuses without enable
PYTHONPATH=src uv run python -m scripts.run_research_09_vss_treewidth
SLM_ENABLE_RESEARCH_09=1 PYTHONPATH=src uv run python -m scripts.run_research_09_vss_treewidth --write
```

## Authority note

Filing or compiling this pilot is not evidence of production readiness.
Treewidth instrumentation remains research-only; production refutation
authority stays EVID-09 exhaustive replay / checked certificate policy.
