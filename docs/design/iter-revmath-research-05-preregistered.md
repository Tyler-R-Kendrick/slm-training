# RESEARCH-05 — VSS SAT backend with LRAT (SLM-563)

**Status:** preregistered evidence (accept)  
**Experiment key:** `RESEARCH-05`  
**Linear:** [SLM-563](https://linear.app/quickdeploy-ai/issue/SLM-563)  
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

For a declared bounded VSS subset, verified VSS→CNF + checked LRAT preserves
exact semantic agreement / mutation rejection while making warm LRAT-check
cheaper than exhaustive replay.

## Contract

| Arm | Role |
| --- | --- |
| Exhaustive VSS/CNF replay | matched control |
| Deterministic CNF + LRAT via EVID-10 `lrat_pilot` | treatment |

| Gate | Result |
| --- | --- |
| Witness disagreements | 0 |
| Mutation rejection rate | 1.0 |
| Supported-subset coverage | 1.0 |
| Median warm LRAT / exhaustive ratio | 0.870722 |
| Decision | **accept** |

Decision rule: correctness gates pass **and** ratio < 1.0; else reject/retire.
Timeout / tool failure / unsupported → `unknown` (candidates preserved).

## Campaign lock

- Manifest sha256: `28f468da369df47c00779b5ce183e3e100443dd71afbc27bd47ed9a2e109010d`
- Lock artifact: `src/slm_training/resources/formal/research_05_campaign_lock.v1.json`
- Claim class: `fixture` (research pilot; no promotion)

## Toolchain pin

```json
{
  "backend_id": "vss_lrat_sat_pilot",
  "certificate_format": "lrat_pilot",
  "certificate_schema": "vss_lrat_certificate/v1",
  "checker_backend": "python_lrat_rup_pilot",
  "encoder_family": "cnf_ref",
  "encoder_hash": "0752e7e4570417794e8a3cbed145d659438cd055170e59d05ddcfe043bc4e270",
  "encoder_version": "cnf_ref_v1",
  "toolchain_id": "hermetic_python_lrat_pilot_v1",
  "trust_domain": "encoding_bridge/cnf_ref+lrat_pilot"
}
```

## Artifacts

| Artifact | Path |
| --- | --- |
| Results JSON | [`iter-revmath-research-05-preregistered-results.json`](iter-revmath-research-05-preregistered-results.json) |
| Adapter | `src/slm_training/formal/vss_lrat_backend.py` |
| Experiment | `src/slm_training/harnesses/experiments/research_05_vss_lrat.py` |
| Encoding bridge | `src/slm_training/formal/encoding_adapter.py` (EVID-10) |

## Run

```bash
# default-off: refuses without enable
PYTHONPATH=src uv run python -m scripts.run_research_05_vss_lrat
SLM_ENABLE_RESEARCH_05=1 PYTHONPATH=src uv run python -m scripts.run_research_05_vss_lrat --write
```

## Authority note

Filing or compiling this pilot is not evidence of production readiness.
Warm LRAT checking remains research-only; production refutation authority
stays EVID-09 exhaustive replay / checked certificate policy.
