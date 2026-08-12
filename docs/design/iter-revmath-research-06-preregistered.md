# RESEARCH-06 — VSS pseudo-Boolean backend with PBLean (SLM-564)

**Status:** preregistered evidence (reject)  
**Experiment key:** `RESEARCH-06`  
**Linear:** [SLM-564](https://linear.app/quickdeploy-ai/issue/SLM-564)  
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

For bounded VSS subsets dominated by cardinality-style constraints, verified
VSS→PB + checked PBLean/VeriPB-style certificates preserve exact semantic
agreement / mutation rejection while making warm certificate-check cheaper
than exhaustive replay.

## Contract

| Arm | Role |
| --- | --- |
| Exhaustive VSS/CNF replay | matched control |
| Deterministic CNF→PB + PBLean via EVID-10 `pblean_pilot` | treatment |
| RESEARCH-05 LRAT (when comparable) | cross-check |

| Gate | Result |
| --- | --- |
| Witness disagreements | 0 |
| Exact semantic agreement (supported PB subset) | 1.0 |
| Mutation rejection rate | 1.0 |
| Supported-subset coverage | 1.0 |
| LRAT comparable / agree | 4 / 4 |
| Median warm PBLean / exhaustive ratio | 1.07199 |
| Decision | **reject** |

Decision rule: correctness gates pass **and** ratio < 1.0; else reject/retire.
Timeout / tool failure / unsupported / missing external PBLean → `unknown`
(candidates preserved; fixture stub never fabricates success).

## Campaign lock

- Manifest sha256: `66de4aee53e75b402b7a9d90e7ae72ac64bd14211c6a013b4afd579b7b3a924f`
- Lock artifact: `src/slm_training/resources/formal/research_06_campaign_lock.v1.json`
- Claim class: `fixture` (research pilot; no promotion)

## Toolchain pin

```json
{
  "backend_id": "vss_pblean_pb_pilot",
  "certificate_format": "pblean_pilot",
  "certificate_schema": "vss_pblean_certificate/v1",
  "checker_backend": "python_pblean_veripb_pilot",
  "encoder_family": "cnf_ref",
  "encoder_hash": "0752e7e4570417794e8a3cbed145d659438cd055170e59d05ddcfe043bc4e270",
  "encoder_version": "cnf_ref_v1",
  "external_pblean": "optional_unavailable_fixture_stub",
  "toolchain_id": "hermetic_python_pblean_pilot_v1",
  "trust_domain": "encoding_bridge/cnf_ref+pblean_pilot"
}
```

## Artifacts

| Artifact | Path |
| --- | --- |
| Results JSON | [`iter-revmath-research-06-preregistered.json`](iter-revmath-research-06-preregistered.json) |
| Adapter | `src/slm_training/formal/vss_pblean_backend.py` |
| Experiment | `src/slm_training/harnesses/experiments/research_06_vss_pblean.py` |
| Encoding bridge | `src/slm_training/formal/encoding_adapter.py` (EVID-10) |

## Run

```bash
# default-off: refuses without enable
PYTHONPATH=src uv run python -m scripts.run_research_06_vss_pblean
SLM_ENABLE_RESEARCH_06=1 PYTHONPATH=src uv run python -m scripts.run_research_06_vss_pblean --write
```


## Successor approach (I14)

This rejection closes the **warm-cost Pareto** approach on the current fixture
foil — not the goal of a checked PB certificate path. Correctness gates passed
(zero disagreements, 100% mutation rejection, LRAT cross-check agree 4/4).

Next approach (file under the same evidence lineage when reopened):
1. Enlarge exhaustive cost foil / amortize warm certificate checks so median
   warm/exhaustive ratio can clear `< 1.0` on a meaningful supported subset; or
2. Wire an optional real PBLean/VeriPB binary behind the same default-off lock
   when available — missing tools remain `unknown`, never fabricated success.

Still `default_off` / research-only; no production authority.

## Authority note

Filing or compiling this pilot is not evidence of production readiness.
Warm PBLean checking remains research-only; production refutation authority
stays EVID-09 exhaustive replay / checked certificate policy. External PBLean
absence is handled via the hermetic fixture stub — still fail-closed / default-off.
