# Research experiment preregistration registry (RESEARCH-02 / SLM-533)

**Status:** RESEARCH-02 — default-off / research-only preregistry  
**Base SHA:** `cae3ed70f713bda9b36dccac21cec0d8e5927d98` (`origin/main` at filing)  
**Machine-readable registry:** [`src/slm_training/resources/research_experiment_preregistry.json`](../../src/slm_training/resources/research_experiment_preregistry.json)  
**Loader:** [`src/slm_training/research_preregistry.py`](../../src/slm_training/research_preregistry.py)  
**Campaign owner:** [`ExperimentCampaignV1`](../../src/slm_training/autoresearch/experiment_campaign.py) / [experiment-campaign-governance.md](experiment-campaign-governance.md)  
**Citation grounding:** [research-citation-catalog.md](research-citation-catalog.md) (RESEARCH-01)  
**Verified by:** `PYTHONPATH=src uv run python -m scripts.verify_research_experiment_preregistry`

> **Authority law:** every registry entry is `default_off` and `research_only`.
> Filing or compiling an experiment is **not** evidence and never grants
> production decode, ship-gate, or serving authority.

## Purpose

Canonical preregistration surface for revmath / computability **RESEARCH-\***
pilots (RESEARCH-03 … RESEARCH-20) so each can be represented without a
bespoke runner. Execution still goes through the existing
`ExperimentCampaignV1` + `CampaignLockV1` owners.

| Field | Meaning |
| --- | --- |
| `experiment_key` | Stable `RESEARCH-NN` key |
| `hypothesis` | Claim under test (≥12 chars) |
| `cited_lineage` | RESEARCH-01 `source_id`s (or `repo-only-hypothesis`) |
| `repo_motivation` | Why this repo needs the pilot now |
| `blockers` | `{blocker_id,status,note}` — incomplete → fail closed |
| `matched_control` | Matched control arm description |
| `primary_metric` | Confirmatory primary metric id/name |
| `secondary_safety_metrics` | Safety / anti-overclaim metrics |
| `decision_rule` | Accept / reject / retire rule |
| `activation_preflight` | Must include blockers_complete, campaign_lock, citations_grounded, default_off |
| `resource_cap` | `max_run_minutes`, `max_gpu_hours`, `notes` |
| `falsifier` | What would kill the hypothesis |
| `evidence_path` | Durable docs/design evidence path pattern |
| `promotion_boundary` | Explicit non-promotion / research-only ceiling |
| `knobs` | Mechanism knobs (not the mechanism itself) |
| `knob_hypothesis_signature` | Stable sha256 over knobs + normalized hypothesis |
| `disposition` | `registered` / `blocked` / `executable` / `completed` / `rejected` / `reopened` |
| `campaign_lock_sha256` | Exact `CampaignLockV1` digest before run (null until locked) |

## Fail-closed gates

1. Missing control, falsifier, resource cap, citation, or blocker → registry reject.
2. `default_off` / `research_only` must be true on document and every entry.
3. Duplicate knob/hypothesis signatures rejected; equivalents of
   `completed` / `rejected` rows require `disposition=reopened` +
   `reopen_evidence_path`.
4. `assert_execution_allowed` requires complete/waived blockers **and** a
   64-hex campaign lock digest. No lock → no run.
5. Terminal dispositions cannot be re-executed.

## Pilot span

Registry seeds **RESEARCH-03 … RESEARCH-20** (eighteen pilots). RESEARCH-08 (SLM-540) fixture evidence: [iter-revmath-research-08-preregistered.md](iter-revmath-research-08-preregistered.md) (disposition `completed`, still `default_off` / no production authority). RESEARCH-05 (SLM-563) fixture evidence: [iter-revmath-research-05-preregistered.md](iter-revmath-research-05-preregistered.md) (disposition `completed`, still `default_off` / no production authority). RESEARCH-06 (SLM-564) fixture evidence: [iter-revmath-research-06-preregistered.md](iter-revmath-research-06-preregistered.md) (disposition `rejected` — correctness ok, ratio ≥ 1.0; still `default_off` / no production authority). Individual experimental mechanisms are **not** implemented here — only preregistration
contracts.

## Validation

```bash
PYTHONPATH=src uv run python -m scripts.verify_research_experiment_preregistry
```

## Ownership

Extends `experiment_campaign` + RESEARCH-01 citation catalog. No parallel
campaign store, evidence ledger, or runner.

## RESEARCH-05 status (SLM-563)

Preregistered VSS LRAT SAT pilot executed under campaign lock
`28f468da369df47c00779b5ce183e3e100443dd71afbc27bd47ed9a2e109010d`.
Disposition: **completed** (accept — correctness gates + warm/exhaustive ratio < 1.0).
Evidence: [`iter-revmath-research-05-preregistered.md`](iter-revmath-research-05-preregistered.md).
Still `default_off` / `research_only` — not production authority.

## RESEARCH-06 status (SLM-564)

Preregistered VSS PBLean/PB pilot executed under campaign lock
`66de4aee53e75b402b7a9d90e7ae72ac64bd14211c6a013b4afd579b7b3a924f`.
Disposition: **rejected** (correctness gates passed; median warm PBLean/exhaustive
ratio ≥ 1.0 on the frozen fixture suite — no Pareto win).
Evidence: [`iter-revmath-research-06-preregistered.md`](iter-revmath-research-06-preregistered.md).
Still `default_off` / `research_only` — not production authority.

**Successor approach (I14):** keep the hermetic encoding+mutation contract; next
attempt should either (a) enlarge the exhaustive cost foil / warm-cache amortization
so checked-refutation cost can beat enumeration on a meaningful subset, or
(b) integrate a real PBLean/VeriPB toolchain under the same default-off lock when
available — without widening production authority from fixture success alone.

