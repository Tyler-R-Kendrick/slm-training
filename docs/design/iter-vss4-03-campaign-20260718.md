# VSS4-03 — Execute the matched verified-scope-solver campaign (SLM-76)

**Date:** 2026-07-18. **Status:** CPU/fixture wiring campaign executed and
published; frontier phases (2-4, 6) and matrix rows R2-R6 are **honestly blocked**
because required artifacts are not present in this repository state. **No model,
no quality claim, no ship/default-on decision.**

## Why this exists

SLM-76 asks for the end-to-end VSS4-02 campaign to be executed with replayable
evidence. The repository already has the VSS4-01 closed benchmark (SLM-74) and
the VSS4-02 matched matrix schema + CPU fixture runner (SLM-75). What it does
not yet have are the trained checkpoints and benchmark families needed for the
model-backed rows:

- `twotower_ranker_checkpoint` (R2/R3)
- `capsule_benchmark_family_c` (R3/R4/R5)
- `cost_to_go_energy_checkpoint` (R4)
- `surface_benchmark_family_e` (R5/R6)
- `surface_ar_checkpoint` (R6)

Rather than silently substitute weaker configs or fabricate frontier numbers,
this iteration adds a campaign runner that executes every CPU-runnable phase,
records the artifact lock, and marks every blocked item with an explicit,
reproducible reason.

## Landed in this iteration

- `src/slm_training/harnesses/experiments/vss4_campaign.py`
  - `ArtifactLock`, `PhaseResult`, `CampaignReport` dataclasses with deterministic
    JSON serialization.
  - Phase runners: artifact lock (0), VSS4-01 correctness reference (1),
    VSS4-02 fixture matrix (5).
  - Honest blocked phases: on-policy supervision (2), energy training (3),
    surface training (4), adversarial/OOD (6).
- `scripts/run_vss4_campaign.py`
  - CLI: `--describe` and `--out-dir`; writes `campaign.json` + `campaign.md`.
- `tests/test_harnesses/experiments/test_vss4_campaign.py`
  - JSON stability, artifact lock, fixture phases pass gates, frontier phases
    blocked with reasons, no silent downgrade of R2-R6.
- `docs/design/vss4-03-campaign-results.json`
  - Durable evidence bundle from the fixture run.
- `outputs/runs/vss4_03_campaign/campaign.{json,md}`
  - Run artifact for this execution.

## Fixture results

- **Phase 1 — VSS4-01 correctness reference:** passed, manifest digest
  `1b0a2754d25d4c8d`, zero false unsupported / unknown preservation / replay
  failures.
- **Phase 5 — VSS4-02 matched matrix:** R0 and R1 ran on CPU; hard gates pass;
  R2-R6 blocked with required-capability reasons.

## Verification

```bash
python -m pytest tests/test_harnesses/experiments/test_vss4_campaign.py -q
python -m scripts.run_vss4_campaign --describe
python -m scripts.run_vss4_campaign --out-dir outputs/runs/vss4_03_campaign
python -m scripts.run_solver_bench --all
python -m scripts.run_verified_solver_matrix --fixture
python -m scripts.repo_policy
```

## Honest boundary / remaining scope

- **Blocked now (requires future work):**
  - On-policy solver-trace collection on a train split (needs trainable checkpoint
    running solver-guided decode).
  - Cost-to-go energy ranker training CLI / integration.
  - Surface AR realizer training CLI / integration.
  - Capsule-aware benchmark family C and surface-realization benchmark family E.
  - Solver-relevant adversarial/OOD suites.
- **When unblocked:** re-run `scripts/run_vss4_campaign.py` with the artifacts
  resolved; the runner will promote phases from `blocked` to `ran` and populate
  the corresponding evidence fields.
