# RESEARCH-10 — proof-strength curriculum (SLM-566)

**Status:** preregistered evidence (accept)  
**Experiment key:** `RESEARCH-10`  
**Linear:** [SLM-566](https://linear.app/quickdeploy-ai/issue/SLM-566)  
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

A proof-strength curriculum ordered by formal obligation difficulty improves
sample efficiency versus unordered mixing on dependency-disjoint eval groups.

## Contract

| Arm | Role |
| --- | --- |
| Unordered mixed-strength schedule (matched budget) | control |
| Obligation-difficulty curriculum | treatment |

| Gate | Result |
| --- | --- |
| Sample efficiency ratio (treatment/control) | 1.33333 |
| Control eval pass rate | 0.5 |
| Treatment eval pass rate | 0.6666666666666666 |
| Train/eval root leakage | False |
| Easy-task regression | False |
| Decision | **accept** |

Reason: `curriculum_improves_sample_efficiency`.

## Corpus

- Frozen records: `src/slm_training/resources/formal/proof_strength_records.v1.json`
- Obligation kinds: original, ablation, reversal, constructivization, counterexample, bound
- HARN-11-style root-family isolation between train and eval

## Campaign lock

- Manifest sha256: `4eb1a8b95a88c54ecf5d80706f9aacf09e05988ba3c5fb5de99856d1a220266b`
- Lock artifact: `src/slm_training/resources/formal/research_10_campaign_lock.v1.json`
- Claim class: `fixture` (research pilot; no promotion)

## Run

```bash
PYTHONPATH=src uv run python -m scripts.run_research_10_proof_strength
SLM_ENABLE_RESEARCH_10=1 PYTHONPATH=src uv run python -m scripts.run_research_10_proof_strength --write
```

## Authority note

Fixture schedule proxy only — no trainer invoked. Filing is not production readiness.
