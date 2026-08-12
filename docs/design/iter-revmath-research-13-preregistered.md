# RESEARCH-13 — formal counterexample under dropped assumptions (SLM-568)

**Status:** preregistered evidence (accept)  
**Experiment key:** `RESEARCH-13`  
**Linear:** [SLM-568](https://linear.app/quickdeploy-ai/issue/SLM-568)  
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

Bounded counterexample synthesis with independent HARN-06 validation produces
checked finite models under dropped assumptions without promoting search failure
to refutation.

## Contract

| Arm | Role |
| --- | --- |
| Proof-search terminal only | control |
| Bounded counterexample + independent validation | treatment |

| Gate | Result |
| --- | --- |
| Independently checked finite-model rate | 1 |
| Search failure as refutation | 0 |
| Model-check disagreements | 0 |
| Decision | **accept** |

Reason: `checked_finite_models_without_search_failure_refutation`.

## Corpus

- Paired task records: `src/slm_training/resources/formal/dropped_assumption_tasks.v1.json`
- Revmath fixtures: `src/slm_training/resources/revmath/fixtures`

## Campaign lock

- Manifest sha256: `e04e671c1269a91b5f406cc5870233d8eb505f201e5093107310217423574a05`
- Lock artifact: `src/slm_training/resources/formal/research_13_campaign_lock.v1.json`
- Claim class: `fixture` (research pilot; no promotion)

## Run

```bash
PYTHONPATH=src uv run python -m scripts.run_research_13_dropped_assumption
SLM_ENABLE_RESEARCH_13=1 PYTHONPATH=src uv run python -m scripts.run_research_13_dropped_assumption --write
```

## Authority note

Fixture counterexample pilot only. Filing is not production readiness.
