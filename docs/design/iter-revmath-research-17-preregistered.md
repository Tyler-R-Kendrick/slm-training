# RESEARCH-17 — proof-mined quantitative bounds (SLM-557)

**Status:** preregistered evidence (accept)  
**Experiment key:** `RESEARCH-17`  
**Linear:** [SLM-557](https://linear.app/quickdeploy-ai/issue/SLM-557)  
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

Proof-mining-style transformation of qualitative project theorems exposes
replayable quantitative bounds validated through HARN-07 and the safe bound AST.

## Contract

| Arm | Role |
| --- | --- |
| Qualitative-only theorem statements | control |
| HARN-07 proof-mined bound extraction | treatment |

| Gate | Result |
| --- | --- |
| Actionable bound extraction success | 1 |
| Spurious constant rate | 0.0 |
| Bound uselessness rate | 0.0 |
| Decision | **accept** |

Reason: `proof_mining_yields_replayable_bounds`.

## Corpus

- Frozen theorem records: `src/slm_training/resources/formal/proof_mined_theorems.v1.json`
- Revmath fixtures: `src/slm_training/resources/revmath/fixtures`

## Campaign lock

- Manifest sha256: `96a76aa3fbe098392f8befcab033fec5686345d32b97adb450c5307281cb9531`
- Lock artifact: `src/slm_training/resources/formal/research_17_campaign_lock.v1.json`
- Claim class: `fixture` (research pilot; no promotion)

## Run

```bash
PYTHONPATH=src uv run python -m scripts.run_research_17_proof_mined
SLM_ENABLE_RESEARCH_17=1 PYTHONPATH=src uv run python -m scripts.run_research_17_proof_mined --write
```

## Authority note

Fixture theorem mining only. Filing is not production readiness.
