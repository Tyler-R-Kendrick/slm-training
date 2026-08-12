# RESEARCH-11 — process-verified reward shaping (SLM-574)

**Status:** preregistered evidence (reject)  
**Experiment key:** `RESEARCH-11`  
**Linear:** [SLM-574](https://linear.app/quickdeploy-ai/issue/SLM-574)  
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

Verifier-grounded process rewards from checked tactic prefixes and failure
localization improve learning signal correlation vs terminal-only rewards.

## Contract

| Arm | Role |
| --- | --- |
| Terminal-only proof success reward | control |
| Process-verified dense shaping | treatment |

| Gate | Result |
| --- | --- |
| Process reward correlation | 0.9977 |
| Correlation gain vs terminal | -0.0023004401300430466 |
| Unverifiable intermediate rate | 0.0 |
| Reward hacking incidents | 0 |
| Decision | **reject** |

Reason: `process_reward_no_correlation_gain`.

## Corpus

- Frozen proof attempts: `src/slm_training/resources/formal/process_verified_reward_corpus.v1.json`

## Campaign lock

- Manifest sha256: `bf26ab9c01c38088aeedb004505428af57339c12db60a42108ecf0326e27f6bb`
- Lock artifact: `src/slm_training/resources/formal/research_11_campaign_lock.v1.json`
- Claim class: `fixture` (research pilot; no promotion)

## Run

```bash
PYTHONPATH=src uv run python -m scripts.run_research_11_process_reward
SLM_ENABLE_RESEARCH_11=1 PYTHONPATH=src uv run python -m scripts.run_research_11_process_reward --write
```

## Authority note

Fixture reward simulation only. Filing is not production readiness.
