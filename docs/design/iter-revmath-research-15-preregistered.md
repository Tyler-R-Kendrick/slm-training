# RESEARCH-15 — anytime-valid adaptive promotion (SLM-569)

**Status:** preregistered evidence (accept)  
**Experiment key:** `RESEARCH-15`  
**Linear:** [SLM-569](https://linear.app/quickdeploy-ai/issue/SLM-569)  
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

Anytime-valid e-process evidence sequences can govern adaptive autoresearch
promotion without invalidating type-I control under optional stopping.

## Contract

| Arm | Role |
| --- | --- |
| Holm gate at adaptive peeks | control |
| Mixture e-process confidence sequence | treatment |

| Gate | Result |
| --- | --- |
| Type-I control score | 1 |
| Control null promotion rate | 0.055 |
| Treatment null promotion rate | 0.02 |
| Treatment planted power | 0.42 |
| Fixed-n oracle power | 0.38 |
| Power loss vs oracle | 0.0 |
| Decision | **accept** |

Reason: `anytime_valid_controls_optional_stopping_with_usable_power`.

## Corpus

- Frozen simulation spec: `src/slm_training/resources/formal/anytime_valid_promotion_corpus.v1.json`
- Null scenarios: 200 (p=0.5)
- Planted scenarios: 50 (p=0.65)

## Campaign lock

- Manifest sha256: `e98337a6f0aea62a62559383deacbdcff9051e42dfa833747d7b587011e1a75c`
- Lock artifact: `src/slm_training/resources/formal/research_15_campaign_lock.v1.json`
- Claim class: `fixture` (research pilot; no promotion)

## Run

```bash
PYTHONPATH=src uv run python -m scripts.run_research_15_anytime_valid_promotion
SLM_ENABLE_RESEARCH_15=1 PYTHONPATH=src uv run python -m scripts.run_research_15_anytime_valid_promotion --write
```

## Authority note

Fixture sequential-inference simulation only. Filing is not production readiness.
