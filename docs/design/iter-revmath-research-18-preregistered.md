# RESEARCH-18 — anytime-valid adaptive promotion (SLM-551)

**Status:** preregistered evidence (accept)  
**Experiment key:** `RESEARCH-18`  
**Linear:** [SLM-551](https://linear.app/quickdeploy-ai/issue/SLM-551)  
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

Weihrauch/oracle classification of multi-valued proof/completion tasks
separates computable cores from oracle-relative remainders without Big-Five inflation.

## Contract

| Arm | Role |
| --- | --- |
| Total-computable collapse (control) | control |
| Weihrauch + practical classification | treatment |

| Gate | Result |
| --- | --- |
| Faithful Weihrauch assignment rate | 1 |
| Big-Five label inflation count | 0 |
| Oracle-relative smuggling count | 0 |
| Unknown collapse rate | 0.0 |
| Control misclassifies oracle tasks | True |
| Decision | **accept** |

Reason: `faithful_weihrauch_without_big_five_inflation`.

## Corpus

- Frozen simulation spec: `src/slm_training/resources/formal/weihrauch_multivalued_corpus.v1.json`
- Eval treatment tasks: 5

## Campaign lock

- Manifest sha256: `d5f0b411cc447948ef65437178f8211663287d54252e1beefc360cf9386126fc`
- Lock artifact: `src/slm_training/resources/formal/research_18_campaign_lock.v1.json`
- Claim class: `fixture` (research pilot; no production authority)

## Run

```bash
PYTHONPATH=src uv run python -m scripts.run_research_18_weihrauch_multivalued_classification
SLM_ENABLE_RESEARCH_18=1 PYTHONPATH=src uv run python -m scripts.run_research_18_weihrauch_multivalued_classification --write
```

## Authority note

Fixture Weihrauch/oracle taxonomy only. Filing is not production authority.
