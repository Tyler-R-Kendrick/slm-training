# RESEARCH-19 — anytime-valid adaptive promotion (SLM-558)

**Status:** preregistered evidence (accept)  
**Experiment key:** `RESEARCH-19`  
**Linear:** [SLM-558](https://linear.app/quickdeploy-ai/issue/SLM-558)  
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

Lean-checked Rademacher/generalization certificates on a toy hypothesis class
clarify assumptions before any provable-training claim.

## Contract

| Arm | Role |
| --- | --- |
| Narrative generalization claim | control |
| Lean-exported Rademacher certificate | treatment |

| Gate | Result |
| --- | --- |
| Certificate validity rate | 1 |
| Decision usefulness rate | 1.0 |
| Ship-gate authority smuggling | 0 |
| Bound vacuity count | 0 |
| Assumption under-declaration | 0 |
| Decision | **accept** |

Reason: `lean_checked_toy_bound_is_decision_useful`.

## Corpus

- Frozen simulation spec: `src/slm_training/resources/formal/lean_rademacher_corpus.v1.json`
- Eval cases: 2

## Campaign lock

- Manifest sha256: `f9c927b9594902aada494180215e81dbd72751a8a3f4d9f9a86b2d1e66e7aa24`
- Lock artifact: `src/slm_training/resources/formal/research_19_campaign_lock.v1.json`
- Claim class: `fixture` (research pilot; no promotion)

## Run

```bash
PYTHONPATH=src uv run python -m scripts.run_research_19_lean_rademacher_bound
SLM_ENABLE_RESEARCH_19=1 PYTHONPATH=src uv run python -m scripts.run_research_19_lean_rademacher_bound --write
```

## Authority note

Toy statistical certificate only. Filing is not ship-gate authority.
