# RESEARCH-16 — rational-to-float transfer certificates (SLM-570)

**Status:** preregistered evidence (accept)  
**Experiment key:** `RESEARCH-16`  
**Linear:** [SLM-570](https://linear.app/quickdeploy-ai/issue/SLM-570)  
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

Exact rational reference plus conservative float interval certificates can
bound selected metric/gate evaluations without silent cast-induced flips.

## Contract

| Arm | Role |
| --- | --- |
| Direct float64 gate comparison | control |
| Exact rational + interval certificate | treatment |

| Gate | Result |
| --- | --- |
| Transfer certificate coverage | 1 |
| Float flip count | 1 |
| Float flip detections (indeterminate) | 1 |
| Silent cast incidents | 0 |
| Decision | **accept** |

Reason: `certificates_cover_selected_gates`.

## Corpus

- Frozen gate cases: `src/slm_training/resources/formal/rational_float_gate_corpus.v1.json`
- EVID-04 exact Fraction semantics; float64 ULP enclosures

## Campaign lock

- Manifest sha256: `a07c24e7905e58b5b9469089e241b10e14e9d41dfd23c58c769028b2496dd8df`
- Lock artifact: `src/slm_training/resources/formal/research_16_campaign_lock.v1.json`
- Claim class: `fixture` (research pilot; no promotion)

## Run

```bash
PYTHONPATH=src uv run python -m scripts.run_research_16_rational_float
SLM_ENABLE_RESEARCH_16=1 PYTHONPATH=src uv run python -m scripts.run_research_16_rational_float --write
```

## Authority note

Fixture gate simulation only. Filing is not production readiness.
