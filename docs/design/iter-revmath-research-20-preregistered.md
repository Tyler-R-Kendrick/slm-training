# RESEARCH-20 — local cost kernel vs CSLib/Calf/AARA (SLM-552)

**Status:** preregistered evidence (reject)  
**Experiment key:** `RESEARCH-20`  
**Linear:** [SLM-552](https://linear.app/quickdeploy-ai/issue/SLM-552)  
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

Comparing the local event-trace cost kernel against CSLib/Calf/AARA clarifies
which cost claims transfer and which remain repository-local heuristics.

## Contract

| Arm | Role |
| --- | --- |
| Local event-trace cost kernel | control |
| Hermetic CSLib/Calf/AARA adapters | treatment |

| Gate | Result |
| --- | --- |
| Transferability disagreement rate | 0.5 |
| Silent equivalence claims | 0 |
| Incomparable metric smuggling | 4 |
| LOC burden delta vs local kernel | 161 |
| New bound classes enabled | 0 |
| LOC reduced vs local | False |
| Decision | **reject** |

Reason: `external_formalism_proof_burden_without_transfer_gain`.

## Corpus

- Frozen simulation spec: `src/slm_training/resources/formal/cost_kernel_compare_corpus.v1.json`
- Eval fixtures: 4

## Campaign lock

- Manifest sha256: `9e9b114697caace39a16b258e9925813265162601ee23228db5592e8322cf1ac`
- Lock artifact: `src/slm_training/resources/formal/research_20_campaign_lock.v1.json`
- Claim class: `fixture` (research pilot; no merge without transfer theorems)

## Run

```bash
PYTHONPATH=src uv run python -m scripts.run_research_20_cost_compare
SLM_ENABLE_RESEARCH_20=1 PYTHONPATH=src uv run python -m scripts.run_research_20_cost_compare --write
```

## Authority note

Hermetic formalism comparison only. External CSLib/Calf/AARA are not imported.
This rejection closes the **external-formalism merge** approach on frozen cost
fixtures: dependency/proof burden rises without LOC reduction or new bound classes.
Local event-trace kernel remains authoritative for represented counters.
