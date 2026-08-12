# RESEARCH-12 — successor-theorem semantic validation (SLM-567)

**Status:** preregistered evidence (accept)  
**Experiment key:** `RESEARCH-12`  
**Linear:** [SLM-567](https://linear.app/quickdeploy-ai/issue/SLM-567)  
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

Successor-theorem semantic validation catches generated theorem statements that
compile but change meaning relative to the parent obligation.

## Contract

| Arm | Role |
| --- | --- |
| Compile-only acceptance | control |
| Successor + fingerprint + counterexample validation | treatment |

| Gate | Result |
| --- | --- |
| Semantic catch rate on meaning-changing successors | 1 |
| False-positive block rate | 0.0 |
| Successor incremental catches | 1 |
| Decision | **accept** |

Reason: `successor_validation_catches_subtle_meaning_change`.

## Corpus

- Frozen theorem records: `src/slm_training/resources/formal/successor_theorem_corpus.v1.json`

## Campaign lock

- Manifest sha256: `be369585113325f5303fc7e09939991fe15a1fb6c3a9125f8db3a1fed3573c51`
- Lock artifact: `src/slm_training/resources/formal/research_12_campaign_lock.v1.json`
- Claim class: `fixture` (research pilot; no promotion)

## Run

```bash
PYTHONPATH=src uv run python -m scripts.run_research_12_successor_theorem
SLM_ENABLE_RESEARCH_12=1 PYTHONPATH=src uv run python -m scripts.run_research_12_successor_theorem --write
```

## Authority note

Fixture semantic-validation pilot only. Filing is not production readiness.
