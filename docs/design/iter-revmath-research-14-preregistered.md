# RESEARCH-14 — checker diversity (SLM-572)

**Status:** preregistered evidence (accept)  
**Experiment key:** `RESEARCH-14`  
**Linear:** [SLM-572](https://linear.app/quickdeploy-ai/issue/SLM-572)  
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

Adding genuinely distinct checker implementations across independent EVID-08
trust domains increases seeded fault-class detection versus a single checker
family without false-alarm blow-up.

## Contract

| Arm | Role |
| --- | --- |
| Single structural-family checker | control |
| Structural + Lean kernel + encoding bridge | treatment |

| Gate | Result |
| --- | --- |
| Unique defect detection gain | 0.666667 |
| Unique fault-class gain count | 4 |
| Control detected classes | ['parser_proposition', 'parser_toolchain'] |
| Treatment detected classes | ['encoding_digest', 'encoding_literal', 'kernel_axiom', 'kernel_binding', 'parser_proposition', 'parser_toolchain'] |
| False alarm rate | 0.0 |
| Shared blind-spot residual | 1 |
| Decision | **accept** |

Reason: `diverse_checkers_add_unique_fault_classes`.

## Corpus

- Frozen cases: `src/slm_training/resources/formal/checker_diversity_corpus.v1.json`
- EVID-08 trust domains: structural, lean4_kernel, encoding_bridge
- EVID-11-aligned seeded fault families (fixture simulation)

## Campaign lock

- Manifest sha256: `2f3969f98f653574496fc50fcaba6c9b790d22df3f663c98f43459b5042492c2`
- Lock artifact: `src/slm_training/resources/formal/research_14_campaign_lock.v1.json`
- Claim class: `fixture` (research pilot; no promotion)

## Run

```bash
PYTHONPATH=src uv run python -m scripts.run_research_14_checker_diversity
SLM_ENABLE_RESEARCH_14=1 PYTHONPATH=src uv run python -m scripts.run_research_14_checker_diversity --write
```

## Authority note

Fixture trust-domain simulation only — no mandatory external checker dependency.
Filing is not production readiness.
