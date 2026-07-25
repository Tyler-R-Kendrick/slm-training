# SLM-218: cross-attention and subspace-retention retrospective

**Overall / H1 / H2:** `inconclusive` / `inconclusive` / `inconclusive`

**Report hash:** `04fa873a3615b0f695e0bea745bd968516092d5f2ac51ff13e93ba466cf14a72`

**Family manifest:** `9f2905e4a46b7393c86e7e3fdd74d5289bd34f5130b63df76168533aeed36907`

**Semantic floor:** `7839ef6b6e37710d487757da9170017d7b76a9d12ca1fb314bdb0fa23a4dd83d` (`inconclusive`)

## Coverage

| Family | Source | Declared children | Resolved local checkpoints | Complete | Exclusion |
| --- | --- | ---: | ---: | --- | --- |
| `context` | `docs/design/iter-e135-hf-context-control-20260715.json` | 0 | 0 | `false` | unresolved local-history checkpoints prevent matrix comparison |
| `context` | `docs/design/iter-e136-hf-context-32step-20260715.json` | 0 | 0 | `false` | unresolved local-history checkpoints prevent matrix comparison |
| `context` | `docs/design/iter-e138-hf-seed1-8step-20260715.json` | 0 | 0 | `false` | unresolved local-history checkpoints prevent matrix comparison |
| `context` | `docs/design/iter-e139-hf-seed2-8step-20260715.json` | 0 | 0 | `false` | unresolved local-history checkpoints prevent matrix comparison |
| `context` | `docs/design/iter-e176-broad-corpus-20260716.json` | 0 | 0 | `false` | unresolved local-history checkpoints prevent matrix comparison |
| `retention` | `docs/design/iter-e501-e396-e500-warm-start-20260719.json` | 4 | 0 | `false` | unresolved local-history checkpoints prevent matrix comparison |
| `retention` | `docs/design/iter-e502-initialization-prior-retention-20260719.json` | 4 | 0 | `false` | unresolved local-history checkpoints prevent matrix comparison |
| `retention` | `docs/design/iter-e503-initialized-weight-retention-20260719.json` | 4 | 0 | `false` | unresolved local-history checkpoints prevent matrix comparison |
| `retention` | `docs/design/iter-e504-parent-corpus-replay-20260719.json` | 5 | 0 | `false` | unresolved local-history checkpoints prevent matrix comparison |

## Verdict rationale

- all declared context-family checkpoints are unresolved local history
- retention families retain a durable parent reference but rejected child checkpoints are local-only and absent
- synthetic controls validate geometry but cannot rank historical outcomes
- SemanticFloorGateV1 is inconclusive; semantic interpretation is blocked

No cross-attention role is ranked and no retention target is nominated. The compact synthetic controls validate principal-angle, overlap, inside/outside update energy, Q/K orientation, context alignment, and activation-side restriction-energy formulas only.

No new training, checkpoint, semantic evaluation, causal intervention, optimizer change, promotion, or ship decision was performed.

## Reproduction

```bash
timeout 170s env PYTHONPATH=src .venv/bin/python -m scripts.run_cross_attention_retention --check
```
