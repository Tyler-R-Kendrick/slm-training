# Runtime performance notes

Measured on `twotower_v1_ship` (CPU, scratch context, LTR primary).

## Hotspots

| Path | Before (initial) | After round 1 | After round 2 |
| --- | --- | --- | --- |
| Single `generate` | ~111ms | ~71ms | **~63ms** |
| Eval 64× RICO (sequential) | — | — | ~4070ms |
| Eval 64× RICO (`generate_batch` 16) | — | — | **~1005ms (~4×)** |
| DESIGN.md lint (repeat) | ~75ms | ~0.005ms | same |
| Eval gold design lint | ~75ms×N | ~0 (meta) | same |
| Cactus / NEON kernel | n/a | separate | separate |

## Round 2 changes

1. **Batched LTR decode** (`generate_batch`) with progressive canvases; eval uses it automatically.
2. **Active-row subsetting** — finished EOS rows drop out of the transformer forward.
3. **Stages `(32, 48, 96)`** — shorter programs exit earlier.
4. **Skip Node finalize validate on generate** (eval already validates); optional via `grammar_finalize_validate`.
5. **OpenUI parse/validate result cache** in `lang_core`.

## Kernel boundary (unchanged)

```
prompt → TwoTower (PyTorch) → OpenUI text
                ↓ export_checkpoint_bundle
         portable .pt + tokenizer  →  (offline) cactus-compute → .cact / NEON
```

Do not vendor Cactus NEON kernels into `slm_training.models`.

## Re-bench

```bash
python -m scripts.bench_cactus --checkpoint outputs/runs/twotower_v1_ship/checkpoints/last.pt
```
