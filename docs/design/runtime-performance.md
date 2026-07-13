# Runtime performance notes

Measured on `twotower_v1_ship` (CPU, scratch context, LTR primary).

## Hotspots (before)

| Path | Cost | Notes |
| --- | --- | --- |
| LTR generate | ~93–111ms | Full canvas `T=96` every step until EOS (~47 tokens) |
| DESIGN.md lint (one-shot Node) | ~75ms / call | Eval re-linted every gold record |
| OpenUI validate / stream_check (REPL) | ~1.4ms | Already amortized via persistent REPL |
| Cactus / NEON kernel | n/a | **Kept separate** — not in the PyTorch loop |

## Simplifications / optimizations (this revision)

1. **Progressive LTR canvases** (`grammar_ltr_stages=(48, 96)`) — short programs finish on the 48-token stage and skip the long canvas.
2. **`torch.inference_mode()`** on `generate`; cached structural-bias vectors.
3. **DESIGN.md bridge REPL + hash cache** — same pattern as OpenUI; avoids cold Node per lint.
4. **Eval gold lint from `meta.design_lint`** — corpus already scored; no per-record Node in eval.
5. **Stream-check LRU** for constrained decode probes.
6. **Cactus adapter stays a thin boundary** (`slm_training.cactus`) — bundle/export/bench only; no kernel code in `models/`.

## Kernel boundary

```
prompt → TwoTower (PyTorch) → OpenUI text
                ↓ export_checkpoint_bundle
         portable .pt + tokenizer  →  (offline) cactus-compute transpile → .cact / NEON
```

Do not vendor or inline Cactus NEON kernels into this repo’s generate path.

## Re-bench

```bash
python -m scripts.bench_cactus --checkpoint outputs/runs/twotower_v1_ship/checkpoints/last.pt
```
