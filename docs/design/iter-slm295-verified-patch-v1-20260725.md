# SLM-295 verified retrieval plus grammar-patch baseline

Status: local diagnostic; non-promotable

| Arm | n | Strict meaningful | Binder/ref F1 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `learned_rerank_fallback` | 1 | 0.0 | 0.0 | 3.9419059903593734 | 3.9419059903593734 |
| `oracle_topk_diagnostic` | 1 | 0.0 | 0.0 | 3.4330360067542642 | 3.4330360067542642 |
| `patching` | 1 | 0.0 | 0.0 | 3.0458789988188073 | 3.0458789988188073 |
| `top1` | 1 | 0.0 | 0.0 | 3.8551409961655736 | 3.8551409961655736 |

The index contains only verified training records and excludes exact, prompt, structure, split-group, and derivative overlaps. Oracle top-k is diagnostic-only. No checkpoint, human rating, remote replay, HF run, or ship claim is involved.
