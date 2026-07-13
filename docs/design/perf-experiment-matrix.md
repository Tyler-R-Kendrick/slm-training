# Inference-speed experiment matrix (P-series)

Decode-only optimizations for TwoTower LTR grammar-constrained generate.
No retraining — each row overlays flags on an existing checkpoint (default:
`fixtures/checkpoints/playground_demo/last.pt`).

See also: [runtime-performance.md](runtime-performance.md).

## Why CPU generate felt like ~3.4s/line

The demo denoiser is tiny; the cost is structural:

1. **O(T²) grammar work** — every token rebuilt a Lark `InteractiveParser`,
   re-lexed the full prefix, and re-`tokenizer.decode`d the prefix ids
   (often 2–3× per token via force-emit + pick + per-candidate admits).
2. **Up to top-k Node/Lark `stream_check` probes per token** on unique growing
   prefixes (SHA256 cache rarely hits).
3. **O(T) full-canvas denoiser forwards** with no KV reuse.
4. **Playground multipliers** — repair + finalize + up to 3 generate attempts.

## Experiments

| ID | Lever | Expected effect |
| --- | --- | --- |
| **P0** | Baseline (legacy grammar state, no P2–P7) | Attribution reference |
| **P1** | Persistent DFA + prefix-text cache (`grammar_incremental_state`) | Biggest grammar win; bit-exact intent |
| **P2** | Verify-chosen-only + skip exact DFA stream probes | Fewer Node/Lark probes |
| **P3** | Multi-token accept per forward (`grammar_multitoken_accept`) | Fewer denoiser forwards |
| **P4** | Prefix+K lookahead (`grammar_canvas_lookahead=32`) | Cheaper attention per forward |
| **P5** | Dynamic int8 Linear quant (`use_dynamic_quant`) | CPU matmul speedup |
| **P6** | MaskGIT-primary (`grammar_ltr_primary=False`) | Quality/latency tradeoff |
| **P7** | Playground budget (`generate_max_attempts=1`, finalize-last-only) | Caps worst-case retries |
| **P8** | Combo P1+P2+P3+P4 | Shippable recipe candidate |

## How to run

```bash
# Phase breakdown on the demo checkpoint
python -m scripts.profile_generate --rounds 2 --out outputs/runs/profile_generate.json

# Full P-matrix (smoke prompts, quality guardrails vs P0)
python -m scripts.run_perf_matrix --limit 8 --out-dir outputs/runs/perf_matrix

# Subset
python -m scripts.run_perf_matrix --only P0,P1,P8 --limit 4
```

Results land in `outputs/runs/perf_matrix/scoreboard.json` and
`docs/design/perf-matrix-results.json`.

## Guardrails

An optimization **fails** its gate when parse rate or placeholder fidelity drops
more than 5 points absolute vs P0 on the same prompt set. P3/P4/P6 may change
outputs; P1 is expected to stay bit-exact on force-emit / DFA-agreed paths.

## Measured results (CPU, playground demo, `--limit 4`)

Captured in [`perf-matrix-results.json`](perf-matrix-results.json):

| ID | latency_ms_mean | speedup vs P0 | forwards/call | probes/call | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| P0 | 1319 | 1.00× | 65.5 | 2039 | Legacy grammar state |
| P1 | 1304 | 1.01× | 65.5 | 2039 | Incremental DFA; admits still dominate |
| P2 | 479 | **2.75×** | 65.5 | 332 | Verify-chosen cuts stream probes |
| P3 | 466 | **2.83×** | **8.5** | 204 | Multi-token accept |
| P4 | 807 | 1.64× | 43.2 | 1625 | Lookahead=32 |
| P5 | 959 | 1.38× | 65.5 | 1245 | Dynamic int8 |
| P6 | 3385 | 0.39× | 73.0 | 4183 | MaskGIT+grammar slower here |
| P7 | 4819 | 0.27× | 357.5 | 16370 | Repair path ≈ playground 3.4s culprit |
| P8 | 542 | **2.43×** | 9.5 | 113 | P1+P2+P3+P4 combo |

Takeaways:

- **Playground slowness is repair amplification (P7)**, not the tiny denoiser.
- **P2 + P3** are the highest-leverage decode levers on CPU.
- **P1** removes O(T²) prefix re-lex for the committed path; remaining DFA cost
  is mostly throwaway admit probes on broad `COMPONENT`/`NAME` terminals — P2
  avoids most of those by accepting a legal argmax early.

## Config flags (`TwoTowerConfig`)

| Flag | Default | Notes |
| --- | --- | --- |
| `grammar_incremental_state` | `True` | P1 on by default after this change |
| `grammar_verify_chosen_only` | `False` | P2 |
| `grammar_skip_exact_stream_probe` | `True` | Skip Node probes when DFA terminals are exact |
| `grammar_multitoken_accept` | `False` | P3 |
| `grammar_multitoken_max` | `8` | Max run length per forward |
| `grammar_canvas_lookahead` | `0` | P4; `0` disables |
| `use_dynamic_quant` | `False` | P5 |
| `generate_max_attempts` | `3` | P7 playground budget |
| `grammar_finalize_on_last_attempt_only` | `False` | P7 |

## Instrumentation

`model.generate_with_stats(prompt) -> (text, DecodeStats)` records
`denoiser_ms`, `dfa_sync_ms`, `stream_check_ms`, `detok_ms`, `context_ms`,
`pick_ms`, `forwards_count`, `probes_count`, `tokens_emitted`,
`accepted_run_tokens`.
