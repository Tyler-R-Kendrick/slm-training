# Inference-speed experiment matrix (P/Q/R-series)

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
| **P0** | Baseline (legacy grammar state, no P2–Q2) | Attribution reference |
| **P1** | Persistent DFA + prefix-text cache (`grammar_incremental_state`) | Committed-path reuse |
| **P2** | Verify-chosen-only + skip exact DFA stream probes | Fewer Node/Lark probes |
| **P3** | Multi-token accept per forward (`grammar_multitoken_accept`) | Fewer denoiser forwards |
| **P4** | Prefix+K lookahead (`grammar_canvas_lookahead=32`) | Cheaper attention per forward |
| **P5** | Dynamic int8 Linear quant (`use_dynamic_quant`) | CPU matmul speedup |
| **P6** | MaskGIT-primary (`grammar_ltr_primary=False`) | Quality/latency tradeoff |
| **P7** | Playground budget (`generate_max_attempts=1`, finalize-last-only) | Caps worst-case retries |
| **P8** | Combo P1+P2+P3+P4 | Pre-Q recipe |
| **Q1** | `InteractiveParser.copy()` admit probes + memo (`grammar_copy_probes`) | Cheaper per-candidate DFA |
| **Q2** | Whitespace fast-admit + early-exit pick (`grammar_early_exit_pick`) | Fewer candidates scored |
| **Q9** | P8 + Q1 + Q2 | Pre-R shippable recipe |
| **R1** | Skip `dfa_admits` when tid already in exact DFA `allowed` | Fewer copy/throwaway probes |
| **R2** | Skip redundant `set_prefix` when engine already synced | Fewer no-op re-lexes |
| **R4** | Repair/BOS fill uses multitoken + lookahead | Repair forwards ≈ greedy LTR |
| **R5** | Wire `generate_max_attempts`; skip redundant BOS ensure | Caps playground repair×N |
| **R9** | Q9 + R1/R2 (decode recipe) | New shippable decode path |
| **PG** | R9 levers + repair/finalize (R4+R5) | Real playground path |

## How to run

```bash
# Phase breakdown on the demo checkpoint
python -m scripts.profile_generate --rounds 2 --out outputs/runs/profile_generate.json

# Full matrix (smoke prompts, quality guardrails vs P0)
python -m scripts.run_perf_matrix --limit 8 --out-dir outputs/runs/perf_matrix

# Subset
python -m scripts.run_perf_matrix --only P0,Q9,R9,PG --limit 4
```

Results land in `outputs/runs/perf_matrix/scoreboard.json` and
`docs/design/perf-matrix-results.json`.

## Guardrails

An optimization **fails** its gate when parse rate or placeholder fidelity drops
more than 5 points absolute vs P0 on the same prompt set. The scoreboard records
`bridge_available` and `quality_pipeline_ok`; if a known-good OpenUI snippet
fails `validate()`, the run exits with a **vacuous guardrail** error (exit 2)
so a broken Node bridge cannot silently zero all parse rates.

Quality uses the same meaningful-program check as eval (`_is_meaningful_program`).

## Round 1 measured results (CPU, playground demo, `--limit 4`)

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

Round-1 takeaway: after P8, **`dfa_sync_ms` was 75% of remaining latency**
(~29 throwaway full-prefix admits per token).

## Round 2 measured results (CPU, bridge up, `--limit 4`)

| ID | latency_ms_mean | speedup vs P0 | dfa_ms | forwards | probes | dfa_syncs | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| P0 | 941 | 1.00× | 318 | 42.5 | 1248 | 1802 | Legacy baseline |
| P8 | 619 | 1.52× | 311 | 13.5 | 180 | 2176 | Pre-Q combo |
| Q1 | — | <1× alone | high | — | — | — | Copy probes alone still re-lex full prefix |
| Q2 | 622 | 1.51× | **89** | 114 | 90 | 809 | Early-exit cuts candidates hard |
| **Q9** | **294** | **3.20×** | **112** | **13.0** | **77** | **778** | P8+Q1+Q2 shippable recipe |
| PG | 2135 | 0.44× | 442 | 469 | 88 | 4391 | Repair path; was ~4820ms in P7 (**~2.3×** faster) |

Round-2 takeaways:

- **Q9 is the pre-R ship recipe** (~3.2× vs P0; ~2× vs P8).
- **Q2 early-exit** is the main remaining probe reducer; Q1 copy probes pay off
  once candidate count is already low (combo, not alone).
- **Playground repair path** dropped from ~4.8s → ~2.1s with Q9 levers
  (raw_syntax_rate=1.0, parse_rate=0.25 on the tiny demo ckpt).
- Playground service now enables Q9 flags by default
  (`grammar_verify_chosen_only`, `grammar_multitoken_accept`,
  `grammar_copy_probes`, `grammar_early_exit_pick`, `grammar_canvas_lookahead=32`).

## Round 3 — R-series (admit skip + repair budget)

After Q9, remaining hotspots were:

1. **Redundant `dfa_admits`** even when `tid` was already in the exact DFA
   `allowed` set (copy-probe / throwaway re-lex).
2. **Redundant `set_prefix`** in pick/force-emit when P1 `advance_token` already
   left the engine synced.
3. **Repair ignored P3/P4** — `_constrained_ltr_repair` did 1 forward/token
   (~114) while greedy LTR did ~13.
4. **`generate_max_attempts` unused** by `_ensure_valid_openui` (defaulted to 3
   BOS redos on top of `_repair_ltr_texts`) → PG ~469 forwards.

| ID | Lever | Notes |
| --- | --- | --- |
| R1 | Skip admit when exact-allowed | Always-on in `pick_constrained_token` |
| R2 | Skip synced `set_prefix` | pick / force_emit / admit |
| R4 | Repair uses multitoken+lookahead | Same forward budget as greedy LTR |
| R5 | Wire attempt budget; skip redundant ensure | PG with attempts=1 → 0 extra BOS |
| R9 | Q9 + R1/R2 | New decode recipe |
| PG | R4+R5 on playground path | Repair+finalize with R-series |

### Round 3 measured results (CPU, bridge up, `--limit 4`)

| ID | latency_ms_mean | speedup vs P0 | forwards | probes | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| P0 | 862 | 1.00× | 42.5 | 1109 | Legacy baseline (this host) |
| Q9 | 332 | 2.60× | 13.8 | 76 | Pre-R recipe (R1/R2 code present) |
| **R9** | **302** | **2.85×** | **13.8** | **0** | Exact-admit skip zeros stream probes |
| **PG** | **520** | **1.66×** | **26.8** | 63 | Was ~2135ms / 469 fwd (**~4.1×** vs round-2 PG) |

Round-3 takeaways:

- **PG is the headline win** — repair now shares P3/P4 with greedy LTR, and
  `generate_max_attempts=1` + prior `_repair_ltr_texts` skips a redundant BOS
  ensure (forwards 469 → ~27).
- **R9** trims decode further mainly by skipping stream probes on exact DFA
  terminals (probes 76 → 0); latency ~9% under Q9 on this host.
- New ship recipe: **R9** for decode-only; **PG flags** for playground.

```bash
python -m scripts.run_perf_matrix --only P0,Q9,R9,PG --limit 4
```

## Config flags (`TwoTowerConfig`)

| Flag | Default | Notes |
| --- | --- | --- |
| `grammar_incremental_state` | `True` | P1 |
| `grammar_verify_chosen_only` | `False` | P2 (playground forces True) |
| `grammar_skip_exact_stream_probe` | `True` | Skip Node probes when DFA terminals are exact |
| `grammar_copy_probes` | `True` | Q1 |
| `grammar_early_exit_pick` | `True` | Q2 |
| `grammar_multitoken_accept` | `False` | P3 (playground forces True; also used by repair/R4) |
| `grammar_multitoken_max` | `8` | Max run length per forward |
| `grammar_canvas_lookahead` | `0` | P4; playground forces 32; also used by repair/R4 |
| `use_dynamic_quant` | `False` | P5 |
| `generate_max_attempts` | `3` | P7/R5 playground budget (honored by `_ensure_valid_openui`) |
| `grammar_finalize_on_last_attempt_only` | `False` | P7 |

## Instrumentation

`model.generate_with_stats(prompt) -> (text, DecodeStats)` records
`denoiser_ms`, `dfa_sync_ms`, `stream_check_ms`, `detok_ms`, `context_ms`,
`pick_ms`, `forwards_count`, `probes_count`, `tokens_emitted`,
`accepted_run_tokens`.
