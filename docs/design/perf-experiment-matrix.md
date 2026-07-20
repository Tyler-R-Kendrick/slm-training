# Inference-speed experiment matrix (P/Q/R-series)

Decode-only optimizations for TwoTower LTR grammar-constrained generate.
No retraining — each row overlays flags on an existing checkpoint (default:
`src/slm_training/resources/checkpoints/playground_demo/last.pt`).

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

| ID | latency_ms_mean | speedup vs P0 | forwards | Notes |
| --- | ---: | ---: | ---: | --- |
| P0 | 862 | 1.00× | 42.5 | Legacy baseline (this host) |
| Q9 / R9 | ~330–340 | ~2.6× | 13.8 | Decode recipe; R1/R2 always-on |
| **PG** | **~520–580** | **~1.6×** | **26.8** | Was ~2135ms / 469 fwd (**~4×** vs round-2 PG) |

Round-3 takeaways:

- **PG is the headline win** — repair now shares P3/P4 with greedy LTR, and
  `generate_max_attempts=1` + prior `_repair_ltr_texts` skips a redundant BOS
  ensure (forwards 469 → ~27).
- **R1/R2** are always-on in the pick/force-emit path; they do not need a
  separate flag. Alone they are within noise of Q9 on this demo ckpt once Q2
  already cut candidate count.
- Ship recipe unchanged for decode-only (**Q9/R9 flags**); playground uses
  the same levers plus repair/finalize with the R4/R5 wiring.

### Overnight rerun (2026-07-14, CPU, bridge up, `--limit 4`)

Fresh worktree from `origin/main`, committed `playground_demo/last.pt`,
`uv sync --extra dev --extra grammar`, and `npm ci` in the pinned OpenUI
bridge. This rerun is a performance result, not a ship claim: the demo
checkpoint produced `parse_rate=0.0` for P0/Q9/R9 and `0.25` for PG, so no
champion promotion was made.

| ID | latency mean | p50 | effective tok/s | parse | fidelity | outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| P0 | 4358.8 ms | 7086.4 ms | 9.75 | 0.00 | 0.00 | control; invalid demo output |
| Q9 | 1424.1 ms | 1422.5 ms | 80.05 | 0.00 | 0.00 | 3.06x faster; quality failure |
| R9 | 1415.9 ms | 1438.9 ms | 80.52 | 0.00 | 0.00 | fastest; quality failure |
| PG | 1585.9 ms | 1593.4 ms | 73.14 | 0.25 | 0.00 | partial parse; not promotable |

The focused matrix now marks candidate-only runs as unanchored instead of
reporting them as passing; use `--only P0,<candidate>` for a quality-gated
comparison. Dynamic quantization measured 10.18 tok/s and MaskGIT 0.58 tok/s
in the same environment, so both remain rejected for this CPU path.

### Telemetry correction (2026-07-15)

`aggregate_stats` previously used floor indexing for p95, which could report a
p95 below p50 for `n=2` (for example, a Q9 control reported p50≈1,687 ms and
p95≈1,527 ms). It now uses nearest-rank indexing. A fresh two-prompt control
reported P0 p50/p95≈1,996/6,976 ms and Q9≈1,421/1,436 ms; the quality anchor
remained invalid, so this was a telemetry-only correction and not a promotion.
The model-evaluation runner now uses the same percentile definition so
dashboard quality and perf latency summaries agree.

### Guardrail fix rerun (2026-07-15, CPU, bridge up, `--limit 8`)

The previous larger rerun exposed a second harness bug: zero-quality P0 values
created negative floors, allowing unusable candidates to pass. The guardrail
now rejects every candidate when P0 has zero parse or placeholder fidelity.
The corrected rerun measured Q9 at 80.7 tok/s, R9 at 85.7 tok/s, and PG at
79.7 tok/s, but all remain non-promotable because P0 was parse=0/fidelity=0.

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
| `compiler_decode_mode` | `off` | C-series: `forced`, `restricted`, or packed `tree` |

## Instrumentation

`model.generate_with_stats(prompt) -> (text, DecodeStats)` records
`denoiser_ms`, `dfa_sync_ms`, `stream_check_ms`, `detok_ms`, `context_ms`,
`pick_ms`, `forwards_count`, `probes_count`, `tokens_emitted`,
`accepted_run_tokens`.

The verified-solver matrix (VSS4-02,
[verified-scope-solver-benchmark.md](verified-scope-solver-benchmark.md)) reads
the same `DecodeStats` but reports solver/certificate time
(`solver_ms`, `certificate_ms`) **separately** from denoiser/projection/global-
verifier time (`denoiser_ms`, `projection_ms`, `global_verifier_ms`), plus
median/p95/p99 request latency and the exact-search work counters, so an
apparent latency win cannot mask added verification cost. Those rows run under
the fail-closed correctness gates before any latency comparison; a
latency/throughput gain never overrides a correctness-gate failure.

## C-series: compiler-drafted constrained decoding (2026-07-15)

```bash
OPENUI_BRIDGE_CLI=/home/codex/repos/slm-training/src/apps/openui_bridge/cli.mjs \
AGENTV_RUNNER=/home/codex/repos/slm-training/scripts/run_agentv_eval.mjs \
python -m scripts.run_perf_matrix --only C0,C1,C2,C3,C4 --limit 2 --warmup 0
```

Recipe: CPU, committed `playground_demo/last.pt`, smoke `n=2`, no training,
same-run C0 control, official bridge healthy. The run emitted AgentEvals JSONL
and a pinned `@agentv/core` result bundle (`total=5`, `passed=0`,
`executionErrors=0`). Full evidence is in
[`perf-matrix-results.json`](perf-matrix-results.json).

| ID | mode | mean ms | p50 ms | forwards/call | forced tokens | fallbacks | parse | fidelity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 | R9 / `off` | 1293.53 | 1210.37 | 13.5 | 0.0 | 0.0 | 0.00 | 0.00 |
| C1 | `forced` | 1069.76 | 604.55 | 14.5 | 1.5 | 1.0 | 0.00 | 0.00 |
| C2 | `restricted` | 511.96 | 461.00 | 9.0 | 0.0 | 1.0 | 0.00 | 0.00 |
| C3 | `tree` | 478.91 | 375.23 | 9.0 | 0.0 | 1.0 | 0.00 | 0.00 |
| C4 | hierarchy + repair | 715.17 | 695.39 | 12.0 | 0.0 | 1.0 | 0.00 | 0.00 |

This is wiring/performance evidence, not a ship result. The committed demo has
the compositional tokenizer, so C2–C4 correctly classified semantic coverage as
partial and exercised the prefix-seeded V7 fallback; they did not exercise a
complete lexer-native completion tree. C4 reduced p50 and neural forwards, but
the zero-parse/zero-fidelity C0 anchor is invalid. All five AgentV cases failed
the honest quality gate, so `compiler_decode_mode` remains **`off` by default**.
A promotion rerun requires an honestly evaluated lexer-native V7 checkpoint and
must preserve parse/fidelity within five absolute points while reducing both
forwards and p50.

Instrumentation now also records `backbone_ms`, `projection_ms`, `compiler_ms`,
`trie_ms`, `compiler_candidates`, `forced_spans`, `forced_tokens`, `trie_nodes`,
`restricted_projections`, `full_projections`, `compiler_fallbacks`, and
`seeded_fallbacks`.

## C5-C8 constraint-system definitions (proposed, unrun)

| ID | Lever | Acceptance boundary |
| --- | --- | --- |
| C5 | Cached terminal-equivalence token classes | Same allowed IDs as uncached mapping |
| C6 | C5 plus request-active symbol bitset intersection | Gold/active symbols retained; inactive entity/state rows removed |
| C7 | Conservative completion bounds plus compact active canvases | Unknown fallback never truncates; quality stays within guardrail |
| C8 | Combined C5-C7 stack | Lower work only counts with a valid same-run quality anchor |

Definitions can be inspected without loading a checkpoint or running a benchmark:

```bash
python -m scripts.run_perf_matrix --only C5,C6,C7,C8 --list
```

No C5-C8 latency or quality result exists. Reported speedups in CFGzip,
XGrammar-2, WGrammar, TruncProof, or related papers are external evidence only.

## E289 exact choice-state cache (2026-07-17)

The production choice decoder now caches exact legal sets by immutable symbolic
state. Against E288's byte-identical checkpoint, standalone p50 improved 2.65×
to 5.86× across all five suites while parse stayed 1.0 and dead ends stayed
zero. Hit rates ranged from 57.6% to 76.4%; cold-state p95 remains 5.9–8.7
seconds. Meaningful parse, fidelity, reward, and AgentV remain zero, so the
result is non-promotable. Full recipe and suite telemetry:
[E289](iter-e289-choice-state-cache-20260717.md).

## E290 grammar-derived choice candidates (2026-07-17)

Production/frame partitions plus available-ref/slot filters avoid 34.8% of
vocabulary probes on exact-cache misses. Exhaustive-oracle tests and two
all-suite evaluations preserve E289 behavior (parse 1.0, zero dead ends,
semantic metrics zero). Median p95 improves 1.14×–1.19×, while p50 regresses to
0.59×–0.89× of E289. The mixed result is non-promotable; next work targets
exact completion lower bounds and candidate-set allocation.
[Full E290 evidence](iter-e290-choice-direct-candidates-20260717.md).

## E291 exact completion-state cache (2026-07-17)

Exact minimum-completion memoization reaches 90.7–91.9% hit rates and removes
E290's p50 regression: median p50 improves 1.29×–1.99× and p95 improves
1.51×–1.93× versus E290. Against E289, all p95s improve 1.73×–2.30×; OOD p50
remains 13% slower. Parse stays 1.0 with zero dead ends, while semantic metrics
and AgentV remain zero. [Full E291 evidence](iter-e291-choice-completion-cache-20260717.md).

## H14: description-based retrieve-then-rerank (SLM-176, 2026-07-20)

Wiring/fixture harness for reducing the candidate set passed to the learned
reranker (`_project_candidates`) using deterministic description retrieval over
the complete live legal action set.  Default-off; controlled by
`action_shortlist_mode` and related `TwoTowerConfig` fields.

```bash
python -m scripts.run_slm176_action_shortlist_rerank_fixture --mode fixture --seeds 0 --d-model 32
```

**Claim class:** wiring / fixture only. No ship-gate claim is made.

### Expected effect

When enabled (`description_retrieval`), the legal action set is filtered to a
top-k description-retrieval shortlist (plus mandatory ids and margin ties)
before the neural reranker scores candidates.  The intended throughput effect is
fewer gathered projection rows per decode step for large legal sets, with a
fallback to the full set when confidence is flat or the legal set is small.

### Measured results

Placeholder — no trained-model perf run has been executed. The fixture verifies
that the shortlist plumbing preserves the full-set top candidate on synthetic
legal sets using the deterministic `FixtureDescriptionEncoder`.

### Honest caveats

- The retrieval encoder is a hash surrogate, not a trained text model.
- Query vectors are derived from the decoded prefix as a wiring placeholder.
- A promotion rerun must show parse/fidelity within five absolute points of the
  off baseline while reducing forwards or projection cost.
