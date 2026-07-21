# E648 running C5-C8 for the first time (and a composition gap they expose)

E645-E647 all lived inside `TwoTowerModel._required_slot_margin_bias` and the
quality-experiment-matrix lineage. This session diversifies to the bench/perf
phase instead: `docs/design/perf-experiment-matrix.md`'s own "C5-C8
constraint-system definitions (proposed, unrun)" section states plainly "No
C5-C8 latency or quality result exists." Reading `scripts/run_perf_matrix.py`
found the four rows (`grammar_equivalence_cache`, `grammar_active_symbol_
bitsets`, `grammar_completion_bounds`, `compact_active_canvas`) were already
implemented and covered by a registration-only unit test
(`test_c5_c8_are_registered_without_running`) — genuinely proposed, genuinely
unrun. This session runs them.

## A methodology gap, found before trusting the numbers

C1-C4 stay comparable to C0 (the R9 control) despite not setting C0's
`grammar_verify_chosen_only`/`grammar_multitoken_accept`/`grammar_canvas_
lookahead=32` overlay explicitly, because `compiler_decode_mode` (forced/
restricted/tree) substitutes for those levers — confirmed by re-reading the
already-committed `perf-matrix-results.json`: C1-C4's `forwards_count_mean`
(9.0-14.5) stays close to C0's 13.5. C5-C8 leave `compiler_decode_mode` at its
default `off`, so they fall back to the full per-token grammar-constrained
LTR/MaskGIT decode path *without* the R9 speed levers: `forwards_count_mean`
jumps to 103.25-114.0, 8-9x C0's 13.5. A raw C5-C8-vs-C0 latency comparison
would conflate "R9 levers absent" with "the C5-C8 lever itself does
something." Flagged, not fixed — deciding whether C5-C8 should carry the R9
overlay (matching C0/C1-C4's convention) or intentionally test against a bare
baseline is a harness-design call for `improve-openui-harnesses`, not made
here.

## Official matrix run

```bash
NODE_OPTIONS="" OPENUI_BRIDGE_CLI=$PWD/src/apps/openui_bridge/cli.mjs \
AGENTV_RUNNER=$PWD/scripts/run_agentv_eval.mjs \
python -m scripts.run_perf_matrix --only C0,C5,C6,C7,C8 --limit 4 --warmup 1 \
  --out-dir outputs/runs/e648-perf-c5c8-20260721
```

(`NODE_OPTIONS` in this session's shell defaults to `--import tsx
--max-old-space-size=8192`; the openui_bridge Node subprocess rejects
`--import` inside `NODE_OPTIONS`, so it was cleared for every bridge-touching
command — an environment quirk, not a code bug.)

Committed `playground_demo/last.pt` (compositional tokenizer, same checkpoint
as the prior C0-C4 run); smoke suite falls back to `test_seeds.jsonl`
(`outputs/data/eval/v1/smoke` absent on this host); `n=4`, `warmup=1`, CPU.

| ID | lever | mean ms | p50 ms | p95 ms | forwards/call | parse | raw_syntax | fidelity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 | R9 control | 809.65 | 795.52 | 876.87 | 13.5 | 0.0 | 0.0 | 0.0 |
| C5 | equivalence cache | 1295.78 | 1291.17 | 1329.98 | 114.0 | 0.0 | 0.0 | 0.0 |
| C6 | C5 + active-symbol bitsets | 1339.61 | 1314.31 | 1388.49 | 114.0 | 0.0 | 0.0 | 0.0 |
| C7 | completion bounds + compact canvas (MaskGIT) | 2374.36 | 1670.83 | 3646.70 | 103.25 | 0.25 | 0.5 | 0.0 |
| C8 | C5+C6+C7 combined (MaskGIT) | 2344.58 | 1586.56 | 3766.95 | 103.25 | 0.25 | 0.5 | 0.0 |

All five guardrail rows fail: C0 itself is `invalid zero-quality baseline;
not promotable` (`parse_rate=0.0`) — the same pre-existing compositional-
tokenizer limitation already documented for the C1-C4 run, not new — so
C5/C6/C7/C8 inherit `invalid P0 quality anchor` by construction. AgentEvals:
5 total, 0 passed (same pattern as C0-C4).

## Matched-control run (isolating each lever from the composition gap)

To separate "the C5-C8 lever does something" from "R9 levers are absent,"
this session built two ad hoc matched controls — same base recipe as C5-C8,
target lever off, identical smoke prompts via the canonical `_load_prompts`
helper — by importing `scripts.run_perf_matrix.PerfExperiment`/`run_one`/
`_load_prompts` unmodified in a scratchpad script (no new CLI, no shadow
harness). **A first attempt used a hand-written prompt list instead of
`_load_prompts` and produced a since-discarded, spurious "C7 regresses parse
rate 0.5→0.25" reading** — corrected before it went into this writeup by
rerunning with the exact same prompts as the official matrix run.

| Comparison | latency delta | forwards delta | quality delta | read |
| --- | ---: | ---: | ---: | --- |
| C5 vs C5CTRL (bare LTR) | +0.69% | 0 | 0 | noise |
| C6 vs C5 (+ bitsets) | +3.38% | 0 | 0 | marginally slower, not faster |
| C7 vs C7CTRL (bare MaskGIT) | -3.19% | 0 | 0 | noise (p50/p95 don't agree in a way consistent with a real forwards-driven speedup) |
| C8 vs C7 (full stack) | -1.25% | 0 | 0 | no improvement over C7 alone |

`C5CTRL`: 1286.85ms mean, 114.0 forwards, parse/fidelity 0.0.
`C7CTRL`: 2452.59ms mean, 103.25 forwards, parse 0.25 / raw_syntax 0.5 /
fidelity 0.0.

## Verdict

C5-C8 now have real measured numbers for the first time. Two honest findings:
(1) a real methodology gap in how C5-C8 compose with the R9 control (flagged,
not fixed); (2) once measured against a matched same-recipe control, **none
of C5, C6, or C7 show a measurable change in `forwards_count_mean` or quality
at n=4** — latency deltas are a few percent and don't agree in sign across
p50/p95, consistent with CPU-timing noise. C8 (the combined stack) does not
improve on C7 alone either. This is a negative-but-genuine result, distinct
from (and additional to) the pre-existing zero-quality-anchor limitation
already known from the C1-C4 run. No promotion, no default change (all four
flags stay default-off/on exactly as before). No checkpoint trained,
promoted, or synced. No source code touched — no `version_stamp` bump
applies.

## Incidental finding (not fixed, out of scope)

Attempting a combined `C0-C8` re-run (to refresh `perf-matrix-results.json`
in place per the `documenting-experiment-results` artifact map) surfaced a
real, previously-undocumented crash: with `compiler_decode_mode="forced"`
(C1) and `warmup>=1`, `model.generate()` — the batched warmup path, distinct
from `generate_with_stats()` used by the scored main loop — raises
`AttributeError: 'OpenUITokenizer' object has no attribute 'kind_ids'` from
`compiler_draft.py::_binder_scope` (`build_completion_forest` →
`_compiler_ltr_decode_batch` → `_greedy_ltr_decode_batch` → `generate_batch`
→ `generate`). The previously-committed C0-C4 result used `--warmup 0` and
never exercises this path, so the bug was silent there. **Not investigated
or fixed** — a runtime-generate-path bug is a separate, larger change from
executing the already-implemented C5-C8 rows. `docs/design/perf-matrix-
results.json` was left untouched (the crash happened before the script's
final write, so nothing was corrupted); its C0-C4 evidence remains valid.

Repro: `... python -m scripts.run_perf_matrix --only C0,C1 --limit 4
--warmup 1 --out-dir <tmp>`.

## Next steps

1. A genuine C5-C8-vs-C0 speedup claim needs either C5-C8 extended with the
   R9 overlay flags (a `scripts/run_perf_matrix.py` change via
   `improve-openui-harnesses`), or a formally registered bare-baseline
   control (this session's `C5CTRL`/`C7CTRL` were ad hoc/informal only).
2. At `n=4` none of the four levers show an effect; scale to the full smoke
   suite (`n=16`) or a real eval suite before concluding they're inert.
3. The checkpoint's zero-parse/zero-fidelity anchor (compositional tokenizer,
   not lexer-native) remains the pre-existing, already-documented blocker for
   any promotion claim on this family — unrelated to C5-C8, out of scope
   here.

Evidence: `outputs/runs/e648-perf-c5c8-20260721/` (ephemeral, not committed)
and [JSON](iter-e648-perf-c5-c8-execution-20260721.json).
