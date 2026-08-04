# `compiler_ms` decode-cost metering gap (finding + fix, not a ship claim)

**Honesty:** `fixture_or_scratch` / diagnostic reproduction, isolated single
records. Not a ship claim.

**Routed from:** the loop's `repair_harness` handoff for `model_build`
(frozen manifest `becbf08df082ca96a0f5b686cbd81d21130ca5be60637b95ac8601945f2adf7e`,
[`continuous-openui-local-gd6j83-c2-dual-arm-decode-timeout.md`](continuous-openui-local-gd6j83-c2-dual-arm-decode-timeout.md)),
which asked `improve-openui-harnesses` to profile `strict_compiler_tree`
decode under the seed-100002 fixture records and determine why
`compiler_ms_mean` jumped ~5-6x symmetrically vs. the seed-100001 control one
cycle earlier — the same class of symmetric dual-arm decode timeout already
left open, unresolved, in
[`autotrain-cycle-c5-dual-arm-decode-timeout-unresolved.md`](autotrain-cycle-c5-dual-arm-decode-timeout-unresolved.md).

## What this session found

Reproducing the frozen control checkpoints directly (`evaluate_model.py
--ship-gates`, `--eval-limit 1`, decode timeout raised to 60s purely so the
record finishes instead of being cut off — never used to claim a ship-gate
pass) surfaced something neither prior session checked: **the two cycles'
checkpoints decode through two different mechanisms.** The seed-100001
control checkpoint's own persisted config declares `compiler_decode_mode:
"off"` (it decodes via the legacy LTR/`exact_forced_token_id` deterministic-
singleton bypass in `models/grammar.py`); the seed-100002 checkpoints declare
`compiler_decode_mode: "tree"` (the compiler-tree greedy path in
`_compiler_ltr_decode_one`/`_compiler_ltr_decode_batch`). `evaluate_model.py
--ship-gates` labels both runs' `evaluation_policy` as `strict_compiler_tree`,
but it does not force `compiler_decode_mode` onto a checkpoint unless
`--compiler-decode-mode` is also passed as an explicit CLI flag (which makes
it a `runtime_override_field`); otherwise `eval_runner.evaluate()` restores
the checkpoint's own declared value after `build_model()`. Neither prior
investigation checked this field before comparing `compiler_ms_mean` across
the two cycles.

That divergence hid a real bug: `models/grammar.py`'s `exact_forced_token_id()`
— the legacy path's per-token I2 bypass, invoked once per token from
`_constrained_ltr_repair` — calls `build_completion_forest()`, **the exact
same expensive `CompletionSession`/`outgoing()`-edge grammar-authority
computation the compiler-tree path times as `compiler_ms`**, with **zero
`timed_ms` instrumentation anywhere in its call chain**. A checkpoint that
decodes through this mechanism can spend essentially all of its real decode
wall time inside this one unmetered call while `compiler_ms` reads `0.0` and
the true cost silently lands in `unattributed_ms`.

Isolated single-record reproduction, before this session's fix:

| | seed 100001 control (`compiler_decode_mode=off`) | seed 100002 control (`compiler_decode_mode=tree`) |
|---|---|---|
| `compiler_ms_sum` | **0.0 ms** | 14,864.2 ms |
| `total_ms_sum` | 31,630.7 ms | 15,239.0 ms |
| `unattributed_ms_sum` | 31,307.4 ms | 338.2 ms |
| `attributed_fraction` | **0.0102** | 0.9778 |
| `completion_unique_states` | 163,216 | 90,295 |

After wrapping `exact_forced_token_id`'s `build_completion_forest` call (and,
defense in depth, `completion_forced_closure`, called immediately after the
already-timed forest build in both `_compiler_ltr_decode_one` and
`_compiler_ltr_decode_batch`) in the same `compiler_ms` bucket:

| | seed 100001 control, same record, after fix |
|---|---|
| `compiler_ms_sum` | **38,001.6 ms** |
| `total_ms_sum` | 39,266.1 ms |
| `unattributed_ms_sum` | 74.0 ms |
| `attributed_fraction` | **0.9981** |

## Verdict: a metering bug, not a runaway/exponential search

The completion kernel's own bounds are intact and were **not** the defect:

- `CompletionSession.terminal_witness` caps each top-level candidate to a
  fixed `node_budget` (16 nodes).
- `LatticeSearchState.backtrack_limit` defaults to 8.
- `check_decode_deadline()` is honored throughout every recursive expansion
  (`outgoing`/`advance`/`terminal_witness`), so a slow decode fails closed
  with a typed `TimeoutError` rather than hanging.
- `SemanticState` carries a real structural `__eq__` and a memoized
  structural `__hash__`; `OpenUIIncrementalEngine.parser_state_key()` is
  structural (LALR state stack + pending tail) — state interning is not
  defeated by an identity-based key.

The large counters observed (tens of thousands of `completion_unique_states`/
`completion_edges_built` per single ~256-token record) reflect that this is a
genuinely non-trivial, repeated Lark-based incremental-parse-plus-scope-
analysis invoked once per token (or once per top-level candidate per token)
— not an unbounded or exponential blow-up. Per the loop law, this closes the
*approach* of chasing a search-algorithm bug in the compiler-tree search
itself (nothing there was found broken) without closing the *goal* (I14):
the successor approach is re-measuring the actual dual-arm timeout with
correct metering, below.

## Fix

- [`src/slm_training/models/grammar.py`](../../src/slm_training/models/grammar.py):
  wrap `exact_forced_token_id`'s `build_completion_forest` call in
  `with timed_ms(get_active_stats(), "compiler_ms")`.
- [`src/slm_training/models/twotower.py`](../../src/slm_training/models/twotower.py):
  widen the existing `with timed_ms(stats, "compiler_ms")` block in both
  `_compiler_ltr_decode_one` and `_compiler_ltr_decode_batch` to also cover
  the immediately-following `state.completion_forced_closure(...)` call
  (the same kind of `outgoing()`-edge walk, previously timed only for the
  initial forest build).

Both changes are **purely additive timing instrumentation**: no decode
invariant is touched (I2 singleton bypass and I6 constrained legality are
unchanged; nothing about which tokens are legal or chosen changes), no
generated output changes. Only `decode_stats` attribution changes —
`compiler_ms` now measures real cost that was previously invisible, and
`unattributed_ms` shrinks correspondingly. `model.twotower` bumped v309 →
v310 in `src/slm_training/resources/versions.json` (behavior-changing for
the *metric*, not for decode output).

## Regression test

[`tests/test_dsl/test_exact_forced_horizon.py::test_exact_forced_token_id_charges_compiler_ms_not_unattributed`](../../tests/test_dsl/test_exact_forced_horizon.py):
monkeypatches `build_completion_forest` to add a measurable delay while still
returning its real result, calls `exact_forced_token_id` under an active
`DecodeStats`, and asserts `compiler_ms` captures at least 90% of the delay.
Verified this test fails on the pre-fix code (`compiler_ms == 0.0`) and
passes after the fix.

## Test results

- `tests/test_dsl/test_exact_forced_horizon.py` — 3 passed.
- `tests/test_dsl/test_exact_forced_horizon.py tests/test_models/test_compiler_decode.py`
  — 235 passed, 2 failed; the 2 failures are pre-existing (missing
  `@openuidev/lang-core` npm bridge deps in this sandbox) and fail
  identically on an unmodified checkout.
- `tests/test_scripts/test_run_autotrain_continuous.py` — 222 passed, 1
  skipped. `test_replayed_dual_arm_timeouts_remain_inconclusive_and_require_repair`
  passes unmodified — the dual-arm-timeout contract this repair must not
  weaken is intact.
- Wider sweep (`test_grammar_diffusion.py`, `test_completion_decode_state.py`,
  `test_completion_kernel.py`, `test_grammar_fastpath_lexer_cache.py`,
  `test_engine_direct_feed.py`) — 101 passed, 2 failed; pre-existing
  (progspec generator oracle rejection), fail identically unmodified.
- `.githooks/check-changed` — 15 pre-existing failures reproduced identically
  on an unmodified checkout (Node bridge deps, operator-policy-view fixture
  drift, tree-edit value-mode label drift, twotower checkpoint-preserve
  fixtures); zero new failures introduced.
- `python -m scripts.repo_policy` — ok.
- `python -m scripts.verify_version_stamps --check` — ok after the
  `model.twotower` bump.

## Conclusion

This is a genuine, now-fixed decode-cost **metering** defect, not evidence of
a runaway or exponential compiler-tree search. It directly undermines the
specific evidence chain the c5 and gd6j83-c2 docs relied on ("`compiler_ms_mean`
jumped ~5-6x symmetrically"): that comparison silently pitted a fully-metered
decode (`compiler_decode_mode=tree`) against an almost-entirely-unmetered one
(`compiler_decode_mode=off`), because the two checkpoints/cycles happened to
decode through different mechanisms and `evaluate_model.py --ship-gates`
does not force one uniformly without an explicit `--compiler-decode-mode`
flag. Once metered identically, the seed-100001 control's real per-record
grammar-authority cost (~38s in this single-record reproduction) is
comparable to or larger than seed-100002's (~15-23s/record) — the two are
not obviously different regimes once measured on the same basis.

This finding neither confirms nor rules out that the gd6j83-c2 dual-arm
timeout itself would reproduce identically under corrected metering; per the
loop's existing rule (never auto-retire a symmetric dual-arm timeout, always
require repair first — the exact contract
[`test_replayed_dual_arm_timeouts_remain_inconclusive_and_require_repair`](../../tests/test_scripts/test_run_autotrain_continuous.py)
locks in), that determination is left to the queued `retry_measurement`
replay on the identical frozen arm now that metering is fixed.

Machine evidence:
[`compiler-tree-forced-closure-decode-metering-gap.json`](compiler-tree-forced-closure-decode-metering-gap.json).
