# Deadline-swallow fix applied: does it get the hero suite under the 30s wall? (NOT SHIP)

**Honesty:** `fixture_or_scratch`, isolated per-record diagnostic (suite `n=3`,
1 rep each, byte-identical recipe to both prior docs). **Not ship. A narrow,
scoped fix plus a measurement rerun.**

## Why this run

Iteration 1 of this autonomous loop
([`decode-compiler-tree-lexer-cache-hero-rerun-20260728.md`](decode-compiler-tree-lexer-cache-hero-rerun-20260728.md))
found the Lark lexer/scanner cache fix (#1173) alone does **not** get the
`exposure12` quality-champion smoke suite under `decode_timeout_seconds=30`,
and reconfirmed
[`decode-compiler-tree-deadline-swallow-finding.md`](decode-compiler-tree-deadline-swallow-finding.md)'s
`proposed_fix_sketch` (items 1-2) is the operative blocker. That doc had
already scoped the exact fix but not applied it. This session implements that
fix narrowly and re-runs the identical isolated-record protocol.

## Hypothesis

Fixing both the deadline-swallow bug (`compiler_draft.py:2313`'s bare
`except Exception` masquerading a cooperative-deadline `TimeoutError` as an
empty completion forest) and adding `check_decode_deadline()` calls to the
compiler-tree per-position loop (`twotower.py`'s `_compiler_ltr_decode_one`)
lets the 3 smoke records finish inside the 30s wall, where neither the lexer
cache alone nor the pre-fix baseline did.

## Fix applied (narrow, exactly the prior doc's sketch)

1. **`compiler_draft.py`** (`build_completion_forest`): added
   `except TimeoutError: raise` before the existing bare
   `except Exception: return CompletionForest((), "none")`, so a cooperative
   decode-deadline `TimeoutError` propagates instead of being recorded as a
   grammar dead-end. Non-timeout errors still fail closed exactly as before
   (regression-tested).
2. **`twotower.py`** (`_compiler_ltr_decode_one`): the per-position
   `while len(prefix) < length and prefix[-1] != eos_id:` loop now calls
   `check_decode_deadline()` at the top of every iteration, matching the
   existing coverage in `_constrained_ltr_repair` and the MaskGIT loop (both
   added in PR #1167). `_compiler_ltr_decode_batch` delegates to
   `_compiler_ltr_decode_one` per row, so this one loop-head edit covers both
   the single- and batch-decode compiler-tree paths.
3. Two new regression tests in
   `tests/test_harnesses/model_build/test_decode_deadline.py`:
   `test_build_completion_forest_propagates_deadline_timeout` and
   `test_build_completion_forest_still_fails_closed_on_other_errors`.
4. `model.twotower` bumped v261 → v262. `harness.model_build.eval` gets a
   `no-bump:` note (only its own test file gained tests; no owned file
   changed).

## Recipe

Byte-identical to both prior docs: rebuild `lever_exposure12_v1` (107
records, 5 `abstraction_ladder`), train the identical scratch checkpoint
(`checkpoint_sha256=cadeb1d3...` — matches both prior docs bit-for-bit), then
evaluate each of the 3 smoke records **in isolation**
(`--eval-limit 1 --eval-offset {0,1,2}`) with the fix applied
(`code_git_sha=4345d4b`, `code_dirty=true` — fix not yet committed at eval
time).

```bash
python -m scripts.build_train_data --source fixture --profile strict \
  --max-records-per-parent 12 --version lever_exposure12_v1 \
  --output-root outputs/data/train

python -m scripts.train_model \
  --train-dir src/slm_training/resources/data/train/lever_exposure12_v1 \
  --model twotower --context-backend scratch --steps 16 --batch-size 2 \
  --lr 1e-3 --structural-bias 1.5 --seed 47 \
  --run-id exp_lever_data_exposure12_s16_lr1e3_bs2_sb15_seed47 \
  --no-sync-checkpoints --device cpu

python -m scripts.evaluate_model \
  --test-dir src/slm_training/resources/data/eval/e938_role_safe_all_targets_v2 \
  --suite smoke \
  --train-dir src/slm_training/resources/data/train/lever_exposure12_v1 \
  --model twotower --device cpu \
  --checkpoint outputs/runs/exp_lever_data_exposure12_s16_lr1e3_bs2_sb15_seed47/checkpoints/last.pt \
  --grammar-constrained --decode-timeout-seconds 30 --seed 47 \
  --constraint-debt-routing-mode fixed_asap --run-class scratch_matrix \
  --eval-limit 1 --eval-offset <0|1|2>
```

`env -u NODE_OPTIONS` prefix used for build/train/eval, same sandbox caveat
as both prior docs.

## Results: before any fix → lexer-cache-only → both fixes (this run)

| record | meaningful (before → lexer-only → both) | decode_outcome (before → lexer-only → both) | latency_ms_p50 (this run) |
| --- | --- | --- | ---: |
| smoke_hero_01 (offset 0) | 0.0 → 0.0 → 0.0 | fallback_output → fallback_output → **runtime_timeout** | 30000.7 |
| smoke_button_01 (offset 1) | 0.0 → 0.0 → 0.0 | fallback_output → runtime_timeout → **runtime_timeout** | 30100.5 |
| smoke_callout_01 (offset 2) | 1.0 → 1.0 → **0.0** | fallback_output → fallback_output → **runtime_timeout** | 30000.4 |

`checkpoint_sha256=cadeb1d346d863e5a727a64847e59bfcb5c38db65fd6d5e3ef8c996c757539e9`
matches both prior docs exactly — same model, same seed, only the code
under test differs.

## Decision

**REJECT the hypothesis as stated.** Fixing both the deadline-swallow bug and
adding cooperative deadline checks to the compiler-tree loop does **not** get
any of the 3 records to finish inside `decode_timeout_seconds=30`. All three
still consume the full budget (`latency_ms_p50` 30.00-30.10s), essentially
identical wall-clock shape to both the pre-fix baseline and the
lexer-cache-only rerun.

**ACCEPT the fix on its own narrower, correctly-scoped claim: classification
is now honest.** Before any fix, all 3 records misreported as
`fallback_output`. After the lexer-cache-only fix, 1 of 3 (button)
inconsistently flipped to `runtime_timeout` on an otherwise-identical rerun
(flagged, not claimed, in the prior doc). After **this** fix, all 3 records
consistently and correctly report `runtime_timeout` — the deadline
`TimeoutError` now propagates instead of being caught and recorded as a fake
`empty_completion_forest` grammar dead-end.

**Important honesty note:** `smoke_callout_01`'s `meaningful_program_rate`
drops from the previously-reported **1.0 to 0.0** once the swallow bug is
fixed. That prior 1.0 was scored against the certified/minimal
`fallback_output` the swallow bug produced (`build_completion_forest`
returning an empty forest was not a real grammar dead-end, just a masked
timeout) — not against a genuine on-time decode. This is a correction of a
measurement bug, not a new regression: the champion's true smoke-suite
meaningful rate for this record was always 0; the swallow bug was hiding it.
The `exposure12` champion's honestly-measured smoke suite is now 0/3
meaningful, not the previously reported 1/3.

## Interpretation

This confirms the deadline-swallow finding's root-cause chain end to end.
The compiler-tree per-position loop genuinely has no way to finish early once
it starts spending multi-second `compiler_ms` per `build_completion_forest`
call inside the 30s wall — the fix changes what the harness truthfully
*reports* about that outcome, not how fast the model decodes. The underlying
performance blocker
([`decode-compiler-tree-lexer-cache-hero-rerun-20260728.md`](decode-compiler-tree-lexer-cache-hero-rerun-20260728.md)'s
per-call cost analysis: `build_completion_forest` still costs roughly
0.9-1.7s per call even with the lexer cache, over 17-33 calls per record) is
untouched and remains the real next-steps item for finishing inside the wall.

## Scope note

The exposure12 quality-champion's own reported smoke-suite quality is now
*worse* by this honest measurement (0/3 vs the previously reported 1/3) —
this is a correction of a measurement bug, not a claim the model got worse.
No promotion or roster claim is made or implied in either direction.

## Non-goals

No performance/latency improvement claimed (wall time unchanged, still
pinned to the 30s wall). No re-investigation of why `build_completion_forest`
costs what it costs (prior doc's next-steps item 2, still open). No
promotion, checkpoint sync, or `--ship-gates` claim. No multi-rep
confirmation (n=1 rep per record, matching both prior docs' own protocol).

## Next steps

1. Separately investigate `build_completion_forest`'s per-call wall cost
   (Node-bridge round-trip vs candidate-set size vs cache misses were not
   isolated in this or the prior session) — the actual remaining lever for
   finishing inside the wall.
2. Consider whether the compiler-tree search should reserve a fixed slice of
   `decode_timeout_seconds` for a final best-effort emission once
   `check_decode_deadline`'s cooperative signal is reliably observed, rather
   than running the deadline out to the exact millisecond every time.

## Tests / checks

Targeted regression subset (chosen after the full
`test_compiler_decode.py`/`test_grammar_fastpath.py` combined run exceeded
several minutes of wall time on this shared CPU sandbox for reasons unrelated
to this diff — real `TwoTowerModel` construction per test dominates, not the
2-line/8-line change under test):

```bash
python -m pytest -q \
  tests/test_harnesses/model_build/test_decode_deadline.py \
  tests/test_models/test_compiler_decode.py::test_canonical_valid_openui_propagates_decode_deadline \
  tests/test_models/test_compiler_decode.py::test_compiler_tree_batches_only_ambiguous_prefills_with_bound \
  tests/test_dsl/test_grammar_fastpath.py::test_completion_forest_closes_component_reference_array_before_postfix \
  tests/test_dsl/test_grammar_fastpath.py::test_completion_forest_excludes_symbol_consumers_after_contract_exhaustion \
  tests/test_dsl/test_grammar_fastpath.py::test_completion_forest_rejects_empty_component_children \
  tests/test_dsl/test_grammar_fastpath_lexer_cache.py
# 17 passed in 9.00s
```

This subset directly exercises every changed code path: `build_completion_forest`'s
exception handling (new + existing tests), the compiler-tree per-position
loop (`test_compiler_tree_batches_only_ambiguous_prefills_with_bound`),
deadline propagation (`test_canonical_valid_openui_propagates_decode_deadline`),
a representative sample of `completion_forest` structural behavior, and the
full lexer-cache suite.

`python -m scripts.verify_version_stamps --check` passes with the
`model.twotower` v261→v262 bump and the `harness.model_build.eval`
`no-bump:` note.

## Required artifacts

This JSON/Markdown pair
(`docs/design/decode-compiler-tree-deadline-fix-applied-rerun-20260728.{json,md}`).

## Cleanup note

The `lever_exposure12_v1` train-data rebuild and the scratch checkpoint used
for this diagnostic are not committed (`outputs/` is gitignored; the
auto-published copy this session created under
`src/slm_training/resources/data/train/lever_exposure12_v1` was removed after
the runs above since it is reproduction scaffolding, not new curated training
data — same convention as
[`decode-compiler-tree-deadline-swallow-finding.md`](decode-compiler-tree-deadline-swallow-finding.md)).

Captured: 2026-07-28T02:15:00Z
