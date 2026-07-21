# E649 fixing the `_binder_scope` AttributeError E648 flagged (root cause was broader than "warmup path")

E648 (perf phase, prior session) ran C5-C8 for the first time and, while
attempting an incidental combined `C0-C8` re-run, hit and flagged (not
fixed) a crash: `AttributeError: 'OpenUITokenizer' object has no attribute
'kind_ids'` from `compiler_draft.py::_binder_scope`, characterized as living
in "the batched `model.generate()` warmup path, distinct from
`generate_with_stats()` used by the scored main loop," triggered by
`compiler_decode_mode="forced"` (C1) with `warmup>=1`. This session
root-causes and fixes it.

## Repro

```bash
NODE_OPTIONS="" OPENUI_BRIDGE_CLI=$PWD/src/apps/openui_bridge/cli.mjs \
AGENTV_RUNNER=$PWD/scripts/run_agentv_eval.mjs \
python -m scripts.run_perf_matrix --only C0,C1 --limit 4 --warmup 1 \
  --checkpoint src/slm_training/resources/checkpoints/playground_demo/last.pt \
  --out-dir <tmp>
```

Confirmed deterministic, with the exact traceback E648 described:
`model.generate` -> `generate_batch` -> `_generate_batch_once` ->
`_greedy_ltr_decode_batch` -> `_compiler_ltr_decode_batch` ->
`_compiler_ltr_decode_one` -> `build_completion_forest` ->
`_references_resolved` -> `_binder_scope` ->
`tokenizer.kind_ids("bind")` -> `AttributeError`.

## Root cause is broader than E648's "warmup path" framing

E648's diagnosis said this was specific to the batched warmup call
(`model.generate()`, invoked directly from `run_one`'s warmup loop) as
opposed to the scored loop's `model.generate_with_stats()`. Tracing the
actual call graph shows this is wrong as a *mechanism* claim:
`generate_with_stats()` is itself just `collect_decode_stats()` wrapped
around `self.generate(...)`, i.e. **the exact same code path**
(`twotower.py:10674-10696`). There is no separate "warmup-only" branch.

Reproducing directly disproves the "`--warmup 0` never exercises this path"
claim too: running the *exact* command that produced the committed
`docs/design/perf-matrix-results.json` C1-C4 rows —

```bash
python -m scripts.run_perf_matrix --only C0,C1,C2,C3,C4 --limit 2 --warmup 0
```

(`docs/design/perf-experiment-matrix.md`'s own C-series command, dated
2026-07-15) — **also crashes** with the identical `AttributeError`, on this
checkpoint, on the current `compiler_draft.py`. `build_completion_forest`
calls `_references_resolved(tokenizer, prefix_ids)` unconditionally on
every decode step (compiler_draft.py:844), and `_references_resolved` calls
`_binder_scope`, which reads `tokenizer.kind_ids("bind")`
unconditionally at its very first line (compiler_draft.py:256, before this
fix) — so this crashes on the *first* token of *any* `forced`/`restricted`/
`tree`-mode decode with this checkpoint, warmup or not. The committed
`perf-matrix-results.json` C1-C4 evidence is therefore **stale relative to
current `compiler_draft.py`**, not merely warmup-blind as E648 assumed —
some change since 2026-07-15 introduced (or generalized) an unguarded
`kind_ids` call reachable from the scored path too. `perf-matrix-results.json`
itself is left untouched here (refreshing committed evidence is a separate,
larger task — the diff is large and dominated by unrelated `DecodeStats`
schema drift over the past six days, not just this fix); see "Not done" below.

## Actual root cause: one unguarded call in an otherwise-consistent pattern

`OpenUITokenizer` (`src/slm_training/models/tokenizer.py`) is the plain
word/char tokenizer — unlike `DSLTokenizer`/`ChoiceTokenizer`
(`src/slm_training/models/dsl_tokenizer.py`,
`src/slm_training/models/choice_tokenizer.py`), it has no
kind-classification API (`kind_ids`) by design. `compiler_draft.py`
already has an established, consistent convention for this: several
`kind_ids` call sites degrade gracefully when it's missing —
`emitted_component_count`'s `except Exception: # noqa: BLE001 - tokenizer
without kind_ids -> no gate` (line 311), and the `callable(kind_ids)` /
`callable(kind_of)` guards at lines 920-922 and 959-968. `_binder_scope`
(line 252) was the one place in this family that called
`tokenizer.kind_ids("bind")` with no guard at all — and because
`_references_resolved`, `_active_declaration_scope`,
`active_declaration_binder_id`, and `active_parent_component_ids` all call
`_binder_scope` first, the crash propagated through every one of them.

This is a genuine bug (a missing guard inconsistent with the rest of the
file), not an architectural mismatch — the "no gate" fallback semantics
this module already uses elsewhere is exactly the correct behavior here
too: a tokenizer with no binder-kind token space has no declarations or
references to reason about, so decode should proceed unconstrained by
binder-scope reasoning rather than crash.

## Fix

`src/slm_training/dsl/grammar/fastpath/compiler_draft.py`, `_binder_scope`
(around line 252): wrap the `tokenizer.kind_ids("bind")` read in
`try/except AttributeError` and return `([], [], None)` — no declarations,
no references, no active declaration slot — matching the "no gate"
convention already documented in this file. Every downstream caller
inherits the fix transitively through `_binder_scope`'s return value (no
other call site needed a change).

```python
def _binder_scope(
    tokenizer: Any, prefix_ids: list[int]
) -> tuple[list[int], list[int], int | None]:
    """Return declarations, references, and the active declaration slot."""
    try:
        bind_ids = set(tokenizer.kind_ids("bind"))
    except AttributeError:
        return [], [], None
    ...
```

Added a regression test,
`tests/test_dsl/test_grammar_fastpath.py::test_binder_scope_no_gates_without_kind_ids`,
which builds a plain `OpenUITokenizer` (confirms `not hasattr(tokenizer,
"kind_ids")`) and calls `build_completion_forest`,
`active_declaration_binder_id`, and `active_parent_component_ids` directly
against it, asserting no exception and the expected vacuous return values.

## Verification

- E648's exact repro (`--only C0,C1 --limit 4 --warmup 1`) no longer
  crashes; the run completes and writes a scoreboard/results JSON.
- The canonical `--warmup 0` command (`--only C0,C1,C2,C3,C4 --limit 2
  --warmup 0`, matching `perf-experiment-matrix.md`'s committed C-series
  invocation) also completes without crashing post-fix — confirms the path
  the committed `perf-matrix-results.json` evidence nominally relies on is
  not broken by this change (it was already broken before this fix, on
  current code — see above).
- `tests/test_models/test_compiler_decode.py`,
  `tests/test_dsl/test_grammar_fastpath.py`,
  `tests/test_models/test_choice_tokenizer.py`,
  `tests/test_harnesses/model_build/test_dsl_tokenizer.py`: byte-identical
  pass/fail sets before and after the fix (68 failed / 120 passed both
  times — all 68 pre-existing failures unrelated to `_binder_scope`,
  confirmed by diffing the sorted `FAILED` line lists from a `git stash`
  before/after run). New test passes; full `test_grammar_fastpath.py` file
  is 29/29 green.
- `python -m scripts.verify_version_stamps --check` passes with the
  `model.twotower` v83 bump recorded.

## Not done (explicitly out of scope here)

Refreshing `docs/design/perf-matrix-results.json` in place. Re-running the
canonical command post-fix produces a diff dominated by unrelated
`DecodeStats`/phase-summary schema drift accumulated over the six days
since the 2026-07-15 baseline (new/renamed counters, a
`counters_omitted_zero` field that didn't exist before, timing-noise
deltas) — a much larger, separate refresh task, not a side effect of this
bug fix. The committed file is left as-is; it should be treated as stale
for the C1-C4 rows until a dedicated refresh runs it.

## Files changed

- `src/slm_training/dsl/grammar/fastpath/compiler_draft.py` — `_binder_scope` guard (the fix).
- `tests/test_dsl/test_grammar_fastpath.py` — regression test.
- `src/slm_training/resources/versions.json` — `model.twotower` v82 -> v83.
- `docs/design/quality-experiment-matrix.md` — `## E649` entry (doc convention for infra fixes per this repo's practice).
- This file and [E649 JSON](iter-e649-binder-scope-kind-ids-crash-fix-20260721.json).

Evidence: ad hoc verification runs under `outputs/runs/e649-*` (ephemeral,
not committed).
