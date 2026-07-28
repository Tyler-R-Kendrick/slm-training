# Fix: `probe_chunk` re-lexes only the unsettled tail, not the whole prefix

**Honesty:** `fixture_or_scratch`. This is a real code change (`is_fix: true`)
with a benchmark-backed unit test, but **not a ship claim** — no
`train_model`/`evaluate_model --ship-gates` run, no checkpoint, no
scoreboard. Same status tier and immediate follow-up to
[`decode-compiler-tree-incremental-sync-relex-finding.md`](decode-compiler-tree-incremental-sync-relex-finding.md),
itself following
[`decode-compiler-tree-lexer-cache-fix.md`](decode-compiler-tree-lexer-cache-fix.md)
(PR #1173) and
[`decode-compiler-tree-lexer-rebuild-cost-finding.md`](decode-compiler-tree-lexer-rebuild-cost-finding.md)
(PR #1172).

## Task

The `_incremental_sync` fix's "Next steps" flagged: *"`probe_chunk`
(`engine.py`) has the same full-relex-of-`new_prefix` shape as pre-fix
`_incremental_sync` did, but was out of this session's scope... noted as a
further candidate, not measured or fixed here."* This session measures and
fixes it.

`probe_chunk` is `OpenUIIncrementalEngine`'s Q1 copy-probe: it tests whether
`prefix + chunk` stays a legal prefix without mutating engine state (used by
`compiler_draft.py`'s recursive witness search — the same I3 lookahead-then-
verify search `_incremental_sync`'s fix doc describes). Pre-fix, it called
`_lex_tokens(self._prefix + chunk)` — the entire current prefix plus the
probed chunk — on **every** call, the identical whole-prefix-re-lex shape
`_incremental_sync` had before its own fix.

## Measured: confirmed — same pathology, smaller absolute amplitude (many probes per prefix)

Environment: same `.venv-diag` used for the `_incremental_sync` fix (fresh
`pip install -e .`, Node bridge built,`NODE_OPTIONS` unset for the sandbox
reason documented previously). No train/eval/checkpoint — engine exercised
directly.

**Session shape**: the real 326-char full-record OpenUI DSL string (same one
used for `_incremental_sync`'s measurement), fed via 140 chunked `advance()`
calls, with one throwaway `probe_chunk("abc")` issued immediately before
each `advance()` — the realistic pattern callers use (probe candidate
continuations before committing to one):

| | total chars re-lexed | ideal O(n) chars | amplification | wall_ms |
| --- | ---: | ---: | ---: | ---: |
| before fix | 27,483 | 746 | **36.8x** | 46.90 |
| after fix | 1,934 | 746 | 2.6x | 28.42 |

(`copy_probes=114`, `copy_probe_fallbacks=26` — identical before and after,
confirming the probe's decisions themselves are unaffected by the fix.)

Scaling sweep (same synthetic `Stack([layer_N], ...)` records as the
`_incremental_sync` measurement, 326→3,580 chars, probe+advance interleaved
at every step):

| chars | wall_ms before | wall_ms after | speedup |
| ---: | ---: | ---: | ---: |
| 326 | 49.04 | 26.94 | 1.82x |
| 620 | 144.93 | 59.66 | 2.43x |
| 1212 | 442.00 | 169.87 | 2.60x |
| 2396 | 1718.19 | 572.46 | 3.00x |
| 3580 | 3580.26 | 1117.08 | **3.20x** |

Speedup grows with prefix length (the O(n²)→O(n) signature), though less
steeply than `_incremental_sync`'s standalone fix (2.6x→14.4x) because this
combined benchmark mixes `probe_chunk`'s (now-fixed) lex cost with
`InteractiveParser.copy()`'s own row-stack-depth-dependent cost and
`advance()`'s (already fixed) lex cost — `probe_chunk`'s re-lex was never
the *only* O(n)-scaling term in this combined workload, just the one this
fix removes.

## Fix applied: reuse `_settle_pos`, re-lex only the unsettled tail

`src/slm_training/dsl/grammar/fastpath/engine.py`, `probe_chunk`:

- Replaced `tokens = self._lex_tokens(new_prefix)` with the same
  tail-relex approach `_incremental_sync` now uses: lex only
  `new_prefix[self._settle_pos:]`, offset the resulting tokens' `start_pos`
  back to absolute coordinates, and splice onto
  `self._fed_tokens[:prev_settled_count]` for the identity/shrink check.
- `delta` (the tokens fed into the throwaway `InteractiveParser.copy()`) is
  now sliced from `tail_tokens` instead of the full `tokens` list, using the
  same index arithmetic as `_incremental_sync`.
- **Read-only preserved exactly**: `probe_chunk` still never mutates
  `self._settle_pos`, `self._fed_token_count`, or `self._fed_tokens` — it
  only reads `self._settle_pos` (already maintained by `_full_sync`/
  `_incremental_sync`) to know where the safe tail boundary is.

Same safety argument as `_incremental_sync`'s fix applies unchanged:
`openui.lark` has no lookbehind, so every token before `self._settle_pos` is
provably final regardless of what `probe_chunk` appends after it; any
divergence (shrink, gluing, unlexable tail) falls back to `None`/`False` via
the same guard the pre-fix code already had, never producing a wrong
legality decision.

### Verification (not just this session's script)

`tests/test_dsl/test_grammar_fastpath_probe_chunk_relex.py` (8 tests,
committed):

- `test_probe_chunk_matches_full_resync_shadow_across_sessions` (4
  parametrized sessions × 9 representative probe chunks per step, 15-140
  steps each) — compares the tail-relex `probe_chunk`'s `bool | None`
  return against an independent full-resync shadow implementation at
  **every** probe call, and additionally asserts `probe_chunk` never
  mutates `_prefix`/`_fed_tokens`/`_settle_pos` (its read-only contract).
- `test_probe_chunk_relexes_near_linear_not_quadratic` — regression guard:
  total characters re-lexed across the interleaved probe+advance session
  stays under an 8x ceiling over the O(n) ideal (measured 2.6x post-fix vs.
  36.8x pre-fix).
- `test_probe_chunk_faster_than_always_full_resync_on_long_session` —
  wall-time benchmark; asserts the tail-relex path beats an always-full-resync
  shadow by at least 1.2x (measured ~1.65x locally on the 326-char session
  without scaling), conservative floor for a slower/loaded CI host.
- `test_build_completion_forest_unchanged_on_finding_prefixes` (parametrized
  x2, `prefix8`/`prefix9`) — `build_completion_forest` (which calls
  `probe_chunk` internally via `compiler_draft.py`'s I3 witness search)
  still returns `coverage == "complete"` and `n_paths == 2`/`12`. I3/I6
  proof at the decode-output layer.

Sanity check the tests are not vacuous: ran the identical file against
pre-fix `engine.py` (`git stash`) — 2 of 8 fail
(`test_probe_chunk_relexes_near_linear_not_quadratic` with the exact 36.8x
amplification measured above, and
`test_probe_chunk_faster_than_always_full_resync_on_long_session`, which
inverts — the pre-fix path is *not* faster than the shadow, since they're
nearly the same code); the other 6 pass both before and after — expected,
since they assert the invariant (parity with the shadow oracle, read-only
contract, decode legality) the fix must preserve, not the fix's existence.

```
env -u NODE_OPTIONS .venv-diag/bin/python -m pytest tests/test_dsl/test_grammar_fastpath_probe_chunk_relex.py -v
# 8 passed in 6.31s (post-fix)
```

## Regression sweep (existing suite, unaffected by this change)

Ran with `.venv-diag`, each command under the 3-minute cap:

| suite | result |
| --- | --- |
| `tests/test_dsl/test_grammar_fastpath_lexer_cache.py tests/test_dsl/test_grammar_fastpath_incremental_sync.py tests/test_dsl/test_grammar_fastpath_probe_chunk_relex.py` | 27 passed |
| `tests/test_dsl/{test_pack,test_speculative_rank,test_exact_forced_horizon,test_lattice_search,test_solver_state,test_packs,test_solver_decode}.py` | 102 passed, 3 failed |
| `tests/test_dsl/test_grammar_fastpath.py -k "lexer or force or completion_forest or engine or pick_constrained"` | 24 passed, 1 failed, 18 deselected |
| `tests/test_harnesses/model_build/{test_lexer_smoke,test_dsl_tokenizer}.py` | 27 passed, 1 failed, 1 deselected |
| `tests/test_harnesses/model_build/{test_grammar_hf,test_v4_levers,test_a3_coverage_remask,test_template_fill,test_v6_remask}.py` | 30 passed, 1 failed |

**All 6 failures are the exact same pre-existing failures confirmed unrelated
in the `_incremental_sync` fix's own regression sweep** (this session's diff
touches only `probe_chunk` in the same already-audited file, so the prior
session's `git stash` confirmations still apply — re-ran the sweep here to
prove no *new* failures were introduced, not to re-litigate causes already
established):

- `test_speculative_rank.py::test_committed_table_matches_its_builder` /
  `::test_committed_table_ranks_real_branch_points_confidently` — stale
  committed `speculative_ngram_v1.json`.
- `test_packs.py::test_pack_fixture_loop_generate_train_eval` —
  `require_constrained_generation` rejects the fixture's
  `grammar_constrained=False` test config.
- `test_grammar_fastpath.py::test_completion_forest_admits_null_for_optional_string_after_slots_exhaust`
  — `build_completion_forest() got an unexpected keyword argument
  'enforce_schema_component_types'` (signature drift).
- `test_lexer_smoke.py::test_surface_identifier_arm_is_prohibited`.
- `test_v4_levers.py::test_generate_batch_requests_consumes_harness_slot_contract`
  — slot-contract prompt-string formatting.

`ruff check` on both changed files: clean.
`python -m scripts.verify_version_stamps --check`: `2 changed file(s), 0
component(s) touched` — same as both prior sessions; `engine.py` is not a
registered `versions.json` path.
`python -m scripts.repo_policy`: `ok (tracked + untracked)`.
`git diff --check`: clean.

## Interpretation

Closes the last open full-relex site named in the `_incremental_sync` fix's
"Next steps." `probe_chunk` shared the identical bug shape — re-lexing the
whole prefix (plus probed chunk) from scratch on every call — but is called
*more frequently per decode step* than `_incremental_sync` in the I3
recursive witness search (candidate continuations are probed before one is
committed), so its pre-fix amplification (36.8x on the combined
probe+advance session) sits between `_incremental_sync`'s single-call
(1.0x — no compounding for one call) and multi-call session (81.2x)
numbers, consistent with the same root cause at a different call frequency.
The fix reuses the exact `_settle_pos` boundary machinery
`_incremental_sync`'s fix introduced, so no new safety argument was needed
— `probe_chunk` inherits the same "no lookbehind in `openui.lark`" proof.

## Scope note (what this does NOT establish)

- No train/eval run, no checkpoint, no `--ship-gates` scoreboard —
  harness-internal correctness+perf fix with unit-test coverage, not a
  readiness claim.
- Not re-measured against the full per-record `compiler_ms` telemetry or the
  `exposure12` quality-champion eval protocol from PR #1171 — still an open
  next step across all three sessions in this chain.
- The combined-workload scaling sweep does not isolate `probe_chunk`'s own
  lex cost from `InteractiveParser.copy()`'s cost or `advance()`'s
  (already-fixed) lex cost — the reported wall-time speedups are for the
  realistic *combined* probe+advance pattern, not a `probe_chunk`-only
  microbenchmark.
- This closes the currently-known full-relex sites in `engine.py`
  (`_incremental_sync`, `probe_chunk`); no further systematic grep for other
  `_lex_tokens(...)` call sites with a similar shape was performed this
  session.

## Next steps

1. Re-run the PR #1171 isolated per-record eval protocol (or the seeded
   multi-rep `lever-hard-decode-timeout-wall` protocol) now that all three
   lexing-cost fixes in this chain (lexer-object cache, `_incremental_sync`
   tail-relex, `probe_chunk` tail-relex) are in place, to measure aggregate
   `compiler_ms` impact against the `exposure12` quality-champion hero.
2. If `InteractiveParser.copy()` cost (independent of lexing) shows up as a
   significant remaining term once lexing is no longer the bottleneck,
   profile it separately — not measured or attributed in this session.

## Cleanup note

`.venv-diag`, `src/apps/openui_bridge/node_modules/` (both gitignored) and
the session's reproduction scripts (`measure_probe_chunk.py`,
`measure_probe_chunk_scaling.py`, `verify_probe_chunk_fix.py` — session
scratchpad, not reusable harness tools) are not committed.

Captured: 2026-07-28T14:00:44Z
