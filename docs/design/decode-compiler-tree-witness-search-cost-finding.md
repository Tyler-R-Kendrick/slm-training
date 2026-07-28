# Finding: post-fix `compiler_ms` is pinned by the recursive witness search's `InteractiveParser.accepts()` cost, not the lexer or the Node bridge

**Honesty:** `fixture_or_scratch`, isolated per-record diagnostic (`n=3`,
1 rep each, same protocol as the finding it follows up on) plus one direct
cProfile of a real (non-synthetic) record decode. **Not ship. Not a fix — a
diagnostic finding**, same status tier as
[`decode-compiler-tree-lexer-rebuild-cost-finding.md`](decode-compiler-tree-lexer-rebuild-cost-finding.md)
(PR #1172) and
[`decode-compiler-tree-deadline-swallow-remeasure.md`](decode-compiler-tree-deadline-swallow-remeasure.md)
(PR #1190), which named this exact question as its "Next steps" item 1.

## Task

PR #1190's own next-step #1: root-cause why `compiler_ms` is still pinned at
~30s post-fix (both the deadline-swallow fix, PR #1189/`model.twotower`
v262, and the lexer-cache fix, PR #1173/`dsl.grammar_fastpath_lexer_cache`,
are in the tree) — compare `compiler_prefill_batches`/candidate-set size
against the pre-fix numbers to isolate whether this checkpoint's
compiler-tree search genuinely needs >30s per record independent of lexer
cost, or whether the lexer-cache benefit is concentrated on cache hits this
record's decode path doesn't exercise.

## Why this needed new instrumentation

`evaluate_model`'s existing `metrics["decode_stats"]` aggregate — the field
PR #1171's original finding read directly from `eval.json` — is **only
populated when `decode_stats_rows` is non-empty**
(`eval_runner.py:1817-1819`). Post-fix, a compiler-tree deadline
`TimeoutError` now correctly *propagates* out of
`_generate_chunk_unbounded`'s `with collect_decode_stats() as stats:` block
instead of being swallowed inside `build_completion_forest`. That means the
`decode_stats_rows.append(stats)` call at `eval_runner.py:1066` — reached
only on a *normal* return from that `with` block — is skipped on every
timeout. `eval_runner.py:1160-1165` tries `getattr(exc, "decode_stats",
None)` to recover the in-flight stats bucket from the exception, but nothing
in the codebase ever attaches a `decode_stats` attribute to the raised
`TimeoutError` — confirmed by grep (`grep -rn "\.decode_stats\s*=" src/`:
zero hits). **Post-fix, `compiler_ms` telemetry is silently dropped for
every timed-out record** — verified directly: `eval.json`/`eval_smoke.json`
from PR #1190's own reproduced runs (`outputs/runs/eval_off{0,1,2}_postfix/`,
still on disk from that session) contain no `decode_stats` key at all
(`'decode_stats' in json.dumps(d).lower()` → `False` on all three). PR
#1190's "compiler_ms stays pinned at ~30.0-30.5s (unchanged from pre-fix)"
claim was therefore an *inference* from `latency_ms` alone (since
`compiler_ms` was ~97% of `total_ms` pre-fix), not a direct post-fix
`compiler_ms` measurement — that gap is what this session closes.

To get the real number, this session monkeypatches
`decode_stats.collect_decode_stats` (scratchpad script, not committed) to
capture the live `DecodeStats` bucket's `.as_dict()` unconditionally on exit
— exception or not — instead of relying on the eval harness's own (broken,
for this path) recovery attempt. No production code changed.

## Reproduction

Reused the checkpoint/train-data already on disk from PR #1190's session
(`outputs/runs/exp_lever_data_exposure12_s16_lr1e3_bs2_sb15_seed47`,
`outputs/data/train/lever_exposure12_v1`) — same recipe/seed as the finding
this follows up on, `code_commit` = this branch's parent tip
(`fec8c08`, PR #1189 + PR #1173 + PR #1190 all present, `model.twotower` v262).

```bash
# monkeypatches collect_decode_stats() to record bucket.as_dict() on any exit
python diag_deadline_stats.py <offset> capture_off<offset>.json
# in-process call of scripts.evaluate_model.main([... same args as the
# finding/remeasure's evaluate_model invocation ..., "--eval-offset", "<offset>"])
```

A second script cProfiles the identical single-record run (offset 0) end to
end (no synthetic prefix — the real `evaluate_model` call path, same
technique as PR #1172 but on this checkpoint's actual decode instead of a
hand-picked dead-end prefix).

## Measured: per-call `compiler_ms` dropped 2x-11x; total stays pinned because 2x-11x more calls now complete

| record (offset) | pre-fix `compiler_prefill_batches` | pre-fix `compiler_ms` | pre-fix ms/call | post-fix `compiler_prefill_batches` | post-fix `compiler_ms` | post-fix ms/call | ms/call speedup | post-fix `tokens_emitted` | post-fix `constrained_dead_ends` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke_hero_01 (0) | 3 | 29209.7 | 9736.6 | 8 | 29154.1 | 3644.3 | 2.67x | 13 | 0 |
| smoke_button_01 (1) | 4 | 29378.1 | 7344.5 | 43 | 28460.6 | 661.9 | 11.10x | 76 | 0 |
| smoke_callout_01 (2) | 4 | 29186.4 | 7296.6 | 8 | 29112.9 | 3639.1 | 2.00x | 13 | 0 |

All three still hit the 30s deadline (`total_ms` ≈ 30000.0-30000.3ms,
matching the finding's ~30.0-30.0s and PR #1190's ~30.0-30.5s `latency_ms`).
Pre-fix, all three records *dead-ended* (`constrained_dead_ends=1`) at a
short, identical prefix (`root = Card([b1` / `root = Card([b1,`, ~7-8
tokens) — the swallowed deadline masqueraded as "no legal continuation
here." Post-fix, **zero dead-ends on any record** — the compiler-tree search
makes genuine, sustained forward progress (13-76 tokens emitted across
8-43 completed `build_completion_forest` calls) and is still mid-decode,
correctly interrupted, when the 30s wall fires.

## Root cause (direct cProfile, offset-0 record, real decode path — not a synthetic prefix)

```
43,603,521 function calls in 34.069s (30.000s wall inside generate_batch_requests)

build_completion_forest                              3 calls   28.866s cumulative
  _openui_completion_domain (pack.py:273)             3 calls   28.866s
    _build_openui_completion_forest (compiler_draft)  261 calls 28.832s
    _tail_from (pack.py:392)                           31 calls  28.802s  <- witness search entry
      _tail (pack.py:402, lru_cache-wrapped)        7,386 calls  28.801s  <- recursive fan-out
        OpenUIIncrementalEngine._sync (engine.py:210) 23,532 calls 18.443s
          _refresh_accepts (engine.py:148)            23,530 calls 12.891s
            InteractiveParser.accepts() [lark]         23,530 calls 12.625s  <- DOMINANT COST
              (534,955 probe feed_token() calls + 1.5M copy.copy() calls
               into scratch parser-state copies, lark internals)
          _lex_tokens -> _lex (engine.py:122/132)      34,222 calls  5.511s / 5.493s
            BasicLexer._build_scanner (lark)                2 calls  0.007s  <- CONFIRMS CACHE HIT
OpenUIIncrementalEngine.__init__ (engine.py:93)        10,967 calls  3.263s
  Path.resolve()/os.path.realpath (per-init, pre-lru_cache-key)   ~2.9s + ~2.2s
lang_core._invoke (Node DSL bridge)                       307 calls  0.545s  <- still cheap
```

**`_build_scanner` was called exactly 2 times in this entire 30s record
decode (0.007s total)** — the lexer cache is genuinely hit on this real
decode path; PR #1173's fix is working exactly as designed here, not
"concentrated on cache hits this path doesn't exercise." The Node bridge is
also still cheap (307 round trips, 0.545s), matching PR #1172's finding.

The ~29s is now dominated by two costs neither prior fix touched:

1. **Lark's own `InteractiveParser.accepts()`** (12.6s of the 28.9s
   `compiler_ms`, called 23,530 times) — it determines the legal next
   terminals by copying the parser state and probing every candidate
   terminal (`lalr_parser_state.py:feed_token`/`copy`: 534,955 and 1.5M
   calls respectively). This is Lark-internal cost, orthogonal to lexing.
2. **The recursive `_tail_from`/`_tail` witness search's raw call-count
   fan-out** (pack.py:392-428): a fresh `@lru_cache(maxsize=64)`-wrapped
   `_tail` closure is created **per top-level candidate, per `_tail_from`
   call** (31 calls this record), so the memoization never carries across
   sibling candidates or across the decode loop's 8 outer iterations — the
   same grammar state can get re-explored (and re-pay the full `_sync` +
   `accepts()` cost) from scratch under a different top-level candidate.
   That is the multiplier turning 261 `_build_openui_completion_forest`
   calls into 23,532 `_sync` calls.
3. Smaller, secondary: `OpenUIIncrementalEngine.__init__` is constructed
   fresh 10,967 times in this one record's decode (once per recursive
   `_build_openui_completion_forest` call, further multiplied by however
   many predicates re-derive an engine), and each construction pays a real
   `Path.resolve()` syscall (~2.9s `pathlib.resolve` + ~2.2s
   `os.path.realpath` cumulative) **before** the string result even hits
   `_load_parser`/`_load_lexer`'s `lru_cache`.

## Answer to the task's root-cause question

**This checkpoint's compiler-tree search genuinely needs well beyond 30s per
record, independent of lexer cost.** The lexer-cache fix's benefit is real
and *is* being exercised on this exact decode path (not concentrated
elsewhere) — it cut per-call `compiler_ms` by 2.0x-11.1x, consistent with
(and in the button record's case, exceeding) PR #1173's own 2-2.1x
micro-benchmark claim. But removing the lexer-rebuild tax exposed the next
cost layer: Lark's `accepts()` probe plus the witness search's un-shared,
per-candidate memoization. The net effect on wall-clock is a wash for this
checkpoint's records — cheaper calls, but 2x-11x more of them complete
before the deadline, because genuine (previously-masked) decode progress now
happens instead of an early swallowed-timeout dead-end.

## Why this is a finding, not a patch

Same precedent as PRs #1171/#1172: `pack.py`/`compiler_draft.py`/`engine.py`
back `dsl.grammar_capabilities` (v3, watched in `versions.json`) and back
every default constrained-decode path (I3's witness search). A real fix
(sharing the `_tail` memoization cache across top-level candidates within
one `_tail_from`/decode-step call, and/or caching the resolved grammar path
outside `OpenUIIncrementalEngine.__init__`) changes the witness-search's
memoization scope and needs a benchmark-backed unit test proving `n_paths`/
`coverage`/witness *content* stay byte-identical (I6/I3) before it can be
trusted blind — deferred to `improve-openui-harnesses`, not attempted here.

## Proposed fix sketch (not applied this session)

1. Hoist the `@lru_cache`-wrapped `_tail` closure in `pack.py:402` out of
   the per-candidate loop in `_openui_completion_domain` (pack.py:430-451)
   so all top-level candidates for one decode position share one witness
   cache. `nodes_left`'s per-candidate fairness budget (16 nodes each, see
   the existing comment at pack.py:395-399) is a *novel-exploration*
   counter that's already skipped on a cache hit (the `@lru_cache` wrapper
   never calls the decrementing function body on a hit) — sharing the cache
   should only ever give a later candidate a free win on states an earlier
   candidate already proved, never fewer real chances. Needs a test
   confirming `nodes_left` semantics and witness content are unchanged.
2. Cache the resolved grammar path once (module-level, keyed on the raw
   `grammar_path` string) instead of calling `Path(...).resolve()` inside
   every `OpenUIIncrementalEngine.__init__`.
3. Investigate whether Lark's `InteractiveParser.accepts()` itself can be
   memoized per parser-automaton state (larger, Lark-internals-facing
   change; out of scope for a quick sketch here).
4. Re-run this session's protocol after any of the above to confirm
   `compiler_ms`/`_sync` call counts drop with zero change to `n_paths`,
   `coverage`, or emitted token sequences.

## Scope note

- Diagnostic only. No `--ship-gates` scoreboard claim, no promotion, no
  MODEL_CARD update.
- No harness code changed this session — `python -m
  scripts.verify_version_stamps --check` → `ok (vs HEAD; 0 changed file(s),
  0 component(s) touched)`.
- The monkeypatch scripts (`diag_deadline_stats.py`,
  `diag_deadline_profile.py`) live under the session scratchpad, not the
  repo — reproduction scaffolding, not a reusable harness tool, same
  precedent as PRs #1171/#1172's diagnostic scripts.
- `outputs/` artifacts (checkpoint, train data, eval run dirs) reused from
  PR #1190's session are gitignored, not committed.

## Validation

```text
python -m scripts.verify_version_stamps --check
# version-stamps: ok (vs HEAD; 0 changed file(s), 0 component(s) touched)

python -m scripts.repo_policy
# repo-policy: ok (tracked + untracked)

python -m scripts.verify_decode_invariants
# exit 0, agent_surfaces/canonical_defaults/strict_policies/weakening_levers unchanged
```

No source files were touched this session, so no `pytest`/`ruff` targets
apply beyond the eval/diagnostic runs above.

## Next steps (named lever for the following iteration)

1. Apply proposed-fix-sketch item 1 (share the `_tail` memoization cache
   across top-level candidates within one decode position) via
   `improve-openui-harnesses`, with a unit test asserting witness content
   and `nodes_left` fairness are unchanged, then re-run this exact protocol
   to measure the `compiler_ms`/`_sync`-call-count delta.
2. Re-run the seeded multi-rep `lever-hard-decode-timeout-wall` protocol on
   a non-smoke-scale checkpoint (PR #1190's next-step #2, still open) to see
   whether the honest 0/3 meaningful rate and the >30s-genuinely-needed
   finding here are specific to this 16-step scratch checkpoint or persist
   at exposure12's originally-reported quality-champion training recipe.

## Cleanup note

No new `outputs/` artifacts were created this session (reused PR #1190's
on-disk checkpoint/train-data/eval dirs, all gitignored). Diagnostic scripts
live under the session scratchpad only.

Captured: 2026-07-28T12:16:00Z
