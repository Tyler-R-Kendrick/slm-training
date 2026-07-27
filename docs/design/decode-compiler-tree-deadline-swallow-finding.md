# Finding: compiler-tree decode swallows the deadline as a fake grammar dead-end

**Honesty:** `fixture_or_scratch`, isolated per-record diagnostic (suite `n=3`,
1 rep each). **Not ship. Not a fix — a diagnostic finding**, same status as
[`decode-timeout-hang-seed44-steps72-finding.md`](decode-timeout-hang-seed44-steps72-finding.md).

## Task

Follow-up to
[`lever-hard-decode-timeout-wall-measured-results.md`](lever-hard-decode-timeout-wall-measured-results.md)
(PR #1167, `harness.model_build.eval` v63): that doc's own "Decision" section
says *"Next quality work: improve model so hero finishes inside budget, not by
relaxing the wall."* This diagnostic characterizes **why** the exposure12
quality-champion hero decode grazes/blows `decode_timeout_seconds=30` under
the hard wall.

## Reproduction

Checkpoint wasn't on disk (`outputs/` is gitignored, fresh checkout), so it
was rebuilt from scratch in a fresh `.venv-diag` (Python 3.12) + `npm ci` in
both the repo root and `src/apps/openui_bridge` (AgentV SDK + DSL bridge, both
needed for grammar validation/publish):

```bash
python -m scripts.build_train_data --source fixture --profile strict \
  --max-records-per-parent 12 --version lever_exposure12_v1 \
  --output-root outputs/data/train
# 107 records, 5 abstraction_ladder -- matches
# lever-exposure-cap12-abstraction-ladder-measured-results.md exactly

python -m scripts.train_model \
  --train-dir src/slm_training/resources/data/train/lever_exposure12_v1 \
  --model twotower --context-backend scratch --steps 16 --batch-size 2 \
  --lr 1e-3 --structural-bias 1.5 --seed 47 \
  --run-id exp_lever_data_exposure12_s16_lr1e3_bs2_sb15_seed47 \
  --no-sync-checkpoints --device cpu
# 10.1s wall
```

Then, instead of running the full 3-record smoke suite (which mixes each
record's `DecodeStats` into one aggregate), each record was evaluated **in
isolation** (`--eval-limit 1 --eval-offset {0,1,2}`) so the existing
`metrics["decode_stats"]` aggregate in the scoreboard is exactly that one
record's stats. No new instrumentation was needed — every field used below
(`compiler_ms`, `forwards_count`, `denoiser_ms`, `constrained_dead_ends`, …)
already exists in `src/slm_training/models/decode_stats.py`.

```bash
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

## Measured: all three records hit the identical pattern

| record | meaningful | outcome | total_ms | compiler_ms | compiler_ms % | forwards | tokens_emitted | denoiser_ms | dead_ends | certified_fallbacks |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke_hero_01 (offset 0) | 0.0 | fallback_output | 30006.5 | 29209.7 | 97.3 | 3 | 7 | 0.0 | 1 | 1 |
| smoke_button_01 (offset 1) | 0.0 | fallback_output | 30006.5 | 29378.1 | 97.9 | 4 | 8 | 0.0 | 1 | 1 |
| smoke_callout_01 (offset 2) | 1.0 | fallback_output | 30010.2 | 29186.4 | 97.3 | 4 | 8 | 0.0 | 1 | 1 |

`backbone_ms` (~65-69ms) and `trie_ms` (~67-72ms) are negligible; `denoiser_ms`
is exactly 0.0 on every record. **The wall-clock sink is `compiler_ms`, not
transformer forward passes or step count.** `compiler_search_mode` is
`greedy` (`compiler_lattice_recurrences` omitted-zero), so this is not a
combinatorial/backtracking search blowing up — it's ~3-4 sequential
`build_completion_forest` calls (`compiler_prefill_batches`) averaging
roughly **7-10 seconds each**. All three dead-end traces show the *same*
decoded prefix (`root = Card([b1` / `root = Card([b1,`), reason
`empty_completion_forest` — a separate, likely undertrained-checkpoint
symptom (steps=16 is smoke-scale) worth flagging but out of scope here.

## Root cause chain

1. **`evaluate_model`'s default policy silently overrides the checkpoint's
   own trained decode mode.** `DEFAULT_EVALUATION_POLICY` in
   `src/slm_training/levers.py:91` is `STRICT_COMPILER_TREE_POLICY_ID`, and
   `STRICT_COMPILER_TREE_POLICY` (levers.py:100-108) sets
   `compiler_decode_mode="tree"`. This checkpoint's own `last.meta.json`
   stores `compiler_decode_mode = "off"` (it was trained without the
   compiler-tree path) — yet every eval run's `decode_stats` show
   `compiler_ms`/`compiler_candidates`/`compiler_prefill_batches` populated
   and `dead_end_trace.phase == "compiler_tree"`, proving the tree path ran
   anyway. The eval harness's canonical default always takes this path
   regardless of checkpoint provenance ("the compiler-tree bundle is the
   canonical default" per the levers.py comment).
2. **The compiler-tree per-position loop has zero `check_decode_deadline()`
   calls.** `src/slm_training/models/twotower.py`: the `while len(prefix) <
   length:` loop inside `_compiler_ltr_decode_one` (def at line 10254, loop
   at line 10329) and `_compiler_ltr_decode_batch` (lines 10710-11316)
   contain **no** `check_decode_deadline()` call anywhere (grep-confirmed
   zero hits in both ranges) — unlike `_constrained_ltr_repair` (line
   4900-4908) and the MaskGIT loop (line 13265-13268), which both got the
   cooperative deadline check added in PR #1167 (`model.twotower` v261).
3. **`build_completion_forest` swallows a deadline `TimeoutError` as a
   grammar dead-end.**
   `src/slm_training/dsl/grammar/fastpath/compiler_draft.py:2306-2314`:

   ```python
   try:
       domain = cache.get(cache_key) if cache is not None else None
       if domain is None:
           domain = GrammarCapabilityAdapterV1(get_pack()).completion_domain(request)
           if cache is not None:
               cache[cache_key] = domain
   except Exception:  # noqa: BLE001 - constrained callers fail closed below
       return CompletionForest((), "none")
   ```

   `completion_domain(request)` ultimately calls the Node DSL bridge
   (`lang_core.parse`/`validate`, compiler_draft.py:1143-1151) with no
   duration bound of its own. Any `TimeoutError` raised inside that call —
   including one from the eval harness's own SIGALRM interval re-fire — is
   caught by this bare `except Exception` and silently converted into an
   **empty completion forest**, which the caller (twotower.py:10357+) records
   as a legitimate grammar dead-end (`constrained_dead_ends += 1`, reason
   `"empty_completion_forest"`) instead of a timeout.

## Interpretation

The hard wall from `harness.model_build.eval` v63 does still cut this decode
path off near the 30s budget (consistent with the ~30-33s `max_latency`
measured in `lever-hard-decode-timeout-wall`), but for the compiler-tree
path it does so **by accident**: the deadline exception most plausibly lands
inside `completion_domain()`, is caught at compiler_draft.py:2313, and gets
recorded as "the grammar had no legal continuation here" rather than a
classified timeout. That cascades into the certified/minimal-fallback path
(`certified_fallbacks=1`, `compiler_fallbacks=1` on every record measured
this session), producing `decode_outcome=fallback_output` with
`stop_reason="completed"` — exactly the pattern in 2 of 3 seeded reps in
`lever-hard-decode-timeout-wall-measured-results.json`, while the 1/3 clean
`runtime_timeout` rep is consistent with the interrupt instead landing
somewhere that *does* propagate (e.g. the LTR-repair path, which does call
`check_decode_deadline`).

## Scope correction vs. prior docs

Prior docs (`lever-decode-latency-exposure12-quality-champ`,
`lever-decode-reproducibility-exposure12-seed47`) characterized only the
**hero** record as pathological (160-277s pre-wall). This session's fresh
isolated reruns show **all three** smoke records converging on the identical
~30.0s / ~97-98% `compiler_ms` / one-dead-end pattern under today's default
policy. This is n=1 rep per record, not a multi-rep confirm, so it does not
by itself prove hero's historical 160-277s excursions share this exact cause
— but the mechanism measured here is sufficient on its own to explain
wall-grazing regardless of record identity.

## Why this is a finding, not a patch

Same precedent as
[`decode-timeout-hang-seed44-steps72-finding.md`](decode-timeout-hang-seed44-steps72-finding.md):
the affected files are watched by `dsl.grammar_capabilities` (v3) and
`model.twotower` (v261) in `src/slm_training/resources/versions.json`, and
the fix touches the eval harness's core constrained-decode safety net used by
every default `evaluate_model` invocation. It should go through
`improve-openui-harnesses` with a fixed unit test (analogous to
`tests/test_harnesses/model_build/test_decode_deadline.py`, but exercising
the compiler-tree path specifically) rather than being patched blind here.

## Proposed fix sketch (not applied this session)

1. Add `except TimeoutError: raise` before the bare `except Exception:` in
   `compiler_draft.py:2313` so a deadline `TimeoutError` is never masqueraded
   as an empty completion forest.
2. Add `check_decode_deadline()` calls inside the per-position loop in
   `_compiler_ltr_decode_one` (twotower.py, loop starting ~line 10329) to
   match the LTR-repair/MaskGIT loops' cooperative coverage.
3. Re-run this same isolated per-record diagnostic (or the seeded multi-rep
   protocol from `lever-hard-decode-timeout-wall`) after the fix to check
   whether `decode_outcome` shifts to a correctly-classified
   `runtime_timeout` (if genuinely out of budget) or to a real on-time
   completion (if the swallow was masking otherwise-fast decode).

## Next steps

1. Fix per the sketch above via `improve-openui-harnesses` (bumps
   `dsl.grammar_capabilities` and `model.twotower` in `versions.json`).
2. Separately investigate why each `build_completion_forest` call costs
   roughly 7-10s of wall time on average (29.2-29.4s `compiler_ms` across
   only 3-4 `compiler_prefill_batches` per record) — Node-bridge round-trip
   cost vs. candidate-set size vs. cache misses were not isolated this
   session.
3. Re-run the champion multi-rep seeded eval
   (`lever-hard-decode-timeout-wall`'s exact protocol) after the fix to see
   whether hero `meaningful_program_rate` recovers without needing a 160s+
   unwalled run.

## Cleanup note

The `lever_exposure12_v1` train-data rebuild and the scratch checkpoint used
for this diagnostic are not committed (`outputs/` is gitignored; the
auto-published copy this session created under
`src/slm_training/resources/data/train/lever_exposure12_v1` was removed after
the runs above since it is reproduction scaffolding, not new curated training
data).

Captured: 2026-07-27T18:51:20.617309+00:00
