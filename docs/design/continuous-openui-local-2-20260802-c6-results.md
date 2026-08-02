# Continuous autotrain cycle 6 results (2026-08-02, loop `continuous-openui-local-2`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local-2` (predecessor cycle: c4) |
| Campaign | `continuous-loop-20260802-continuous-openui-local--c94ddb78-c6` |
| Cycle role / intent | `promotion` / `retry_measurement` -- automatic frozen replay (attempt 1/2) |
| Replays | `d136c228a5a967e6884abdbfd2d5de46f6700bd1a110ccdd88fc772a07d30a1f` (c4's frozen confirm/control manifest) |
| Upstream / integration | `b8188a49` / `90021cad` |
| Device | CPU |
| Train | reused from c4 -- no retraining this cycle (frozen checkpoint replay) |
| Eval | `e938_role_safe_all_targets_v2`, suites `smoke,held_out` |
| Budget | `max_wall_minutes=1.1667` (~70s) per experiment |

## Two separate findings this cycle: a driver bug (fixed) and a reproduced timeout (not a bug)

This session's job was c4's queued `retry_measurement` for its champion
confirmation cycle (`held_out` wall-timeout on both arms -- see
[`continuous-openui-local-2-20260802-c4-results.md`](continuous-openui-local-2-20260802-c4-results.md)).

**First invocation (pre-fix): a hard crash, before any subprocess ran.**
`_apply_frozen_replay` raised `RuntimeError('unsupported automatic frozen
replay arm: confirm')`. Champion-queue matrices (Change C/D) name their
recommended arm `-confirm` or `-promote`, not a `_SCREENING_ARM_BANK` slug; a
frozen replay of one of those arms rebuilt its successor matrix through the
generic thrash-rotation branch (since `confirm_levers`/`promote_levers` are
never set on the `retry_measurement` path), which never contains a
`-confirm`/`-promote` experiment id for `_apply_frozen_replay` to patch. This
is the first cycle in this lineage to attempt a frozen replay of a
confirmation cycle, so the gap was previously unexercised (`v60`'s slug-recovery
fix only covered hyphenated `_SCREENING_ARM_BANK` names).

Fixed in `scripts/run_autotrain_continuous.py`
(commit `90021cad314162099f2b3d80389e1b7c0287ba04`): added
`_frozen_replay_champion_levers`, which recovers the frozen candidate's
champion-queue slug and, for `confirm`/`promote`, rebuilds
`confirm_levers`/`promote_levers` from the frozen candidate's own lever knobs
so the successor matrix is built through the same branch that produced the
original arm (`_apply_frozen_replay` overwrites every knob afterwards, so this
only needs to shape the matrix). `_apply_frozen_replay` now also accepts these
two slugs via `_CHAMPION_QUEUE_REPLAY_SLUGS`. Fails closed with a clear
`RuntimeError` if the frozen candidate carries no recognizable lever knobs.
4 new regression tests
(`test_frozen_replay_champion_levers_ignores_bank_slug_candidates`,
`test_frozen_replay_champion_levers_recovers_confirm_and_promote`,
`test_frozen_replay_champion_levers_fails_closed_on_empty_knobs`,
`test_frozen_replay_resolves_champion_confirm_arm`) plus the full
`tests/test_scripts/test_run_autotrain_continuous.py` suite (106 passed, 1
skipped) and `verify_version_stamps --check` both ran clean before commit.
`versions.json`: `harness.autoresearch.experiment_campaign` `v61 -> v62`.

**Second invocation (post-fix): the crash is gone, the retry runs correctly,
and honestly reproduces the same wall-timeout as c4.** The matrix rebuild
worked as intended -- `matrix-proposal.json` contains a `-confirm` arm with
the correct frozen levers (`component_edge_loss_weight=1.0`,
`structural_aux_head_profile=component-edge`), and the successor manifest's
`replay_of_manifest_sha256` correctly points back at the exact frozen digest
from c4. The retry also correctly reused c4's completed training (no
`train_summary.json` was written this cycle; `evaluate_model` was invoked with
`--checkpoint .../c4-{confirm,control}/checkpoints/last.pt`), matching
`contracts.md`'s "training completed but evaluation did not -> frozen replay
resumes at evaluation instead of retraining" rule.

## Smoke suite completed and tied again (`n=3`)

| Metric | control | confirm | delta |
| --- | --- | --- | --- |
| `parse_rate` | 1.0 | 1.0 | 0.0 |
| `meaningful_program_rate` | 0.3333 | 0.3333 | 0.0 |
| `structural_similarity` | 0.4167 | 0.4167 | **0.0 (tied)** |
| `binder_reference_f1` | 0.9524 | 0.9524 | 0.0 |
| `latency_ms_p50` | 9779.53 | 10369.86 | +590.33 (confirm slower) |

## `held_out` again did not finish on either arm

`decode_progress.json` for both arms: `status=interrupted`,
`measurement_complete=false`, `processed_record_n=3` of 5, mean per-record
decode time ~7.2-7.4s this attempt (c4 measured ~11s/record; same order of
magnitude, with some run-to-run variance and less contention now that no
retraining competed for the arm's wall budget). Both arms' `evaluate_model`
stage was killed by the harness wall-time guard again, structurally identical
to c4. No `--ship-gates` scoreboard was written for either arm.

**Root cause (established this cycle, not a budget-sizing bug):** all four
snapshots in `_DEFAULT_EVAL_VERSION_CANDIDATES` carry the same `smoke=3` /
`held_out=5` suite sizes, so `_feasible_eval_version`'s candidate search
cannot find one that fits promotion's 8-record requirement inside the ~70s
arm budget -- it correctly falls back to the default per its documented
design instead of raising. `stage_wall_minutes_for_role` returns the
policy-max (`3`, clamped to `MAX_RUN_MINUTES`) for **both** `screening` and
`promotion` roles, and `_arm_wall_minutes`'s symmetric two-arm split off
`MAX_HARNESS_WALL_SECONDS` already dominates that value for both roles alike
-- there is no slack to reallocate toward promotion's larger suite set without
either violating the fixed `MAX_RUN_MINUTES` invariant or dropping below the
required 1-control/1-candidate arms. This is genuinely the same class of
problem as the parked `continuous-openui-local` lineage's `test_data_scaleup_v1`
finding (`v61` history), just triggered at a smaller record count (8 vs 12)
because it is decode-compute-bound, not budget-arithmetic-bound. **No fix was
attempted for the timeout itself** -- there is no smaller `held_out` suite to
substitute without weakening the ship gate's held-out sample size, and that is
out of scope.

## SDLC Phase A: `NON_POSITIVE` (model), harness fix tracked separately

```text
SDLC_PHASE_A NON_POSITIVE campaign=continuous-loop-20260802-continuous-openui-local--c94ddb78-c6
  reason=measurement_incomplete:c20260802-continuous-openui-local--c94ddb78-c6-control:missing_scoreboard
  reason=measurement_incomplete:c20260802-continuous-openui-local--c94ddb78-c6-confirm:missing_scoreboard
  reason=wall_timeout:7f71efd9c329ecb07c163cff63c32957e40b6da19f05d86d2e628d3c619e7c62
  reason=empty_metrics:7f71efd9c329ecb07c163cff63c32957e40b6da19f05d86d2e628d3c619e7c62
  reason=wall_timeout:a809089ef399ccf0f751c57d2bb2b5b592e6043961c153683edd702c2bb1e25b
  reason=empty_metrics:a809089ef399ccf0f751c57d2bb2b5b592e6043961c153683edd702c2bb1e25b
  reason=primary_metric_null_or_worse:held_out.structural_similarity:control=0.4166666666666667 candidate=0.4166666666666667 improvement=0.0
```

`sdlc_delivery.json`: `positive=false`, `stack_layer=false`,
`stack_action=no_stack_layer_non_positive`, `measurement_complete=false`.
`climb_state=inconclusive`, `ship_state=blocked` in `cycle_handoff.json`. As
with c4, the reported `held_out.structural_similarity` tie is the completed
smoke value carried forward as a placeholder, not a genuine held_out
measurement -- treat this cycle as "no comparison happened" on the model
question. The **code-level** question (does the frozen-replay driver crash on
a champion-confirmation retry?) is answered and fixed, tracked as a harness
delta independent of this cycle's model non-positivity.

`champion_queue.jsonl`'s `component-edge` entry
(`champ-continuous-openui-local-2-3-0d77af2fb9002464`) was already
`status=rejected` as of c4's completion (`resolved_at=2026-08-02T12:05:26Z`,
`confirm_attempts=1`); this session's `retry_measurement` replayed the frozen
c4 manifest without re-touching the champion queue (`open_champion` is forced
`None` on the `retry_measurement` path).

## Handoff actions

`cycle_handoff.json` emitted two actions:

1. `retry_measurement` (owner `autotrain`, frozen manifest
   `a28fd1fe59a94f1a2a70c8fc78fed660e66c1da85e79d1bcfad0cb96c4f322c2`,
   "1/2 consecutive frozen replays so far") -- steering/execution action, not a
   predecessor prerequisite; left queued for the next cycle per `contracts.md`.
2. `document` (owner `documenting-experiment-results`) -- **acknowledged this
   cycle** with this doc pair as evidence.

`harness_signals: []` in the driver's own `cycle_handoff.json` for c6 (the
wall-timeout was already diagnosed as `infrastructure`/`wall_timeout` at c4;
no new canonical-family harness signal was raised by the timeout itself this
cycle). The driver-crash bug this session fixed was found and repaired
directly from process evidence (the crash traceback), not surfaced as a typed
`HarnessSignalV1` -- it was blocking the retry from running at all.

`checkpoint_documentation_required=false` this cycle -- no new checkpoints
were created (training reused from c4, which already updated
`docs/MODEL_CARD.md`/README).

## Recommendation for cycle 7

1. **infrastructure (priority):** one more automatic `retry_measurement`
   (`2/2`) is queued for the same frozen manifest
   (`d136c228a5a967e6884abdbfd2d5de46f6700bd1a110ccdd88fc772a07d30a1f`). Given
   the timeout reproduced identically after the driver-crash fix -- with
   training already free, the best case for this recipe -- the second attempt
   is very unlikely to complete `held_out` either. Expect it to reach
   `measurement.max_consecutive_frozen_replays=2` and trigger the typed
   `repair_harness` handoff action. That action should **not** be answered
   with further budget-arithmetic changes: this cycle's investigation
   established the timeout is decode-compute-bound, not a budget-sizing
   defect, so there is no legitimate code-level repair available within the
   fixed `MAX_RUN_MINUTES` invariant. The honest disposition is to acknowledge
   `repair_harness` with this cycle's and c4's `decode_progress.json` evidence
   as the reason no further automatic replay is possible.
2. **policy (new, worth raising to the loop owner):** should promotion/confirm
   cycles structurally exclude `held_out` from continuous CPU-sandbox
   screening, running it only under a larger, explicitly-authorized compute
   budget outside `MAX_RUN_MINUTES`? Repeatedly rediscovering the same
   decode-compute-bound timeout on every champion confirmation is not making
   progress. `policy.v1.json`'s `promotion_suites=[smoke, held_out]` is an
   explicit choice, not a bug -- this is a scope decision, not something this
   session should force.
3. **model:** no new model evidence this cycle. c3's `component-edge` lever
   remains unconfirmed on `held_out` and its champion-queue entry is now
   `rejected`. Do not re-enqueue it without a plan for how its `held_out`
   confirmation could actually complete.
4. **lineage-wide:** with `continuous-openui-local-2`'s screening pool (c1
   `bounds`, c2 `component-plan`, c3 `component-edge`) largely overlapping the
   parked `continuous-openui-local` lineage's 13-cycle screen, and this
   lineage's only promotion-role attempt (c4/c6) now twice blocked on the same
   structural `held_out` timeout, continued single-lever screening cycles here
   are low marginal value. Recommend the loop shift focus toward either (a)
   the decode-compiler investigation itself -- the actual bottleneck behind
   both lineages' timeouts -- or (b) a bottom-up SDLC review/closeout of
   accumulated non-positive evidence across both lineages, rather than
   generating more screening cycles that cannot reach a confirmed promotion
   under the current suite/budget combination.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local--c94ddb78-c6/`
- Runs: `.../runs/c20260802-continuous-openui-local--c94ddb78-c6-{control,confirm}/`
- Handoff: `.../cycle_handoff.json`
- SDLC delivery: `.../sdlc_delivery.json`
- Champion queue entry: `outputs/autoresearch/loops/continuous-openui-local-2/champion_queue.jsonl` (`entry_id=champ-continuous-openui-local-2-3-0d77af2fb9002464`, `status=rejected`)
- Fix commit: `90021cad314162099f2b3d80389e1b7c0287ba04` (`scripts/run_autotrain_continuous.py`, `tests/test_scripts/test_run_autotrain_continuous.py`, `src/slm_training/resources/versions.json`)
- JSON twin: `continuous-openui-local-2-20260802-c6-results.json`
- Predecessor cycle: [`continuous-openui-local-2-20260802-c4-results.md`](continuous-openui-local-2-20260802-c4-results.md)
