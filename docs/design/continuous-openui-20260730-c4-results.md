# Continuous autotrain cycle 4 results (2026-08-01): champion pipeline + a harness fix

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260730` |
| Campaigns | `continuous-loop-20260801-c2` (enqueue) → `c3` (confirm) → `c4` (promote attempt) → `c5` (successor, crashed pre-fix) |
| Source | `d290da7f508950c9e56bede526f8538383c368e1` |
| Device | CPU, steps 60-62 |

## Champion pipeline (screening → confirm → promote)

| Stage | Campaign | Control | Candidate | p50 latency Δ | mpr held | Disposition |
| --- | --- | --- | --- | ---: | --- | --- |
| Enqueue | `c2` | `c20260801-c2-control` | `c20260801-c2-canvas` (`compact_active_canvas=true`) | **-1012.77ms** | 0.667 → 0.667 | `positive_no_tracked_delta_skip_stack` → `CHAMPION_ENQUEUE` |
| Confirm | `c3` | `c20260801-c3-control` | `c20260801-c3-confirm` | **-27.75ms** | 0.333 → 0.333 | `CHAMPION_STATUS status=confirmed` |
| Promote (attempt 1) | `c4` | — | `c20260801-c4-promote` | n/a | n/a | `PROMOTE_FORMAL_BLOCK skip_execute` → `harness_failure` |

`compact_active_canvas=true` holds quality (parse=1.0, meaningful_program_rate unchanged) while reducing decode latency on both the screening and confirm arms — a genuine efficiency win, correctly routed to the champion queue instead of a code-diff PR (no code changed; it's a runtime lever combination).

## Promote attempt 1: environment gap, not a harness bug

`c4`'s formal preflight came back `status=unknown` because this fresh container had no Lean/`lake` toolchain installed — `PROMOTE_FORMAL_PREFLIGHT` fails closed by design (`ensure_promote_formal_preflight`), so the arm never trains. The champion queue correctly classifies this as `harness_failure` (does not burn a real `promote_attempts`). Fixed for this container: installed `elan` and ran `lake build` inside `src/leverproof_lean/` (`leanprover/lean4:v4.30.0`, no `mathlib` dependency, builds in a few seconds).

## Harness bug found and fixed: non-terminal promote matrix poisons the next cycle

Running the driver again after installing Lean rotated to a different screening candidate (`THRASH_ROTATE`) for cycle `c5`, whose predecessor is `c4`. That crashed:

```
CYCLE_ERROR ValueError('latest hypothesis matrix has no terminal feedback; run a
matrix member before forming its successor')
```

**Root cause:** `run_autotrain_continuous.py`'s promote path calls `autoresearch hypothesize` (forming and locking a matrix) *before* checking the formal preflight. When the preflight isn't proved, it sets `order = []` and skips executing any arm — so `autoresearch run --execute` never runs, and the matrix never gets `outcome_diagnosed` / `hypothesizer_feedback_recorded` events. `cmd_hypothesize`'s predecessor-lineage walk (`_latest_formed_matrix` → `_hypothesis_feedback`) then finds a formed-but-feedback-less matrix and raises — for every future cycle in the loop, not just the one right after the block.

**Fix:** added a new `autoresearch block --campaign-id <id> --experiment-id <id> --reason <reason>` subcommand (`scripts/autoresearch.py::cmd_block_experiment`) that records `experiment_started` → `experiment_finished` (`status=failed`, `error=<reason>`) → `outcome_diagnosed` (`diagnose_outcome` correctly classifies this as `target=infrastructure`) → `hypothesizer_feedback_recorded`, idempotently, for a matrix member that never executed. `run_autotrain_continuous.py` now calls it immediately after setting `order = []` in the `PROMOTE_FORMAL_{BLOCK,TIMEOUT_INCONCLUSIVE}` path.

**Verification:** two new regression tests in `tests/test_autoresearch/test_harness.py`:
- `test_block_experiment_records_terminal_feedback_for_unstarted_arm` — direct contract test of `cmd_block_experiment` (events recorded, feedback produced, idempotent replay).
- `test_hypothesize_after_block_experiment_does_not_raise` — reproduces the exact predecessor/successor loop-lineage shape (`loop-c1` → `loop-c2`, matching `CampaignSpec` cycle/loop validation) and asserts `cmd_hypothesize` no longer raises after the predecessor's only arm was closed via `block`.

Version stamp: `harness.autoresearch.experiment_campaign` v28 → v29. `scripts/autoresearch.py` is also watched by `harness.preference.remine_campaign`; recorded a `no-bump` note there since remine's own behavior is unchanged.

## Next-run priorities

1. Re-run the continuous driver so the confirmed champion (`compact_active_canvas=true`, fingerprint `7dc23b6cf0129a66`) gets a real promote attempt now that the Lean formal preflight can actually build.
2. Watch for further promote-path gaps now that a formal block no longer poisons lineage; route any new disposition problem as its own typed `HarnessSignalV1`.
3. Do not promote RL; ship gates fail by design on fixture n.

## Artifacts

- Campaigns: `outputs/autoresearch/continuous-loop-20260801-c2/` .. `c5/` (not tracked — `outputs/` is gitignored)
- Champion queue: `outputs/autoresearch/loops/continuous-openui-20260730/champion_queue.jsonl` (ephemeral, container-local)
- JSON twin: `continuous-openui-20260730-c4-results.json`
- Code: `scripts/autoresearch.py`, `scripts/run_autotrain_continuous.py`, `tests/test_autoresearch/test_harness.py`
