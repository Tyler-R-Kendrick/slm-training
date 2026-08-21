# Autotrain hill-climb recovery directive (one-shot, subagent swarms)

**Audience:** an LLM coding-agent parent orchestrating parallel subagent swarms
inside this repository (`slm-training`). This document is self-contained: every
number, path, contract, and law a swarm needs is stated here. Do not defer to
any prior conversation.

**Mission:** the continuous autotrain loop (`continuous-openui-local`) has run
527 cycles, queued 75 champion candidates, and promoted **zero**. Convert it
from a noise generator into a hill climber by (a) giving screening a metric
with statistical signal, (b) making progress accumulate across cycles,
(c) buying power with paired statistics and larger suites, (d) spending the
compute budget the law already allows, (e) lifting the structural metric with
deterministic decode levers, and (f) removing the operational self-blocking.

This is not a phased plan. Every swarm below is an independent, atomic,
one-shot task with exclusive file ownership. Launch all swarms in parallel;
the Integrator swarm merges and runs the global gates.

---

## 1. Ground truth (measured evidence — re-verify before trusting)

All paths are repo-relative. Re-derive any number you rely on.

| Fact | Value | Source |
| --- | --- | --- |
| Cycles run | 527 (loop `continuous-openui-local`) | `outputs/autoresearch/loops/continuous-openui-local/state.json` |
| Champion promotions | 0 of 75 queued (all `rejected`) | `outputs/autoresearch/loops/continuous-openui-local/champion_queue.jsonl` |
| Handoff verdicts | 411 rejected / 74 candidate_queued / 37 inconclusive over 522 handoffs | `outputs/autoresearch/continuous-loop-*/cycle_handoff.json` |
| Per-arm training | 22 steps, 3.36 s wall, CPU, final loss 8.04, 1,984 target tokens, cold start (`initialized_weight_count: 0`) | `outputs/autoresearch/continuous-loop-20260820-continuous-openui-local-8c0b60dd-c527/runs/*-control/train_summary.json` |
| Train data | `wf_smoke_v2`, **101 records** | same file, `train_dir`, `record_count` |
| Eval suite | `e938_role_safe_all_targets_smoke6_v1`, **n = 6** | run `scoreboard.json` |
| Screening primary | `smoke.structural_similarity`, `minimum_effect: 0.01` | `src/slm_training/resources/experiments/autotrain_climb/policy.v2.json` (v13) |
| Control-arm value across cycles | 0.035 → 0.577 (16× spread for the "same" baseline) | `cycle_handoff.json` reasons, `primary_metric_null_or_worse:...control=<x>` |
| Identical-arm cycles | 115 of 333 measured cycles had byte-identical control/candidate metrics (`mechanism_no_effect:quality_metrics_identical`) | same |
| Screen wins vs confirms | 171 `primary_metric_win` reasons, 230 `confirmation_rejected` | same |
| Budget chain | `MAX_RUN_MINUTES = 3` (`src/slm_training/levers.py`) → arm wall 70 s → 42 s eval + 8 s overhead + 20 s train floor; fitted decode 7 s | `outputs/autoresearch/continuous-loop-*-c527/thrash_timing.json` |
| Step cap vs usage | policy caps `screening_thrash_steps: 40`; fitter emitted 22; 22 steps took 3.36 s of a 20 s floor (~6.5 steps/s ⇒ ~130 steps fit the floor) | policy v13 + `train_summary.json` |
| Quality shape | `parse_rate` 1.0, `ast_node_f1` 0.179, `ast_edge_f1` **0.0**, `structural_similarity` 0.131, `meaningful_program_rate` 1/6, failure breakdown `low_component_recall` | run `scoreboard.json` |
| Ops tax | 3,377 `thrash_bank_compose` soft-heals; 60 `foreign_dirty_tree` escalations on `.serena/project.yml`; supervisor sleeps 3,600 s per blocked tick | `outputs/autoresearch/loops/continuous-openui-local/cycle_failures.jsonl`, `escalations.jsonl`, `supervisor.jsonl` |
| Lean screening bound | decidability floor n = 6 at α = 1/20 (sign test); `power_floor_n: null` — **decidability only, no power claim**; `promotion_authority: false` | `thrash_timing.json → decode_fit.screening_sample_size` |
| Knob bank | exhausted; loop's own rank-1 priority (confidence 0.95): "quality-arm bank is exhausted; preregister a distinct size-matched quality objective" | c527 `cycle_handoff.json` |

**Causal chain of the failure (why 527 cycles produced nothing):**
22 optimizer steps cannot move a model off random init (loss ≈ 8), so no knob
in the bank has a mechanism; a 6-sample eval of untrained models has a noise
floor (σ of the mean ≈ 0.12) an order of magnitude above the declared
`minimum_effect` of 0.01, so screening wins are coin flips; coin-flip wins
never replicate at confirm (regression to the mean), so the champion queue
rejects 75/75; nothing warm-starts, so cycle 527 begins where cycle 1 did.
The loop's verdict logic is *honest* — do not weaken it. Fix the experiments,
never the gates.

---

## 2. Adversarial critique of the naive fixes (read before designing)

Each fix below was proposed in a first-pass analysis and then red-teamed.
Swarms must implement the **corrected** versions.

1. **"Screen on training loss."** Wrong as stated: many bank knobs change the
   loss *function* (e.g. `binder_arity_loss_weight`), making training loss
   incomparable across arms. **Correction:** screen on **held-out NLL computed
   under one canonical, arm-independent loss definition** (the harness for
   this exists: `scripts/evaluate_loss_suites.py`). Direction = decrease.
   NLL screening is `claim_class: diagnostic` only; it never touches ship
   gates or promotion primaries.
2. **"Pair the evals to cancel variance."** The arms already evaluate the same
   6 fixtures with seeded decode; the deltas are already paired. The real
   problems are (a) n = 6 makes the exact sign test degenerate under ties —
   and 115/333 cycles are *all-ties*; (b) the Lean floor is a decidability
   floor, not a power floor. **Correction:** grow the suite AND use a paired
   test with tie handling (Wilcoxon signed-rank, or exact sign test on
   non-tied pairs with a minimum non-tie count), AND compute an empirical
   power floor (`power_floor_n`) from measured per-record variance instead of
   leaving it `null`.
3. **"Move to GPU (50–100×)."** Unverified hardware assumption (host is
   WSL2). **Correction:** detect the device at fit time; if CUDA exists route
   screening trains to it; the CPU fallback below stands on its own.
4. **"Raise MAX_RUN_MINUTES."** Heavy and unnecessary *first*: the loop uses
   3.36 s of a 20 s train floor, and the floor sits inside a 70 s arm wall
   inside a 180 s stage wall. **Correction:** exploit the unused legal budget
   (steps fitted to fill the floor; floor grown into eval slack) before any
   law change. If a cap change is later justified, it is exactly one constant
   in `src/slm_training/levers.py` plus `python -m scripts.repo_policy
   --sync-run-policy` — but that is out of scope for this directive.
5. **"ASHA / adaptive halving over the hypothesis bank."** Adaptive stopping
   collides with the preregistration law if the rule is chosen after outcomes
   are visible. **Correction:** multi-arm screening is legal when the arm set,
   seeds, and the (static) selection rule are locked in the campaign manifest
   *before* execution; results remain exploratory/diagnostic.
6. **"Warm-start from a champion."** Naive version confounds: a champion that
   trains every cycle is "older" than any cold candidate. **Correction:**
   control = champion + K extra steps with baseline knobs; candidate =
   champion fork + K extra steps with the knob applied. Identical K,
   identical data slice, identical params (or charged via `EG_params` per
   invariant VI). Cap cumulative epochs over the 101-record train set until
   the train-data swarm lands (overfitting guard), and keep the existing
   train/eval leakage checks mandatory.
7. **"Restore `.serena/project.yml` by hand."** Symptom patch: Serena rewrites
   that file on every activation (strips comments), and
   `_is_foreign_dirty_path` in `scripts/run_autotrain_continuous.py`
   deliberately blocks on it ("tracked config still blocks"). A human
   restoring it hourly is not a system. **Correction:** a typed self-heal (see
   SWARM-OPS).
8. **First-pass variance estimates were sloppy.** σ ≈ 0.3/record was inferred
   from *cross-cycle* control spread, which conflates fixture variance, seed
   variance, and knob-bleed. **Correction:** the statistics swarm estimates
   variance components from logged per-record eval data before choosing n;
   the numbers in this directive are priors, not conclusions.
9. **Missed compounding constraint:** `cadence.screening_cycles_per_promotion
   = 3` and `promotion_requires_prior_screening_win = true` mean promotion is
   unreachable while screening is noise, and `promotion_suite_n = 6` is as
   underpowered as screening. Fixing screening alone still dead-ends at an
   underpowered confirm tier; the statistics swarm owns both tiers.
10. **Throughput fixes alone are worth zero.** Faster zero-information cycles
    are still zero information. SWARM-OPS matters only because the other
    swarms make cycles informative.

---

## 3. Repo law digest (binding on every swarm; self-contained)

- **Hard run cap:** every train/eval/bench obeys `MAX_RUN_MINUTES = 3` in
  `src/slm_training/levers.py`. A timed-out/killed run is never evidence. Do
  not change the constant in this work.
- **Never weaken constrained decoding (I6):** production decode stays
  grammar-constrained, fail-closed. Levers may change *how* legal symbols are
  chosen, never *whether* output is legal. Singleton domains commit with zero
  forwards (I2). Speculation verifies against the grammar oracle before
  commit (I3/I5).
- **Parameter-efficiency (invariant VI):** compared arms are size-matched
  (`levers.require_size_matched_arms`) or the difference is charged
  (`EG_params` LCB ≥ 1). Never grow the model to green a gate.
- **Preregistration:** campaign arms, seeds, stopping/selection rules, and
  gates are locked in the `ExperimentCampaignV1` manifest
  (`src/slm_training/autoresearch/experiment_campaign.py`) before outcomes
  are visible. Deviations are append-only and exploratory.
- **Honest ship gates:** everything here is fixture-scale diagnostics, never
  a ship claim. Do not weaken any gate, `rico_held` n, or the
  fixture-vs-ship distinction to make CI green.
- **Version stamps:** changing any watched metric/gate/harness/matrix/
  data-builder file requires a component bump (or `no-bump:` history note) in
  `src/slm_training/resources/versions.json`, enforced by
  `python -m scripts.verify_version_stamps --check`.
- **External test cases:** JSON-shaped pytest inputs live mirrored under
  `src/slm_training/resources/test_cases/`; edit them there and run
  `python -m scripts.refresh_test_cases <target>`; finish with
  `python -m scripts.refresh_test_cases --check --changed`.
- **Docs follow experiments:** any validation train/eval run a swarm executes
  must land JSON + markdown under `docs/design/` with recipe metadata
  (device, steps, backend, suite n, honesty mode) in the same change.
- **Data-quality law:** any data build (`build_train_data`,
  `build_test_data`) requires reading its `quality_report.json`,
  `rejected.jsonl`, `synthesis_feedback.json` and acting on them — fix the
  synthesis harness, never the gates.
- **Repository organization:** new tracked paths follow
  `docs/repository-organization.md`; use `git mv` for relocations. Run
  `python -m scripts.repo_policy` and `.githooks/check-changed` before
  finishing.
- **Live loop safety:** a supervisor may be running
  (`scripts/run_autotrain_supervisor.py`, lock at
  `outputs/autoresearch/loops/continuous-openui-local/driver.lock`). Never
  kill it; land changes on the branch and let the next tick pick them up.
  Any non-allowlisted dirty tracked path parks the loop
  (`_is_foreign_dirty_path`), so commit your work when done.

---

## 4. Pinned shared contracts (fixed here so swarms cannot collide)

Swarms implement against these contracts without negotiating with each other.

### C1 — New screening metric identity

- Metric id: **`smoke.eval_nll`** (leaf `eval_nll`), float, **direction:
  decrease**, computed by the canonical loss-suite evaluator over the same
  published smoke suite the quality eval uses, under one fixed loss config
  that ignores all arm loss-weight knobs.
- Written into each arm's `scoreboard.json` under `suites.smoke.eval_nll` and
  surfaced in the handoff metrics maps exactly like existing leaf metrics
  (see `_metric_from_map` / `_leaf` in
  `src/slm_training/autoresearch/climb_policy.py`).
- `minimum_effect` for NLL screening: computed by SWARM-STATS as the measured
  minimum detectable effect (MDE) at the certified n and recorded in the
  policy artifact; placeholder until then: 0.05 nats.

### C2 — Climb policy artifact (single owner: SWARM-METRIC)

`src/slm_training/resources/experiments/autotrain_climb/policy.v2.json` bumps
`version: v13 → v14` with exactly these deltas (other swarms consume, never
edit):

- `screening_primary`: `{"metric": "smoke.eval_nll", "direction": "decrease", "minimum_effect": <MDE placeholder 0.05>}`.
- `screening_quality_secondary`: `{"metric": "smoke.structural_similarity", "direction": "increase"}` — recorded, never verdict-bearing at screening.
- `promotion_primary` stays `held_out.structural_similarity` (quality remains
  the promotion bar), but `measurement.promotion_suite_n` rises to the
  power-certified value from SWARM-STATS.
- `measurement.thrash_timing.screening_thrash_steps`: replace the static 40
  with `"fit_to_train_floor"` semantics (SWARM-BUDGET defines the fitter; the
  policy carries `screening_thrash_steps_max: 400` as the hard cap).
- `measurement.warm_start`: `{"enabled": true, "source": "climb_champion", "equal_extra_steps": true, "max_cumulative_epochs": 50}` (consumed by SWARM-CHAMPION).
- `measurement.paired_test`: `{"kind": "wilcoxon_signed_rank", "min_nontied_pairs": 5, "alpha": "1/20"}` (consumed by SWARM-STATS).
- `measurement.multi_arm`: `{"max_arms_per_cycle": 6, "shared_control": true, "selection_rule": "best_by_primary_then_smallest"}` (consumed by SWARM-MULTIARM).

### C3 — File / function ownership matrix (exclusive; conflicts are bugs)

| Owner swarm | Exclusive paths / symbols |
| --- | --- |
| SWARM-METRIC | `policy.v2.json` (whole file, applying every C2 delta), `climb_policy.py::primary_for_role`, `classify_positive_metrics`, NLL wiring in the eval stage of `scripts/run_autotrain_continuous.py` (`_run_arm_eval*` region only) |
| SWARM-BUDGET | `scripts/run_autotrain_continuous.py::_screening_thrash_steps`, the thrash-timing decode/steps fitter, device routing at train launch |
| SWARM-CHAMPION | warm-start/checkpoint plumbing: the arm train-launch config builder in `run_autotrain_continuous.py` (checkpoint init argument only), climb-baseline seeding in `src/slm_training/autoresearch/hillclimb.py`, `thrash_regime.py` climb-regime activation |
| SWARM-STATS | `src/slm_training/autoresearch/screening_sample_size.py` (power floor), new module `src/slm_training/autoresearch/paired_stats.py`, smoke-suite growth artifacts (`src/slm_training/resources/test_seeds.jsonl` append-only, new published eval under `src/slm_training/resources/data/eval/`), `climb_policy.py::screening_smoke_n_for_policy` |
| SWARM-DATA | `scripts/build_train_data.py` invocation artifacts, new train snapshot under `src/slm_training/resources/data/train/`, `policy.v2.json` consumers read `defaults.train_version` — the *value* change is delivered as a one-line patch to SWARM-METRIC |
| SWARM-MULTIARM | matrix execution loop in `run_autotrain_continuous.py` (arm scheduling region), `experiment_campaign.py` multi-arm manifest support if missing |
| SWARM-DECODE | `src/slm_training/dsl/grammar/fastpath/speculative_rank.py`, `scripts/build_speculative_ngram_table.py`, `src/slm_training/resources/decode/speculative_ngram_v1.json` (regenerate, train-only), `src/slm_training/runtime/decode_schedule.py` counters |
| SWARM-OPS | `thrash_bank_compose` root cause (compose/bank region of `run_autotrain_continuous.py`), `_is_foreign_dirty_path` + a new `.serena/project.yml` self-heal, escalation dedup |
| INTEGRATOR | `src/slm_training/resources/versions.json`, `docs/design/` result docs, `docs/MODEL_CARD.md` + README summary if any checkpoint is designated reusable, final merge + global gates |

If two swarms discover they need the same symbol, the one listed here owns it
and the other files an interface request in its final report; the Integrator
resolves.

### C4 — Definition of done (every swarm)

1. Code + tests merged on the working branch; all owned tests pass.
2. At least one runnable check per non-trivial behavior (the smallest test
   that fails if the logic reverts).
3. No weakening of any gate, invariant, or honest-n bookkeeping.
4. A short report: what changed, evidence it works (command output), open
   interface requests, and which watched files the Integrator must version-bump.

---

## 5. Swarm task cards

### SWARM-METRIC — screening primary with signal (held-out NLL)

**Problem (atomic):** the screening verdict compares `smoke.structural_similarity`
(6-sample, σ_mean ≈ 0.12) against `minimum_effect = 0.01`. Replace the
screening-tier primary with held-out NLL under a canonical fixed loss, which
is continuous and discriminative even at tiny step counts.

**Context you need:** verdict classification lives in
`src/slm_training/autoresearch/climb_policy.py`
(`primary_for_role`, `classify_positive_metrics`, `_metric_from_map`,
`promotion_primary_effect_met`). Roles: `screening`, `confirm`, `promotion`
(`cycle_role_for_index`). The continuous driver
(`scripts/run_autotrain_continuous.py`, ~15.5k lines) runs per-arm train →
eval → scoreboard; the quality eval writes `scoreboard.json` with
`suites.smoke.*`. A loss-suite evaluator already exists:
`scripts/evaluate_loss_suites.py` — reuse it (or its library entrypoint)
rather than writing a new NLL path. The canonical loss must be computed with
**baseline loss weights regardless of arm knobs** (arms may set
`binder_*_loss_weight` etc.; those must not leak into the NLL definition —
snapshot the baseline loss config once per cycle and pass it to both arms).

**Required behavior:**
- Both arms of every screening cycle get `suites.smoke.eval_nll` in their
  scoreboards, computed on the same published smoke suite as the quality eval
  (same `eval_data_manifest_sha`).
- `primary_for_role(policy, "screening")` returns the C2 NLL primary;
  direction `decrease` is honored end-to-end (a *lower* candidate NLL is a
  win; audit every `improvement =` comparison you touch for sign errors).
- Quality metrics keep being recorded (C2 secondary) and keep their role in
  confirm/promotion tiers.
- NLL cycles remain `claim_class: diagnostic`; nothing here touches ship
  gates (`gates.json` untouched).
- Apply the **entire** C2 policy delta (you own the file), including the
  fields other swarms consume.

**Tests (write; smallest that fail on revert):** direction-decrease win/loss
classification; loss-config immunity (arm with nonzero
`binder_arity_loss_weight` produces the same NLL definition hash as control);
policy v14 loads and `primary_for_role("screening").metric == "smoke.eval_nll"`;
existing screening tests updated via the mirrored JSON cases +
`python -m scripts.refresh_test_cases`.

**Validation:** run the driver's single-cycle/dry-run entrypoint (discover the
flag via `python scripts/run_autotrain_continuous.py --help`; a one-cycle mode
exists for the supervisor) against a scratch loop id and confirm the handoff
carries the NLL primary with non-identical, finite values for both arms.

### SWARM-BUDGET — spend the legal compute budget

**Problem (atomic):** arms train 22 steps / 3.36 s inside a 20 s train floor
inside a 70 s arm wall inside a 180 s stage wall (`MAX_RUN_MINUTES = 3`). The
fitter (`_screening_thrash_steps` in `scripts/run_autotrain_continuous.py`,
plus the decode-fit that wrote `thrash_timing.json`) leaves ≥ 80% of the train
floor and all eval slack unused.

**Required behavior:**
- Steps are fitted, not hardcoded: measure steps/sec from the most recent
  train telemetry for the same config family
  (`train_telemetry.json` / `train_summary.json`: `elapsed_wall_seconds`,
  `steps`) and set `steps = clamp(floor_seconds × measured_steps_per_sec ×
  0.9, 1, screening_thrash_steps_max)` with a cold-start default when no
  telemetry exists. c527 telemetry implies ~130 steps fit today's floor.
- The train floor itself grows into measured eval slack: after decode fit, if
  `eval_budget_seconds` projected < allocated, reassign the residual to the
  train floor. Never exceed the arm wall; never touch `MAX_RUN_MINUTES`;
  never silently extend a wall (the timeout law: recalibrate the recipe, no
  ad-hoc `wall++`).
- Device routing: at train launch, detect CUDA (`torch.cuda.is_available()`)
  and route screening trains to it when present; record the device in
  `train_summary.json` (already emitted) so results stay comparable.
  Size-matching is unaffected (same model, faster wall).
- The fit is recorded in `thrash_timing.json` (extend the existing
  `thrash_timing/v1` payload with the fitted steps and their evidence).

**Tests:** fitter math (telemetry → steps, clamping, cold-start default);
floor-growth never exceeds arm wall; device routing falls back to CPU
cleanly. **Validation:** one scratch-loop cycle whose `train_summary.json`
shows `stopped_on` ≠ `"steps"` at 22 — i.e., ≥ ~100 steps on CPU or a CUDA
device — while total arm wall stays ≤ 70 s.

### SWARM-CHAMPION — accumulating warm-start climb

**Problem (atomic):** every arm cold-starts (`initialized_weight_count: 0`);
no state accumulates, so there is no hill. `thrash_regime.py` already defines
a `climb` regime keyed on a champion baseline, but
`has_climb_baseline: false, reason: no_climb_baseline_causal_ofat` in every
recent handoff — it has never been seeded.

**Required behavior:**
- Maintain a persistent champion checkpoint per loop under the loop state dir
  (`outputs/autoresearch/loops/<loop_id>/champion/last.pt` + a JSON sidecar
  with lineage: source campaign, cumulative steps, train-data manifest sha,
  cumulative epochs over the train set).
- Screening arms warm-start from it per policy `measurement.warm_start`
  (C2): control = champion + K steps at baseline knobs; candidate = champion
  fork + K steps with the knob applied; K identical, data slice identical,
  seeds locked in the manifest. Both arms' extra compute is identical by
  construction — assert it.
- The champion advances **only** on a confirmed win (existing confirm tier),
  by adopting the winning arm's checkpoint; rejected cycles leave it
  untouched. Seed the initial champion from the best existing confirmed
  artifact if one exists, else from a single baseline train at the fitted
  step budget (SWARM-BUDGET's fitter; if that swarm's code is not merged yet
  in your worktree, use a local equivalent — do not block).
- Overfitting guard: when the champion's cumulative epochs over its train
  snapshot exceed `max_cumulative_epochs` (C2: 50), the loop demands a train
  snapshot upgrade (typed park reason `champion_epochs_exhausted`, healed by
  adopting a newer `defaults.train_version`) rather than continuing to grind
  the same 101 records.
- Leakage/disjointness checks between the champion's train manifest and the
  eval suite remain mandatory and fail closed.
- Size-matching: warm-start changes no parameter counts; assert params(control)
  == params(candidate) at launch (invariant VI).

**Tests:** champion sidecar round-trip; warm-start config builder passes the
checkpoint to both arms; champion advances on confirm-win and only then;
epoch-exhaustion park triggers at the cap; params equality assertion.
**Validation:** two consecutive scratch-loop cycles where cycle 2's
`train_summary.json` shows `initialized_weight_count > 0` and the sidecar's
cumulative steps grew.

### SWARM-STATS — paired statistics, power floor, and suite growth

**Problem (atomic):** n = 6 with an exact sign test degenerates under ties
(115/333 all-tie cycles); the Lean bound certifies decidability, not power
(`power_floor_n: null`); `minimum_effect: 0.01` is ~12× below the noise
floor; `promotion_suite_n: 6` has the same disease.

**Required behavior:**
- New module `src/slm_training/autoresearch/paired_stats.py`: Wilcoxon
  signed-rank on per-record paired deltas (per policy C2 `paired_test`),
  exact-sign-test fallback for tiny n, explicit tie handling
  (ties are dropped; verdict requires ≥ `min_nontied_pairs` non-tied pairs,
  else the cycle is `mechanism_no_effect`, not a loss). Pure functions, no
  scipy dependency unless it is already installed — check `pyproject`/lockfile
  first and prefer an exact/permutation implementation otherwise.
- Estimate variance components from real data: pull per-record scores from
  existing run artifacts (`eval_smoke.json` / task scoreboards under
  `outputs/autoresearch/continuous-loop-*/runs/*/`) and compute per-record σ,
  between-cycle σ, and the MDE at candidate n ∈ {6, 12, 24, 48}. Write the
  analysis to `docs/design/screening-power-analysis.md` + JSON sidecar (this
  is a measured result: include recipe metadata and a version stamp).
- Extend `screening_sample_size.py` to emit `power_floor_n` (smallest n whose
  MDE ≤ the policy `minimum_effect` for the active primary) alongside the
  Lean decidability floor; `chosen_n = max(decidability_floor_n,
  power_floor_n)` clamped by suite and budget ceilings, same fail-closed
  `must_generate` behavior when the suite is short.
- Grow the smoke suite to the certified n (target ≥ 24 records): append new
  legal, symbol-only, grammar-valid smoke fixtures to
  `src/slm_training/resources/test_seeds.jsonl` (append-only, unique ids,
  `split`/`meta.suite` = `smoke`, disjoint from the active train manifest),
  build with `python -m scripts.build_test_data`, publish an immutable
  snapshot under `src/slm_training/resources/data/eval/<new_id>/` including a
  `screening_sample_size.json` sidecar, and point the continuous default
  eval resolution at it (the loop resolves `published_smoke`). Never mutate
  frozen `e938_*` snapshots. Read the build's quality report and act on it
  (data-quality law).
- Update `measurement.promotion_suite_n` (deliver the value to SWARM-METRIC's
  policy edit as an interface patch — one number) so confirm/promotion are
  powered too.

**Tests:** Wilcoxon/sign correctness on known vectors; tie-degeneracy returns
`mechanism_no_effect`; power-floor math (MDE monotone decreasing in n);
resolver returns `max(floors)`; publish path refuses to touch frozen dirs.
**Validation:** resolver on the new published suite returns a feasible
verdict with both floors non-null; a scratch cycle screens at the new n
within the arm wall (coordinate decode-fit numbers with SWARM-BUDGET's
contract, not its code).

### SWARM-DATA — train-data growth

**Problem (atomic):** `wf_smoke_v2` has 101 records; any accumulating
champion (SWARM-CHAMPION) will saturate it within ~50 epochs, and data
composition is one of the few lever families with real effect sizes at small
step counts.

**Required behavior:**
- Build a strictly larger train snapshot with
  `python -m scripts.build_train_data` (strict profile: fuzzy+semantic dedup,
  tier floor, n-gram decontamination vs eval suites incl. the *new* smoke
  suite from SWARM-STATS — use `--dedup-against` / the documented flags; read
  `--help`). Target ≥ 1,000 records or the maximum the existing synthesizers
  yield under strict gates, whichever is smaller. Publish under
  `src/slm_training/resources/data/train/<new_id>/`.
- Close the synthesis loop (law): read `quality_report.json`,
  `rejected.jsonl`, `synthesis_feedback.json`; make at least the top named
  synthesizer fix if the feedback names one; file the emitted experiment
  candidates into the loop's evidence dir; document in
  `docs/design/` (markdown + JSON, version-stamped).
- Deliver the new `defaults.train_version` value to SWARM-METRIC as a
  one-line policy patch. Keep `wf_smoke_v2` intact for replay.
- Verify train/eval disjointness against both eval suites; fail closed.

**Tests/validation:** the build's own gates plus a one-command leakage check;
a scratch train run consumes the snapshot (`record_count` reflects it).

### SWARM-MULTIARM — k-arm screening per cycle

**Problem (atomic):** each cycle's hypothesis matrix proposes ~59 experiments
(`matrix-proposal.json`, `hypothesis_matrix_formed count:59`) and executes 2
(one control, one candidate). Information per cycle is 1 noisy bit.

**Required behavior:**
- Screening cycles execute up to `measurement.multi_arm.max_arms_per_cycle`
  (C2: 6) candidate arms against **one shared control**, arm order, seeds,
  and the static selection rule (`best_by_primary_then_smallest`: best
  primary; ties broken toward fewer trainable params, invariant VI §18)
  locked in the `ExperimentCampaignV1` manifest before any arm runs.
- Budget law: each arm individually obeys the arm wall; the cycle schedules
  arms sequentially (they already run sequentially today — verify how the
  stage wall in `stage_wall_minutes_for_role` scopes: per arm or per cycle —
  and size `max_arms` so nothing is ever killed mid-run; a killed run is not
  evidence). If the current scoping cannot fit 6 arms, run the max that fits
  and record the constraint in the handoff.
- Size-matched arms only, per `levers.require_size_matched_arms`; a
  capacity-deviating hypothesis is skipped with a typed reason, never run
  unmatched.
- The winner (if any, by paired test from SWARM-STATS's contract — consume
  the C2 `paired_test` policy shape, not the module, if unmerged) proceeds to
  the existing confirm tier unchanged. Losers write their knob signatures to
  the exhausted-knob ledger exactly as today
  (`climb_policy.py::save_loop_exhausted_ledger`).

**Tests:** manifest locks k arms + rule before execution (assert event order
in the campaign event chain, cf. `results.tsv` event types
`decision_arms_bound` → `experiment_campaign_locked` → `experiment_started`);
selection-rule determinism incl. the smallest-params tiebreak; arm-count
fitting math. **Validation:** one scratch cycle with ≥ 3 candidate arms whose
handoff shows per-arm metrics and one locked selection.

### SWARM-DECODE — deterministic structural lift

**Problem (atomic):** the scoreboard says outputs are legal but structurally
wrong: `parse_rate` 1.0, `ast_node_f1` 0.179, `ast_edge_f1` 0.0,
`structural_similarity` 0.131, failure breakdown `low_component_recall`. The
deterministic decode stack (invariants I1–I4) can move these **without any
training**, which makes it the largest immediately-available lever and one
measurable even at n = 6.

**Required behavior:**
- Diagnose first, from artifacts: decode the c527 arm outputs (task
  scoreboard details in the run dirs) and identify why edges never match —
  wrong component nesting, truncation at decode timeout, or ranking that
  never proposes container symbols. Write the diagnosis into your report.
- Improve the deterministic scorer
  (`src/slm_training/dsl/grammar/fastpath/speculative_rank.py`) and/or its
  n-gram table: regenerate
  `src/slm_training/resources/decode/speculative_ngram_v1.json` with
  `scripts/build_speculative_ngram_table.py` from **train-only** data (the
  new snapshot if SWARM-DATA's id is available in the worktree, else
  `wf_smoke_v2`) — never from eval records. Higher-order or structure-aware
  ranking is in scope; anything that weakens legality or skips
  verify-before-commit is forbidden (I3/I5/I6; weakening levers belong in
  `levers.CONSTRAINT_WEAKENING_LEVERS` and are CI-blocked).
- Preserve/extend `DecodeStats` counters (forwards avoided, scheduled
  prefills) via `src/slm_training/runtime/decode_schedule.py`; utilization
  claims are measured (I4). New decode behavior keeps the singleton
  `forwards_count == 0` bypass property (I2) — extend the existing bypass
  test pattern to your path.
- Measure before/after on the published smoke suite with the standard eval
  harness, both arms deterministic (this is an eval run: document under
  `docs/design/` with recipe + version stamp; the ngram-table artifact and
  its component version are watched — flag the Integrator).

**Tests:** ranker unit tests on fixed grammars (better-than-uniform ordering
on known continuations); table regeneration determinism from a fixed
manifest; leakage guard (building from an eval manifest raises).
**Validation:** side-by-side eval JSON showing `structural_similarity` /
`ast_edge_f1` delta with identical checkpoints — the delta is attributable to
decode only.

### SWARM-OPS — stop the bleeding

**Problem (atomic):** 3,377 `thrash_bank_compose` soft-heals (~1 per cycle —
a chronically failing step healed instead of fixed); the loop hard-parks on
`.serena/project.yml` whenever any Serena-based agent activates the project
(Serena rewrites the file, stripping comments), sleeping 3,600 s per tick —
~9 h lost on 2026-08-20 alone; 60 duplicate escalations for one fingerprint.

**Required behavior:**
- `thrash_bank_compose`: find the recovery writer (grep
  `autotrain_cycle_recovery/v1` / `soft_healed` in
  `scripts/run_autotrain_continuous.py`), reproduce the compose failure from
  the logged evidence, and fix the root cause so the heal becomes the
  exception (< 1 per 50 cycles), not the rule. Keep the heal as a fallback.
- `.serena/project.yml`: add a typed self-heal in the dirty-tree path
  (`_is_foreign_dirty_path` / the self-heal region around
  `SELF_HEAL_DIRTY_TREE_SKIP`): when the **only** foreign dirty path is
  `.serena/project.yml` and `git diff` shows a pure comment/whitespace-strip
  rewrite (no semantic YAML change — compare parsed YAML equality), restore
  it (`git checkout -- .serena/project.yml`) and continue; any semantic
  change still parks (that guard exists on purpose — do not blanket-ignore
  the file).
- Escalation dedup: identical `fingerprint` escalations update
  `last_seen_at`/count on one record instead of appending (the record schema
  `escalation_record/v1` already carries both timestamps).

**Tests:** YAML-equality restorer (comment-strip → restored+continue;
semantic edit → parked); escalation dedup; a regression test for the compose
root cause you find. **Validation:** scratch loop tick with a comment-stripped
`.serena/project.yml` proceeds unparked; compose completes without a
soft-heal on a normal cycle.

### INTEGRATOR — merge, version, gate

**Owns:** final merge of all swarm branches, `versions.json`, global gates,
results documentation, and the live-loop handoff.

**Required behavior:**
- Merge order is dependency-free by construction (C3); resolve any interface
  requests from swarm reports (e.g. SWARM-DATA's `defaults.train_version`
  one-liner, SWARM-STATS's `promotion_suite_n`).
- Bump every watched component in
  `src/slm_training/resources/versions.json` for the files changed
  (candidates: the autotrain campaign/harness component, climb policy,
  screening sample size, decode ngram table, data builders; discover exact
  component ids by reading the registry's `paths` maps) — or add `no-bump:`
  notes for behavior-neutral edits. `python -m scripts.verify_version_stamps
  --check` must pass.
- Global gates, all must pass:
  `python -m scripts.repo_policy`; `.githooks/check-changed`;
  `python -m scripts.verify_decode_invariants`;
  `python -m scripts.verify_version_stamps --check`;
  `python -m scripts.refresh_test_cases --check --changed`;
  the repository's standard pytest selection for touched areas.
- End-to-end validation: run **three consecutive cycles** on a scratch loop
  id (never the live `continuous-openui-local` state dir) and assert, from
  the handoffs: (1) screening primary is `smoke.eval_nll` with finite,
  non-identical arm values; (2) `train_summary.json` shows fitted steps ≫ 22
  and warm start on cycle ≥ 2; (3) the paired test emits a p-value or a typed
  tie verdict; (4) no `thrash_bank_compose` soft-heal; (5) every run's wall
  respects the 3-minute law. Document the validation as a measured result
  under `docs/design/` (markdown + JSON, recipe metadata, version stamp) —
  the iron law applies to these runs.
- If any validation run designates a reusable champion checkpoint, update
  `docs/MODEL_CARD.md` and the README model-card summary (fixture-scale,
  honestly labeled).
- Commit everything (including this directive if still uncommitted); the live
  loop parks on foreign dirty paths, so a clean tree is part of done. Do not
  kill or restart the running supervisor; it picks up the branch on its next
  tick.

---

## 6. What success looks like (falsifiable)

Within the first ~50 post-merge cycles of the live loop, at least one of:

- a screening win on `smoke.eval_nll` that **survives the confirm tier** (the
  first confirmed win in 527+ cycles), advancing the champion checkpoint; or
- a decode-side lever confirmed on the powered suite with a
  `structural_similarity` gain ≥ the recorded MDE; or
- a documented, evidence-backed verdict that the remaining hypothesis bank is
  truly exhausted at this scale — with the successor approach filed in the
  same doc (goals are non-negotiable; approaches are disposable, I14).

"More cycles, still zero promotions, but the logs look busier" is failure.
