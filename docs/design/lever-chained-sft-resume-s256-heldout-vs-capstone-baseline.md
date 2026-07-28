# Chained-resume SFT to 256 steps vs the capstone `held_out` baseline — NOT SHIP

**Honesty:** `fixture_or_scratch`, `held_out` suite, **n=5, 1 rep**. **Not ship.**

## Task

This iteration's own assignment, taken verbatim from
[PR #1197's "Named next lever" item 1](lever-mix-loadable-v2-heldout-sft-vs-capstone-baseline.md#named-next-lever-for-the-next-iteration):
run a genuinely longer SFT (hundreds of steps, not ≤32) via a multi-invocation
checkpoint-resume campaign, on the same `lever_mix_loadable_v2` corpus and the
same architecture/hyperparameters PR #1197 used, then re-eval with the exact
capstone `held_out`/seed=47/`--decode-timeout-seconds 30`/`fixed_asap` recipe
as a direct A/B against PR #1197's `0/5` meaningful baseline.

## Mechanism: `--resume-from` chained across 7 bounded invocations

`python -m scripts.train_model` supports `--resume-from <last_full_state.pt>`
(`src/slm_training/harnesses/model_build/train_loop.py`): it restores model
weights, optimizer state (with a fingerprint check), RNG state, and the
absolute `step` counter, and asserts the resumed checkpoint's
`data_manifest_sha` / `mixture_hash` match the current `--train-dir` (fails
closed on drift). `--steps N` is an **absolute target step count**
(`while step < config.steps`), not an increment, so each invocation trains
only `N_target - N_current` steps. A full-state checkpoint
(`checkpoints/last_full_state.pt`) is written unconditionally at the end of
every invocation (`_save_full_state_now()`), alongside the eval-ready
`checkpoints/last.pt`.

This let a 256-step campaign run as 7 bounded ≤10s invocations instead of one
that would exceed `MAX_RUN_MINUTES=3` per call:

```bash
# invocation 0 (already existed from PR #1197 — 32 steps, fresh)
python -m scripts.train_model --train-dir outputs/data/train/lever_mix_loadable_v2 \
  --model twotower --context-backend scratch --steps 32 --batch-size 2 --lr 0.001 \
  --structural-bias 1.5 --seed 47 --device cpu --asap-decode \
  --run-id exp_lever_mix_loadable_v2_s32_lr1e3_bs2_sb15_seed47 --no-sync-checkpoints

# invocations 1-7 (this session) — same corpus/hparams, --resume-from chained
for STEPS in 64 96 128 160 192 224 256; do
  python -m scripts.train_model --train-dir outputs/data/train/lever_mix_loadable_v2 \
    --model twotower --context-backend scratch --steps "$STEPS" --batch-size 2 \
    --lr 0.001 --structural-bias 1.5 --seed 47 --device cpu --asap-decode \
    --resume-from outputs/runs/<prev-run-id>/checkpoints/last_full_state.pt \
    --run-id chain_lever_mix_v2_s${STEPS}_lr1e3_bs2_sb15_seed47 --no-sync-checkpoints
done
```

Corpus reused unchanged: `outputs/data/train/lever_mix_loadable_v2` (231
records, `data_manifest_sha=1e81b1ec…`) — already built and gitignored from
PR #1197's session; loaded cleanly, no rebuild needed this iteration.
Architecture/hyperparameters unchanged from PR #1197: `twotower`/`scratch`,
`lr=0.001`, `batch_size=2`, `structural_bias=1.5`, `seed=47` — **pure step-count
change, no capacity/size lever** (goal invariant VI: no `d_model`/layer/head
flags touched).

## Steps reached: 256 total (8x the 32-step baseline), all invocations well under cap

| target steps | run_id | resumed_from | elapsed_wall_seconds | last_loss | stopped_on |
| ---: | --- | --- | ---: | ---: | --- |
| 32 (PR #1197) | `exp_..._s32_...` | — (fresh) | 8.88 | 11.379 | steps |
| 64 | `chain_..._s64_...` | s32 | 5.57 | 7.276 | steps |
| 96 | `chain_..._s96_...` | s64 | 4.80 | 5.645 | steps |
| 128 | `chain_..._s128_...` | s96 | 4.85 | 4.628 | steps |
| 160 | `chain_..._s160_...` | s128 | 4.83 | 6.114 | steps |
| 192 | `chain_..._s192_...` | s160 | 4.88 | 5.179 | steps |
| 224 | `chain_..._s224_...` | s192 | 5.33 | 5.752 | steps |
| 256 | `chain_..._s256_...` | s224 | 5.49 | 6.819 | steps |

Every invocation `stopped_on="steps"` (none hit `wall_time_budget`), each
resumed cleanly (`data_manifest_sha` matched at every hop, optimizer
fingerprint matched). Loss fell sharply through step 128 (11.38 → 4.63, a new
low vs any prior checkpoint in this stack) then oscillated upward to 6.82 by
step 256 — consistent with a small 231-record corpus and `bs=2` producing
noisy multi-epoch traversal at this step count, not a bug. The final
checkpoint (`chain_lever_mix_v2_s256_lr1e3_bs2_sb15_seed47`) is a genuinely
more-trained model than PR #1197's 32-step one by any loss measure along the
way, and strictly more trained than it by cumulative step count.

## Re-eval: exact capstone/PR #1197 recipe, 1 rep, n=5

```bash
python -m scripts.evaluate_model \
  --test-dir src/slm_training/resources/data/eval/e938_role_safe_all_targets_v2 \
  --suite held_out --model twotower --device cpu \
  --checkpoint outputs/runs/chain_lever_mix_v2_s256_lr1e3_bs2_sb15_seed47/checkpoints/last.pt \
  --grammar-constrained --decode-timeout-seconds 30 --seed 47 \
  --constraint-debt-routing-mode fixed_asap --run-class scratch_matrix \
  --run-id chain_mix_v2_s256_heldout_rep1_seed47
```

Same shrinkage precedent as PR #1197 (n=5, 1 rep instead of the capstone's 2
reps) to fit corpus-reuse + chained SFT + eval inside this iteration's bounded
budget.

## Measured: `0/5` meaningful (unchanged) — and decode outcomes got *worse*, not better

| metric | capstone baseline (`exp12`, per rep, n=5) | PR #1197 (`mix_v2` s32, n=5) | this run (`mix_v2` s256, n=5) |
| --- | ---: | ---: | ---: |
| `meaningful_program_rate` | 0.0 | 0.0 | **0.0** |
| `parse_rate` | 0.0 | 0.2 | **0.0** |
| `structural_similarity` | 0.0 | 0.0 | 0.0 |
| `component_type_recall` | 0.0 | 0.0 | 0.0 |
| `placeholder_fidelity` | — (n/a) | 0.2 | **0.0** |
| `reward_score` | — | 0.1874 | **0.0** |
| `decode_outcome_counts` | `{runtime_timeout: 5}` | `{runtime_timeout: 4, model_invalid: 1}` | `{runtime_timeout: 5}` |
| `failure_breakdown` | `{parse_error: 5}` | `{parse_error: 4, low_component_recall: 1}` | `{parse_error: 5}` |
| `latency_ms_p50` | — | — | 30000.6 |
| `latency_ms_p95` | — | — | 30200.5 |

`checkpoint_sha256=9341ca71de0fd4a39a4936fabe5feb91fef43cf5d2167b501ec5fb6d487deb01`.

The primary ship-relevant number, `meaningful_program_rate`, is **unchanged at
0.0** for the third consecutive lever tried against this baseline. Unlike
PR #1197's 32-step checkpoint (which let one record slip through the 30s wall
as `model_invalid`), the 256-step checkpoint's completions were **all 5**
`runtime_timeout` — identical to the raw capstone baseline, and strictly worse
on every wall-clock-sensitive metric than the 32-step intermediate result. A
lower training loss did not translate into faster decode; if anything a
better-fit model may be exploring more of the compiler-tree witness search
space per record before the 30s wall cuts it off (consistent with, not
contradicting, the standing "search cost dominates, not undertraining"
interpretation — see below).

## Ship-gate check (`honest-ship-eval` default `held_out` bars)

| criterion | bar | actual | pass? |
| --- | ---: | ---: | --- |
| `insufficient_n` | ≥ 20 | 5 | fail |
| `meaningful_program_rate` | ≥ 0.40 | 0.0 | fail |
| `structural_similarity` | ≥ 0.30 | 0.0 | fail |
| `component_type_recall` | ≥ 0.30 | 0.0 | fail |
| `placeholder_fidelity` | ≥ 0.15 | 0.0 | fail |
| `decode_timeout_count` | = 0 | 5 | fail |

Fails 6 of 6 gates (worse than PR #1197's 5/6 — the one gate PR #1197 passed,
`placeholder_fidelity`, now fails too). **Verdict: `fixture_or_scratch`, not
ship** — same as every doc in this thread.

## Interpretation

**A genuinely longer SFT (8x the step count, 32→256, via chained
`--resume-from`) does not move `meaningful_program_rate` off 0.0 on
`held_out`, and if anything regresses the secondary decode-completion metrics
that PR #1197's shorter run had nudged.** Combined with PR #1197's own
finding (2.16x corpus size, iso), **both data-scale levers named by the
PR #1189-#1196 capstone — corpus size and step count — are now eliminated at
this architecture's current scale.** Neither a bigger loadable-mix corpus nor
substantially more optimization steps on it changes whether a `held_out`
record's compiler-tree witness search finishes inside the fixed 30s wall. The
bottleneck evidence continues to point at the decode-time search cost itself,
not at how well- or under-trained the checkpoint is.

## Decision

**REJECT** — chained-resume SFT to 256 steps does not lift
`meaningful_program_rate` vs the capstone `held_out` baseline (0.0→0.0) or vs
PR #1197's 32-step intermediate (0.0→0.0), and regresses
`decode_timeout_count`/`placeholder_fidelity`/`parse_rate` back to the raw
capstone baseline's fully-timed-out profile.

## Named next lever (for the next iteration)

Three levers have now been tried and rejected against this exact `held_out`
baseline: decode-plumbing micro-optimizations (PRs #1189-#1196), corpus-size +
short-SFT (PR #1197), and step-count via chained-resume SFT (this doc). All
three leave `meaningful_program_rate` at 0.0. The strongest remaining,
evidenced-by-elimination options:

1. **Revisit the 30s decode-timeout wall itself** as the bottleneck, not the
   checkpoint or the corpus — either an algorithmic change to the
   compiler-tree witness search (not another cache-level micro-optimization;
   those are exhausted per PRs #1189-#1195) or a documented, honestly labeled
   longer timeout for this eval protocol, re-validated against
   `honest-ship-eval` bars rather than assumed to help.
2. **Curriculum/mixture reweighting** — order or reweight the mix so easier
   completions are seen first (named but not yet run in
   `lever-train-signal-under-hard-wall-measured-results.md`); still open, but
   now a weaker bet given two independent data-scale levers (size and step
   count) both showed no effect on completion-within-wall odds.
3. Given the 256-step run's data point (lower loss, *worse* wall-clock
   completion), a targeted investigation of whether structural-bias/decode
   settings interact with training progress to change per-step decode cost
   would need to precede any further step-count scaling attempt.

## Validation

```text
python -m scripts.repo_policy
python -m scripts.verify_version_stamps --check
python -m scripts.verify_decode_invariants
```

No harness/metric/gate/matrix file changed this session (pure chained
`train_model` CLI invocations + one `evaluate_model` CLI invocation using
existing canonical scripts) — 0 version-stamp component bumps required.

## Scope note

- Diagnostic lever measurement only. No `--ship-gates` scoreboard claim, no
  checkpoint promotion, no `MODEL_CARD.md` update — all 8 checkpoints in this
  chain are scratch/`--no-sync-checkpoints`.
- `outputs/data/train/lever_mix_loadable_v2/`,
  `outputs/runs/exp_lever_mix_loadable_v2_s32_lr1e3_bs2_sb15_seed47/`,
  `outputs/runs/chain_lever_mix_v2_s{64,96,128,160,192,224,256}_lr1e3_bs2_sb15_seed47/`,
  and `outputs/runs/chain_mix_v2_s256_heldout_rep1_seed47/` are gitignored,
  not committed — this doc is the durable record.

Captured: 2026-07-28T13:45:06Z
