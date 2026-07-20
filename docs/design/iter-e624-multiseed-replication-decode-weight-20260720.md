# E624 — multi-seed replication of E623's decode-weight finding

Date: 2026-07-20
Status: completed, mixed result (2/3 seeds positive, 1/3 seed flat/negative);
not a ship claim

E623 closed a real CLI gap in `scripts/train_model.py` (it never exposed
`semantic_plan_*`/`schema_*` decode-weight flags that `evaluate_model.py`
already had) and reran a 40-step scratch paired control(weight=0)/treatment
(E617's weights) recipe at `seed=0`, finding a large positive delta
(`reward_score` 0->0.67, `ship_score` 0.035->0.657) with the explicit caveat
`n=4, single seed, tiny 40-step checkpoint, not a ship claim`. This iteration
reruns the exact same recipe at `seed=1` and `seed=2` to test whether the
positive direction holds.

## Environment note (unrelated to the experiment)

The first attempt at `seed=1` failed before any training happened:
`RuntimeError: OpenUI bridge returned empty output (exit=9): /opt/node22/bin/node:
--import tsx is not allowed in NODE_OPTIONS`. The session's global
`NODE_OPTIONS="--import tsx" --max-old-space-size=8192` is rejected by the
`node` binary invoked internally by `src/slm_training/dsl/lang_core.py`'s
OpenUI bridge subprocess. Overriding `NODE_OPTIONS="--max-old-space-size=8192"`
(dropping `--import tsx`) for the `train_model` invocation only fixed it; no
repo code was touched. This is an environment/session quirk, not a regression
from E623's code change — flagging here in case it recurs for a future
iteration.

## Method

Same recipe as E623 exactly, varying only `--seed`:

```bash
NODE_OPTIONS="--max-old-space-size=8192" python -m scripts.train_model \
  --train-dir src/slm_training/resources/data/train/e530_visible_semantic_roles_r2_20260719 \
  --model twotower --device cpu --context-backend scratch --output-tokenizer choice \
  --steps 40 --batch-size 1 --seed {1,2} \
  --test-dir src/slm_training/resources/data/eval/remediated --eval-every 40 --eval-suites ood \
  --honest-slot-contract --slot-contract-constrained-decode \
  --no-sync-checkpoints --run-id e624-control-scratch40-seed{1,2}-20260720
```

and the same treatment flags as E623 (`schema_role_slot_decode_weight=8.0`,
`semantic_role_decode_weight=8.0` with `--semantic-role-contract-in-context`,
`slot_coverage_close_decode_weight=2.0`, `schema_value_decode_weight=4.0`,
`schema_opaque_close_decode_weight=4.0`, `semantic_plan_decode_weight=4.0`,
`semantic_plan_margin_decode_weight=2.0`,
`semantic_plan_binding_decode_weight=1.0`,
`semantic_plan_root_decode_weight=8.0`,
`semantic_plan_root_margin_decode_weight=2.0`,
`semantic_plan_repeated_array_close_margin_decode_weight=2.0`,
`semantic_plan_repeated_slot_margin_decode_weight=2.0`,
`semantic_plan_typed_array_nonempty_margin_decode_weight=2.0`,
`semantic_plan_typed_array_item_margin_decode_weight=2.0`), run at
`--seed {1,2}` with `--run-id e624-treatment-scratch40-seed{1,2}-20260720`.
Each `eval_ood.json`'s `evaluation_policy` block was checked to confirm every
weight landed exactly as passed (it did, for both seeds). All four runs
completed well inside the 3-minute cap (control seed1 99.6s, treatment seed1
15.1s, control seed2 91.0s, treatment seed2 18.8s — each run's own
`elapsed_wall_seconds` from `train_summary.json`).

## Result: the positive direction holds at 2/3 seeds, but flips flat-to-negative at 1/3

| seed | arm | parse_rate | meaningful_program_rate | placeholder_fidelity | structural_similarity | reward_score | ship_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 (E623) | control | 1.0 | 0.0 | 0.0 | 0.190625 | 0.0 | 0.034659 |
| 0 (E623) | treatment | 1.0 | 0.75 | 0.5667 | 0.6439 | 0.67475 | 0.657202 |
| 0 | **delta** | 0.0 | **+0.75** | **+0.5667** | +0.4533 | **+0.67475** | **+0.622542** |
| 1 | control | 1.0 | 0.75 | 0.791667 | 0.191625 | 0.8475 | 0.672492 |
| 1 | treatment | 1.0 | 0.75 | 0.516667 | 0.59445 | 0.822 | 0.643415 |
| 1 | **delta** | 0.0 | 0.0 | **-0.275** | +0.402825 | **-0.0255** | **-0.029077** |
| 2 | control | 1.0 | 0.0 | 0.0 | 0.236875 | 0.0 | 0.043068 |
| 2 | treatment | 1.0 | 0.5 | 0.591667 | 0.55105 | 0.825 | 0.572161 |
| 2 | **delta** | 0.0 | **+0.5** | **+0.591667** | +0.314175 | **+0.825** | **+0.529092** |

Training loss is identical between control and treatment at each seed
(seed1: `last_loss` 9.207306861877441 both arms; seed2: 9.675814628601074
both arms), confirming again these levers are decode-only and do not touch
the teacher-forced training loop — same finding as E623.

## Honest reading: this is a floor effect, not an unconditional improvement

`seed=1`'s **control** arm (no decode-time biasing at all) already produces a
fairly strong unbiased decode: `meaningful_program_rate=0.75`,
`reward_score=0.8475`, `ship_score=0.6725` — numbers close to or better than
E623's *treatment* arm at seed=0. At that seed, adding the biases does
**not** help further; `reward_score` and `ship_score` both move slightly
negative (`-0.0255`, `-0.0291`), and `placeholder_fidelity` drops sharply
(`-0.275`) even though `structural_similarity` still rises. This is a real,
small, mixed-direction effect at seed=1 — not noise-free-positive, not
purely negative either.

`seed=2`'s control arm reproduces the seed=0 pattern almost exactly
(`meaningful_program_rate=0.0`, `reward_score=0.0`, near-floor `ship_score`
0.043), and treatment again produces a large positive delta
(`reward_score` +0.825, `ship_score` +0.529) — smaller than seed=0's
`meaningful_program_rate` delta (+0.5 vs +0.75) but the same direction and a
comparable magnitude on `reward_score`/`ship_score`.

This is consistent with, and strengthens, E623's own "floor effect"
hypothesis rather than contradicting it: **the effect's sign and size appear
contingent on whether the untreated (control) checkpoint's unbiased decode
is already near-degenerate at that seed.** When it is (seed=0, seed=2), the
compiler-legal/prompt-plan-compatible decode-time biasing has a lot of room
to help and does. When it isn't (seed=1 — this particular untrained 40-step
network's random init happened to produce non-degenerate decode on its own),
the same biasing has little room to help and is roughly a wash, trading
`placeholder_fidelity` for `structural_similarity` with no net
`reward_score`/`ship_score` gain.

**This directly answers E623's "single-seed" caveat honestly: the positive
direction does *not* unconditionally hold.** 2 of 3 seeds (0, 2) show a
large positive `reward_score`/`ship_score` delta; 1 of 3 (seed 1) shows a
small negative delta on those same two metrics. Averaging across seeds would
obscure this — the correct honest summary is "positive when the control
decode is near-degenerate, a wash otherwise," not "positive."

## Honest caveats

- Still `n=4` per suite per run (`ood`), now `n=3` seeds — a genuinely small
  sample for both eval size and seed count; this is diagnostic, not powered
  evidence (H19's protocol remains the right tool for that).
- All six checkpoints (E623's pair + this iteration's four) remain far too
  undertrained to be promotable or synced (`--no-sync-checkpoints`).
- The seed-1 control arm's already-strong unbiased decode is itself
  interesting and unexplained — whether that reflects genuine luck-of-init
  variance at 40 steps, or some interaction between `seed` and the corpus's
  244-record ordering, is not established by this run.
- No code was changed this iteration (pure replication); `verify_version_stamps
  --check` reports 0 components touched, consistent with that.

## Verification

```bash
python -m scripts.verify_version_stamps --check
python -m scripts.repo_policy
```

Evidence: [JSON](iter-e624-multiseed-replication-decode-weight-20260720.json).
