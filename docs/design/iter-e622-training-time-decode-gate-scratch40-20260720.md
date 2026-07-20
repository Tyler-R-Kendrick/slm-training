# E622 — the E617 decode gate can now go live during training, but nothing behind it fires yet

Date: 2026-07-20
Status: completed, mixed finding, no code change, checkpoint rejected, not promotable

E616-E621 closed out one thread: decode-time bias/eval-correctness bugs found
and fixed by replaying `scripts/evaluate_model.py` control/treatment pairs
against an already-trained checkpoint, entirely outside any live training
loop. This iteration pivots to what the thread never actually did: run a
real `slm sft train` invocation and check whether the E617-fixed
`self._slot_contracts` gate (`src/slm_training/models/twotower.py`) is ever
populated as a side effect of the training process's *own* periodic eval,
rather than only through the separate `evaluate_model.py` CLI.

## Where the bias could plausibly go live during training

Plain SFT's forward/loss step is teacher-forced cross-entropy and never
decodes. The only path inside `slm sft train` that reaches
`generate_batch_requests`/`_generate_batch_once` is periodic ship-eval
(`--test-dir`/`--eval-every`, `train_loop.py:753 _maybe_eval` ->
`eval_runner.evaluate_suites`) and the loss-suite eval. GRPO-lite RL's
on-policy rollout would also reach it every step (the most direct case), but
`slm rl train` is fail-closed on an approved `RLReadinessReport` and none
exists in this repo snapshot -- producing one requires the full
`autoresearch validate-rl` gate (frozen five-suite eval, full `rico_held`,
honest ship gates, AgentV pass, nonzero reward variance), out of scope for a
single 3-minute-capped iteration. So the periodic-eval side effect was the
only training-time path actually reachable this session.

## Run 1: over the cap, discarded

First attempt trained the same corpus/seed/architecture as E616/E617
(`src/slm_training/resources/data/train/e530_visible_semantic_roles_r2_20260719`)
for 80 steps with `--eval-every 40` (two periodic-eval passes over the
4-record `ood` suite). This ran past the repo's 3-minute hard run cap
(`AGENTS.md`): the agent-command 170s timeout silently moved it to a
background job, and it kept running to a real wall time of 3m18.134s before
being killed. Per the iron law, a killed run is never evidence -- its
`outputs/runs/` directory was deleted before retrying, and no number from it
appears anywhere in this doc.

## Run 2: scaled down, completed honestly

Scaled to `--steps 40 --eval-every 40` (single periodic-eval pass, firing
once at the final step) under `timeout 165`:

```bash
python -m scripts.train_model \
  --train-dir src/slm_training/resources/data/train/e530_visible_semantic_roles_r2_20260719 \
  --model twotower --device cpu --context-backend scratch --output-tokenizer choice \
  --steps 40 --batch-size 1 --seed 0 \
  --test-dir src/slm_training/resources/data/eval/remediated --eval-every 40 --eval-suites ood \
  --honest-slot-contract --slot-contract-constrained-decode \
  --no-sync-checkpoints --run-id e622-training-time-decode-gate-scratch40-20260720
```

Completed cleanly: real 51.285s wall (`train_summary.json`'s own
`elapsed_wall_seconds`: 48.681), comfortably inside the cap. Internal
telemetry attributes 88.78% of wall time to the single `eval_suites` span
(39.326s) versus 1-2% each for `forward`/`backward`/`optim_step` across the
40 training steps -- the periodic eval, not training, is what pushed the
first (80-step, 2-pass) attempt over the cap. Budget roughly 40s per fired
eval pass on this suite/backend when sizing future `--eval-every` schedules
against the 3-minute cap, not per training step.

## Finding: the gate opens, nothing behind it is loaded

`outputs/runs/e622-training-time-decode-gate-scratch40-20260720/scoreboard.json`'s
`evaluation_policy` for the in-training periodic eval shows
`slot_contract_constrained_decode: true` and `honest_slot_contract: true` --
`self._slot_contracts` really is populated during this training run's own
periodic eval call. **This is the first time in the E610-E621 lineage that
has happened inside an actual `slm sft train` process, rather than only in a
standalone `evaluate_model.py` replay.**

But every weighted decode lever the E617/E620 fixes gate on --
`schema_role_slot_decode_weight`, `semantic_role_decode_weight`,
`slot_coverage_close_decode_weight`, `schema_value_decode_weight`,
`schema_opaque_close_decode_weight`, and all nine
`semantic_plan_*_decode_weight` fields -- is `0.0` in that same
`evaluation_policy` block. `grep` of `scripts/train_model.py` confirms zero
occurrences of `semantic_plan` anywhere in its CLI: unlike
`scripts/evaluate_model.py`, it never exposes flags to set any of these
levers nonzero, and `ModelBuildConfig` (`harnesses/model_build/config.py`)
defaults them all to `None`/inert. So the E617 gate is open for the first
time during live training, but nothing behind it fires -- this training
run's periodic-eval decode is mechanically no different from a
pre-E617-fix training run would have been, for a *different*, still-live
reason: a CLI surface gap between `train_model.py` and `evaluate_model.py`,
not the E617/E618/E620 bugs already fixed in `twotower.py` itself.

Real numbers for this 40-step scratch checkpoint's single periodic eval
(`ood` suite, n=4, not a ship claim -- this checkpoint is far too
undertrained to be promotable): `parse_rate` 1.0, `meaningful_program_rate`
0.0, `structural_similarity` 0.190625, `placeholder_fidelity` 0.0,
`reward_score` 0.0, `ship_score` 0.03465909090909091.

## Decision

No code change this iteration. The finding is about training-time CLI flag
coverage, not a decode-logic bug -- reporting it rather than silently
patching it under time pressure so it can be triaged deliberately, and to
avoid re-opening the just-closed E617/E618/E620/E621 audit thread with a
new, differently-shaped gap. Concrete next step if training-time exercise of
these biases is wanted: add the missing
`--semantic-plan-*`/`--schema-*`/`--slot-coverage-close-decode-weight` flags
to `scripts/train_model.py`, or an `--eval-policy-preset` flag applying
`eval_policy.STRICT_COMPILER_TREE_POLICY` to the live training config (today
it is only ever applied inside standalone eval call sites), then re-run this
same recipe and diff `evaluation_policy` before/after.

No checkpoint promoted or synced. No metric in this lineage changes; these
are a fresh standalone checkpoint's own numbers, not a comparison claim.

Evidence: [JSON](iter-e622-training-time-decode-gate-scratch40-20260720.json).
