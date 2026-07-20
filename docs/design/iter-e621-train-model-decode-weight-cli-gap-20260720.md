# E621 — closing the train_model.py decode-weight CLI gap

Date: 2026-07-20
Status: completed; wiring fix + real matched scratch evidence; not a ship claim

## Motivation

E622 (still-open PR #616, not yet merged to `main`) traced whether periodic
in-training eval (`train_loop.py:753 _maybe_eval` →
`eval_runner.evaluate_suites`) ever populates the E617-gated decode-bias
state (`self._slot_contracts`) as a side effect of a live `slm sft train`
process, rather than only through the standalone `scripts.evaluate_model`
replay path every prior E608–E620 iteration used. It found the gate opens
(`slot_contract_constrained_decode: true` appears in a live training run's
`scoreboard.json` for the first time in the lineage) but every weighted lever
behind it — `schema_role_slot_decode_weight` and the eleven
`semantic_plan_*_decode_weight` fields — stayed `0.0`, because
`scripts/train_model.py`'s CLI had zero `--semantic-plan-*` /
`--schema-*` / `--slot-coverage-close-decode-weight` flags at all, even
though `scripts/evaluate_model.py` has exposed all of them since E611–E617.
`ModelBuildConfig` already declares every field (`float | None = None`); only
the CLI surface was missing on the train side.

This iteration closes that specific, narrowly-scoped gap: no decode-bias
logic changed, only `scripts/train_model.py`'s argument parser and its
`ModelBuildConfig(...)` construction.

## Change

- `scripts/train_model.py`: added the 19 missing flags
  (`--semantic-role-decode-weight`, `--semantic-role-schema-candidates`,
  `--slot-coverage-close-decode-weight`, `--schema-value-decode-weight`,
  `--schema-opaque-decode-weight`, `--schema-enum-close-decode-weight`,
  `--schema-opaque-close-decode-weight`, `--schema-role-slot-decode-weight`,
  all eleven `--semantic-plan-*-decode-weight` flags, and
  `--visible-reference-decode-weight`), threaded into `ModelBuildConfig(...)`
  right after the existing `slot_component_*` block. Flag names, types, and
  help text mirror `scripts/evaluate_model.py` exactly; defaults follow
  `train_model.py`'s own existing convention for this class of flag
  (`0.0`, not `None`), consistent with `component_inventory_decode_weight`
  and its siblings already in that file.
- `tests/test_harnesses/model_build/test_train_model_cli_decode_weights.py`
  (new): one test drives `train_model.main(argv)` with every new flag set to
  a distinct nonzero value (`train()` monkeypatched to capture the
  constructed `ModelBuildConfig` instead of actually training) and asserts
  each field lands correctly; a second test confirms the untouched-CLI
  default is `0.0` on every new field (not `None`, matching sibling flags);
  a third is a `dataclasses.fields(ModelBuildConfig)` drift guard.
- `src/slm_training/resources/versions.json`: `model.twotower` v59 → v60
  (`scripts/train_model.py` is one of this component's `paths`).

No decode-bias code in `twotower.py`, `choice_tokenizer.py`, or
`eval_runner.py` changed. `src/slm_training/harnesses/model_build/factory.py`
already read every one of these `ModelBuildConfig` fields via
`getattr(config, name, 0.0) or 0.0` before this change — the factory-layer
wiring was already correct; only the CLI-to-config path on the train side was
missing.

## Real evidence: a live `slm sft train` run now sets these weights

Built a small fixture-scale scratch corpus this session (this container's
`outputs/` started empty, same as every prior E608–E622 session):
`outputs/data/train/e621-cli-gap-scratch` (fixture source, quality
synthesizer) and matching held-out suites
`outputs/data/eval/e621-cli-gap-scratch` (`smoke` n=3, no train-manifest
disjointness check — `--allow-without-train-manifest`, isolated-experiment
use only, not a leakage-checked claim).

Two 20-step scratch trains, `--context-backend scratch`,
`--output-tokenizer choice`, `--honest-slot-contract
--slot-contract-constrained-decode`, `--eval-every 20 --eval-suite smoke`,
`--no-sync-checkpoints`, both completing well inside the 3-minute cap
(91.7 s and 91.2 s wall):

- **Control** (`e621-cli-gap-control-choice`): every new flag left at its
  default.
- **Treatment** (`e621-cli-gap-treatment`): `--schema-role-slot-decode-weight
  8.0 --semantic-plan-decode-weight 4.0
  --semantic-plan-repeated-array-close-margin-decode-weight 2.0` (same
  weights E610–E621 have used against standalone-eval replays).

Both runs' own persisted `scoreboard.json` confirms the fix at the source of
truth — `suites.smoke.evaluation_policy`:

| Field | Control | Treatment |
| --- | ---: | ---: |
| `schema_role_slot_decode_weight` | 0.0 | 8.0 |
| `semantic_plan_decode_weight` | 0.0 | 4.0 |
| `semantic_plan_repeated_array_close_margin_decode_weight` | 0.0 | 2.0 |

Treatment's `decode_stats.constrained_selection_traces` also carries
`semantic_plan_decode_weight: 4.0` on every recorded decode step — the bias
is not just recorded, it is live in the generation path exercised by
in-training periodic eval, precisely the path E622 found dead.

## Headline metrics (smoke, n=3, 20-step scratch, informational only)

Both arms use the identical corpus, `--output-tokenizer choice`, and step
count — control differs from treatment only in the newly-wired weights, so
this is now a fair matched pair (unlike a same-session earlier attempt that
accidentally varied `--output-tokenizer` alongside the weights and was
discarded before being recorded anywhere).

| Metric | Control | Treatment | Delta |
| --- | ---: | ---: | ---: |
| syntax parse | 1.0000 | 1.0000 | 0 |
| meaningful v1 | 0.0000 | 0.3333 | +0.3333 |
| strict meaning v2 | 0.0000 | 0.0000 | 0 |
| v2 judgment coverage | 1.0000 | 1.0000 | 0 |
| placeholder fidelity | 0.0000 | 0.3333 | +0.3333 |
| placeholder validity | 0.0000 | 0.3333 | +0.3333 |
| structural similarity | 0.2650 | 0.1885 | -0.0765 |
| component recall | 0.0000 | 0.3333 | +0.3333 |
| reward | 0.0000 | 0.3023 | +0.3023 |
| AST node F1 | 0.4286 | 0.4158 | -0.0129 |
| AST edge F1 | 0.0000 | 0.1212 | +0.1212 |
| latency p50 | 27147.75 ms | 28006.78 ms | +859.03 ms |
| latency p95 | 29629.98 ms | 29669.00 ms | +39.02 ms |
| AgentV | 0/1 | 0/1 | 0 |

Checkpoints (local-only, not synced, not promoted):
`outputs/runs/e621-cli-gap-control-choice/checkpoints/last.pt`
(sha256 `026a9df8…27539223`),
`outputs/runs/e621-cli-gap-treatment/checkpoints/last.pt`
(sha256 `9ed5efbf…af7c0f6a500`).

**Read with real caution.** This is a 20-step scratch checkpoint on a 3-record
`smoke` suite (not `ood`/`held_out`, and not the E610–E621 lineage's usual
corpus) — an order of magnitude smaller and shorter than any prior E-series
scratch checkpoint. It is wiring-verification evidence that the new CLI
surface reaches the live decode path with a real, honest, matched
control/treatment delta, not a quality or promotion claim of any kind. No
checkpoint was synced or promoted; no `docs/MODEL_CARD.md`/README update
applies.

## What this does and doesn't fix

Fixed: `scripts/train_model.py` can now set every decode-weight lever
`scripts/evaluate_model.py` already exposed, so a live `slm sft train`
process's own periodic eval can exercise E610–E621's decode biases instead
of always being silently pinned to `0.0`/inert.

Not fixed, not attempted: whether *training itself* (the loss/backward pass,
as opposed to periodic eval's forward-only generation) should also see these
weights as loss terms is a separate, unopened question — every flag added
here is a decode-time bias, matching its `evaluate_model.py` counterpart
exactly; none of them touch `train_loop.py`'s loss computation.

## Test plan

- `pytest tests/test_harnesses/model_build/test_train_model_cli_decode_weights.py` — 3 passed (new)
- `pytest tests/test_harnesses/model_build/` — 213 passed, 1 skipped (pre-existing skip, unrelated)
- `python -m scripts.verify_version_stamps --check` — ok, 1 component touched (`model.twotower` v59→v60)
- Real training: two live `slm sft train` scratch runs (control, treatment),
  both under the 3-minute cap, both with periodic eval firing at the final
  step
- Real matched control/treatment comparison against the newly built scratch
  corpus and suite (not a prior lineage checkpoint — this container's
  `outputs/` was empty at session start)

Next step: replay this same control/treatment pair against the longer,
higher-quality E619/E620-lineage recipe once a synced checkpoint or a larger
committed corpus is available in this environment, to see whether the
`meaningful v1`/fidelity gains and the `structural_similarity` regression
observed here at 20 steps / n=3 persist at realistic scale.
