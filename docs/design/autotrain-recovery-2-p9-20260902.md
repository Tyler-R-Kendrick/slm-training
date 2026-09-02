# Screening arms that can move weights (P9 / RC7, 2026-09-02)

Claim class: **harness change / fixture-demo** — no model was trained,
evaluated or promoted here. Stamped `harness.autoresearch.experiment_campaign
v275`. Companion cards on the integration branch: P2 (`policy.v3.json`
default), P4 (decode-budget feedback), P7 (certified corpus), P10 (heal
executors).

## Problem (RC7)

`_SCREENING_ARM_BANK` (`scripts/run_autotrain_continuous.py`) held 57 static
arms touching 52 catalog levers, yet:

- zero data-volume arms (no `train_version` lever), zero recipe arms other than
  one `steps` x2 depth confound, while the climb policy allows
  `recipe_tweak_knobs: [steps, lr, batch_size, seed, grad_accum, noise_rate]`;
- three arms whose only knobs are decode-cost levers, which cannot change
  trained weights and are therefore a guaranteed null under the screening
  primary `smoke.eval_nll` (policy.v3). Committed
  `evidence_ledger.v1.json`: `bounds` n_complete=50, mean_delta=0.0,
  m2_delta=0.0, n_positive=0; `canvas` n_complete=5, all null; `both` never
  measured;
- two training-recipe arms (`steps` x2, `batch1`) whose preregistered
  hypotheses are latency / cost claims that the NLL primary cannot test
  (`batch1` n_complete=11, n_positive=0).

Posterior-UCB over the ledger keeps sampling nulls from that pool.

## What changed

| Piece | Where |
| --- | --- |
| Six size-matched screening arms (data / recipe) | `_SCREENING_ARM_BANK` head |
| `_LATENCY_ARM_BANK` (5 arms, stable slugs) | beside the screening bank |
| Latency bank drawn only under a `latency_ms_p50` role primary | `_all_screening_arm_bank`, `_screening_primary_leaf`, `_latency_arms_active` |
| Self-control data arm filter | `_arm_is_self_control` (called from `_all_screening_arm_bank`) |
| Bank knob classifier | `thrash_regime.is_latency_only_arm` + `_bank_lever_categories` / `_latency_only_arm` |
| `lr` joins the confirmation identity set | `_LEVER_KNOB_KEYS` |
| Fill factor never overshoots the fitted floor | `_apply_arm_extras` (`base + 1` minimum for factor < 2; nearest-integer rounding) |
| New arms round-trip to slugs | `_arm_slug_from_knobs` (`lr`, `batch_size == 4`, non-control `train_version`) |
| Static data arms are OFAT levers, not I10 snapshot leftovers | `_slug_is_snapshot_arm`, `_STATIC_DATA_ARM_SLUGS` |
| Additive constants | `thrash_regime.py`: `LATENCY_PRIMARY_LEAF`, `DECODE_COST_LEVER_CATEGORIES`, `DECODE_COST_MODEL_LEVERS`, `TRAINING_LOSS_LEVER_SUFFIX`, `TRAINING_DURATION_LEVERS`, `LATENCY_HYPOTHESIS_SLUGS`, `STEPS_FACTOR_KEY` |
| Tests | `tests/test_scripts/test_run_autotrain_continuous.py::test_p9_*`, `tests/test_autoresearch/test_thrash_regime.py::test_is_latency_only_arm_*` |

## Arm table

Control (screening `_matrix`): `train_version` = policy
`defaults.train_version` (`openui_verified_train_v1`, 1,083 certified,
eval-decontaminated records), `batch_size` = 2, `lr` unset (=
`ModelBuildConfig.lr` 3e-4), `steps` = fitted floor x sps x 0.9. Every arm
below differs from the control in exactly the listed knobs; none touches a
`CAPACITY_SCALING_LEVERS` or `CONSTRAINT_WEAKENING_LEVERS` entry
(`require_size_matched_arms` passes for each, see tests).

### New screening arms (drawn under any primary)

| Slug | Knobs | Category (`lever_catalog`) | Hypothesis (falsifiable, one line) |
| --- | --- | --- | --- |
| `data-certified` | `train_version=openui_verified_train_v1` | data (ExperimentKnobs-only key) | Training on the certified, eval-decontaminated bucket (1,083 records) instead of the loop control corpus lowers smoke eval_nll at fixed model size without lowering parse_rate. |
| `data-strict` | `train_version=hillclimb_strict_v2` | data | Training on the fail-closed `hillclimb_strict_v2` corpus (676 records) instead of the loop control corpus lowers smoke eval_nll at fixed model size without lowering parse_rate. |
| `lr-x2` | `lr=6e-4` | training | Doubling the learning rate (3e-4 -> 6e-4) at fixed steps and model size lowers smoke eval_nll without lowering parse_rate. |
| `lr-x0.5` | `lr=1.5e-4` | training | Halving the learning rate (3e-4 -> 1.5e-4) at fixed steps and model size lowers smoke eval_nll without lowering parse_rate. |
| `batch-x2` | `batch_size=4` | training | Doubling batch_size (2 -> 4) at fixed steps and model size lowers smoke eval_nll without lowering parse_rate. |
| `steps-fill` | `_steps_factor=1/0.9` -> `steps = round(base / 0.9)` | run (training duration; charged as wall time, never parameters) | Filling the fitted train floor lowers smoke eval_nll without lowering parse_rate. |

`data-certified` is a **self-control** whenever the control already trains on
`openui_verified_train_v1` (the committed policy.v3 default after P7):
`_all_screening_arm_bank` drops it in that case, so today the live data arm is
`data-strict`; a loop whose control is the legacy `wf_smoke_v2` fixture gets
both. `openui_verified_v1` (1,682 records) is never a train arm: it contains
the validation/test families the certified smoke suites are sampled from.

### Moved to `_LATENCY_ARM_BANK` (drawn only when the screening role primary leaf is `latency_ms_p50`)

| Slug | Knobs | Category | Why it moved |
| --- | --- | --- | --- |
| `bounds` | `grammar_completion_bounds=True` | decode | Pure decode-cost lever; ledger n_complete=50, mean_delta=0.0, m2_delta=0.0. |
| `canvas` | `compact_active_canvas=True` | model (decode-work only, `DECODE_COST_MODEL_LEVERS`) | Pure decode-cost lever; ledger n_complete=5, all null. |
| `both` | both of the above | decode | Pure decode-cost levers; no ledger observation. |
| `steps` | `_steps_factor=2` | run | Training-duration lever, but its preregistered hypothesis is a unit-decode-latency claim (depth-confound cost control). |
| `batch1` | `batch_size=1` | training | Preregistered hypothesis is a latency claim; ledger n_complete=11, n_positive=0. |

Under a latency primary the latency bank is *prepended*, so the historical
cycle -> slug rotation (`bounds, canvas, both, steps, batch1, ...`) is
preserved for latency loops and ledger history stays attached to the same
slugs. Under the NLL primary the timeout-residual route
(`DECODE_RESIDUAL_SLUGS`) can no longer reach `bounds`/`canvas`/`both` and
lands on `cached-compiler-decision-margin` (P4 already routes budget timeouts
away from residual arms).

### Deliberately **not** moved (deviation from the card, evidence-driven)

The card listed `compiler-decision-token`, `compiler-decision-margin`,
`component-edge-margin`, `container-close` and `literal-margin` as pure
decode/run arms. `lever_catalog()` labels their knobs `decode`, but only by the
`compiler_` name prefix (`levers._category`); the weights enter the training
objective:

- `compiler_alignment_loss_weight` -> `TwoTowerModel` compiler-alignment loss
  over `gold_compiler_decisions` rows (`src/slm_training/models/twotower.py`,
  `alignment_w > 0.0` branch);
- `compiler_decision_token_loss_weight` -> decision-token position weights in
  the mask loss (same file, `decision_token_w`).

Ledger evidence is not comparable to `bounds`: `literal-margin` and
`container-close` have one observation each (0.0), the other three have none.
`is_latency_only_arm` therefore treats any `*_loss_weight` as a training
lever (`TRAINING_LOSS_LEVER_SUFFIX`), and these arms stay in the screening
bank. Their catalog category is an interface defect, recorded below.

### Deliberately **not** added

`noise-rate-0.2`: `noise_rate` is a `StubModel`-only lever
(`harnesses/model_build/plugin.py`) and not an `ExperimentKnobs` field (the
strict schema would reject it). A noise-rate arm could never move real-model
weights, which is the defect this card removes.

## Preregistration note

- Every arm is a static `(slug, hypothesis, knobs)` triple locked before
  outcomes; the selection rule (posterior-UCB over the evidence ledger, then
  rotation) is unchanged. The only selector-adjacent change is the bank
  composition predicate: latency arms are legal iff the *role primary leaf* is
  `latency_ms_p50`, and a `train_version`-only arm equal to the policy control
  corpus is dropped. Both predicates read policy configuration, never
  outcomes.
- Arms are size-matched: no capacity lever is touched (test
  `test_p9_new_screening_arms_are_size_matched_training_or_data_levers`).
  `steps-fill` is charged as wall time inside the already-fitted train floor;
  it never buys parameters and never exceeds the floor
  (`_apply_arm_extras` rounding + `base + 1` minimum).
- Slugs are stable; moved arms keep their ledger history. Confirmation
  identity now includes `lr` (`_LEVER_KNOB_KEYS`).
- Version stamp: `harness.autoresearch.experiment_campaign` v274 -> v275.

## Commands and results

```
PYTHONPATH=$PWD/src .venv/bin/python -m pytest \
  tests/test_scripts/test_run_autotrain_continuous.py \
  tests/test_autoresearch/test_thrash_regime.py -q
```

Result: 348 passed, 1 skipped (`test_run_autotrain_continuous.py` +
`test_thrash_regime.py`). One pre-existing failure on the merged integration
base, `test_sample_adequacy_report_reads_fixture_stats`, was blocking the
pre-commit shard: it wrote `wf_smoke_v2` stats while the P7 policy
`data_intervention` corpus is `openui_verified_train_v1`; the test now reads
the corpus from the policy (no driver change).

Bank state under the committed policy (`eval_nll` primary):
`_all_screening_arm_bank()` = 57 arms, head `data-strict, lr-x2, lr-x0.5,
batch-x2, steps-fill, component-plan, ...`, no latency arm present. Under a
`latency_ms_p50` primary: 62 arms, head `bounds, canvas, both, steps, batch1,
data-strict, ...`.

## Interface needs (for other owners)

1. `levers.lever_catalog()` has no row for `train_version` (an
   `ExperimentKnobs`-only key); the driver carries
   `_EXPERIMENT_ONLY_KNOB_CATEGORIES = {"train_version": "data"}` until the
   catalog (component `config.levers`) registers it.
2. `levers._category` labels `compiler_*_loss_weight` and
   `solver_energy_loss_weight` as `decode` by name prefix although they are
   training-objective weights; the classifier here overrides by suffix.
3. `policy.v3.recipe_tweak_knobs` lists `grad_accum` and `noise_rate`, neither
   of which is an `ExperimentKnobs` field (`grad_accum_steps` exists only on
   `ModelBuildConfig`; `noise_rate` is stub-only). They cannot become arms
   until the schema (and for `noise_rate`, the model) supports them.
4. `_sample_adequacy_report` follows `policy.data_intervention.train_version`;
   its test was pinned to `wf_smoke_v2` and is now policy-driven (P7 owners
   may prefer a dedicated fixture).
