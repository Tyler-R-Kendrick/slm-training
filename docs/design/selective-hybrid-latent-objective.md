# AP-033 selective-hybrid continuous/discrete program-factor latent objective (2026-07-26)

Implementation wiring for Linear [SLM-333](https://linear.app/quickdeploy-ai/issue/SLM-333)
(AP-033, milestone "Selective Hybrid Objective"). Tests, at fixture scale,
whether continuous latents can carry style/layout/global-intent ("soft")
program factors while exact topology, cardinality, and binder/reference
("precision-critical") factors remain fully discrete -- and whether a
selective-hybrid connector matches or exceeds a discrete-only baseline on
strict semantics while improving a soft-factor metric, without corrupting
precision-critical state.
Evidence: [selective-hybrid-latent-objective-20260726.json](selective-hybrid-latent-objective-20260726.json).

Unblocked by SLM-330 (AP-031, oracle-latent causal audit, PR #953,
[program-latent-causal-use.md](program-latent-causal-use.md)), SLM-332
(AP-032, hard-valid contrast/latent-geometry objectives, PR #956,
[iter-slm332-latent-geometry-20260725.md](iter-slm332-latent-geometry-20260725.md)),
and SLM-316 (AP-024, discrete abstract-plan conditioning,
`src/slm_training/models/abstract_plan_connector.py`).

## Decision

Test selective compression: continuous latents carry style/layout/global
intent while exact topology, cardinality, and binder/reference state remain
discrete. Compare `continuous_only`, `discrete_only`, `selective_hybrid`,
`all_continuous`, `random_continuous`, and `oracle_continuous` arms, all
resolving through one connector code path so control arms cannot silently
diverge, and verify with factor swaps that perturbing only the soft-factor
slice never corrupts precision-critical decoded state.

## Scope decision (read before the results)

The issue's target-path list names `abstract_plan_connector.py`,
`binder_precision_channel.py`, `program_latent_codec.py`, `twotower.py`, and
a new `selective_latent_connector.py`, plus a `BinderGraphV1` class. Before
writing any code this iteration confirmed by repo-wide search:
`abstract_plan_connector.py` and `twotower.py` **exist**;
`binder_precision_channel.py`, `program_latent_codec.py`, and any
`BinderGraphV1` class **do not exist anywhere in this repo** -- only
`AbstractPlanV1` (`src/slm_training/dsl/abstract_plan.py`) exists. Building
those two modules or that class now, just to satisfy the issue's literal
file list, would fabricate a discrete binder/reference symbol-table pipeline
that has no other basis in this codebase -- exactly what AP-031's own "Scope
decision" declined to do for the missing factors-to-program decoder, and the
same honest posture is used here:

* **No real discrete binder/reference pipeline exists**, so "precision-
  critical factors remain discrete" is implemented as literally *not
  bottlenecking* those factors through any learned codec: the decoder
  reconstructs them from the raw AP-029 tensorized `binder_reference`/
  `topology`/`cardinality`/`inventory` features directly (the strongest
  available "discrete/exact" proxy given the existing schema), while
  `all_continuous` (the required negative control) routes the same features
  through a `ContinuousLatentCodec` bottleneck instead, to show what is lost
  by giving up that exactness. This is a proxy for the issue's "discrete
  precision channel," not a claim that a real binder-graph implementation
  exists.
* **`selective_latent_connector.py` is the one genuinely new path** the
  issue names, built as the continuous-latent sibling of AP-024's
  `AbstractPlanConnector`: same gated-additive-bias idiom, same
  one-function-per-arm discipline
  (`resolve_selective_latent_vector`/`resolve_plan_vector`), same
  default-off "not constructed at all" guarantee for the off arm. It differs
  in exactly one place -- the input is already a continuous vector, so there
  is no codebook-slot embedding table to own.
* **"Condition TwoTower/MaskGIT through a gated continuous connector" is
  satisfied structurally, not by editing `twotower.py`/`blocks.py`.**
  `SelectiveLatentConnector.bias_for_vocab(vector, candidate_ids=...)` has
  the identical signature and `gate` attribute as
  `AbstractPlanConnector.bias_for_vocab`, and
  `DenoiserTower._apply_plan_connector_bias` (`src/slm_training/models/blocks.py`)
  never does an `isinstance` check -- it only calls `.bias_for_vocab(...)`
  and reads `.gate`. `tests/test_models/test_selective_latent_connector.py::test_selective_latent_connector_is_a_structural_drop_in_for_set_plan_connector`
  attaches a `SelectiveLatentConnector` through the exact same
  `DenoiserTower.set_plan_connector`/`project` path AP-024's own connector
  uses and proves it composes -- zero lines changed in `blocks.py` or
  `twotower.py`. AP-024 required editing those files because it shipped as
  real (if default-off) production conditioning with its own head/trainer
  wiring; AP-029 through AP-032 (this issue's own listed prerequisites)
  deliberately did **not** touch `twotower.py` for this same class of
  bounded factor-reconstruction experiment, reusing a small
  `FactorReconstructionModel`-style scaffold instead. This iteration follows
  that precedent rather than fabricating new production wiring to check a
  box.
* **The soft/precision-critical factor routing split is a judgment call,
  documented rather than hidden.** `PRECISION_CRITICAL_FACTOR_FAMILIES` =
  `(inventory, cardinality, topology, binder_reference)` -- exact structural
  and executable-semantics state, matching the issue's own "topology,
  cardinality, and binder/reference" list plus `inventory` (which component
  families exist, equally executable-semantics-determining). `SOFT_FACTOR_FAMILIES`
  = `(property_role_value, style_layout)` -- the closest existing proxy for
  "style/layout/global intent". `style_layout` is a direct match; grouping
  `property_role_value` (per-role required flags + mean confidence) as
  "soft" is a judgment call: it is not literally "intent" (no first-class
  intent field exists in `SemanticPlanV1`, the same honest caveat
  `semantic_plan_factors.py` already documents for `style_layout`'s own
  archetype-string bridge), but corrupting it does not change what
  components exist or how they are bound, unlike the four precision-critical
  families. `PROGRAM_FACTOR_FAMILIES`'s declared order already places the
  four precision-critical families first and the two soft families last, so
  the split is a single contiguous slice boundary
  (`split_soft_precision_features`), never a gather -- asserted at import
  time in `selective_latent_connector.py`.

## What was added

- `src/slm_training/models/selective_latent_connector.py` -- `SelectiveLatentArm`
  (six arms), `resolve_selective_latent_vector` (the one function every arm
  resolves through), `SelectiveLatentConnector` (gated-additive projection,
  structurally drop-in for `DenoiserTower.set_plan_connector`),
  `PRECISION_CRITICAL_FACTOR_FAMILIES`/`SOFT_FACTOR_FAMILIES`, and
  `split_soft_precision_features` (the factor routing).
- `src/slm_training/harnesses/experiments/cap2_selective_hybrid_latent.py` --
  `SelectiveHybridModel` (per-arm precision/soft routing model reusing
  `ContinuousLatentCodec`), `evaluate_arm` (trains + scores one arm plus its
  factor-swap falsification test), `run_selective_hybrid` (all six arms,
  paired stats vs. `discrete_only`, promotion disposition).
- `scripts/run_cap2_selective_hybrid_latent.py` -- CLI following the
  existing `run_cap2_*` conventions (`--fixture-count`, `--arms`,
  `--train-steps`, `--ci-resamples`, `--dry-run`, JSON+Markdown report
  output).
- `tests/test_models/test_selective_latent_connector.py` -- connector/arm
  resolution unit tests plus the DenoiserTower structural-drop-in proof.
- `tests/test_harnesses/experiments/test_cap2_08_selective_hybrid_latent.py` --
  22 tests: arm coverage, the `discrete_only` bit-exactness regression tests
  (hand-built minimal reference model comparison, content-independence
  check), the factor-swap falsification test for every arm, per-arm metric
  coverage, and an end-to-end six-arm run.

## Method

For each **arm** (`discrete_only`, `continuous_only`, `selective_hybrid`,
`all_continuous`, `random_continuous`, `oracle_continuous`):

1. Tensorize the SLM-144 fixture corpus's train/holdout split (reusing
   AP-031's `load_causal_audit_corpus` unchanged) with AP-029's
   `semantic_plan_factors` tensorizer, then split each feature vector into
   `(precision_critical, soft)` via `split_soft_precision_features`.
2. Build a `SelectiveHybridModel`: a `precise_decoder` reconstructing the
   precision-critical vector (from the raw features directly, except
   `all_continuous`, which routes through a narrow -- 4-dim, vs. the
   precision-critical vector's full width -- `ContinuousLatentCodec`
   bottleneck as the negative control); a learned `soft_baseline`
   (a constant, "zero continuous signal" prediction); and, for every arm but
   `discrete_only`, a `soft_codec` (`ContinuousLatentCodec` trained on the
   soft-factor features) plus a `SelectiveLatentConnector` whose
   gated-additive bias corrects that baseline. `discrete_only` never
   constructs `soft_codec`/`connector` at all -- zero extra parameters,
   matching `AbstractPlanConnector`'s own "off" guarantee.
3. Train end-to-end (Adam, MSE against the full concatenated factor vector)
   on the train split; freeze and evaluate on the held-out split with
   `hard=True`.
4. Score `overall_strict_rate` (every one of the six `PROGRAM_FACTOR_FAMILIES`
   at/under AP-030's `DISTORTION_THRESHOLD=0.05`, reused unchanged),
   `soft_strict_rate` (soft families only -- the soft-factor-specific
   metric the issue asks for), and `precision_strict_rate` (precision-
   critical families only), plus per-family MSE, per held-out example.
5. Run the **factor-swap falsification test**: with the trained model
   frozen, recompute `precise_forward` (which never reads `soft_features` in
   any arm, including `all_continuous` -- see
   `SelectiveHybridModel.precise_forward`) after independently perturbing
   the soft-factor input three ways (between-example shuffle, zeroed,
   fresh-noise), and assert the precision-critical reconstruction is
   bit-identical (`torch.equal`) each time.
6. Compute paired bootstrap CIs (SLM-183 `bootstrap_paired_ci`, 1000
   resamples) and an exact McNemar test (`exact_paired_binary_test`) for
   every non-`discrete_only` arm's `overall_strict_rate`/`soft_strict_rate`
   against `discrete_only`, plus the `binder_reference`-family MSE delta
   (the negative-control metric the acceptance criteria ask to report).

## Fixture run (2026-07-26)

CPU; SLM-144 fixture corpus, `--fixture-count 32` -> 25 train / 7 held-out
records; single seed (0); default 400-step arms; 1000 bootstrap resamples.

```bash
python -m scripts.run_cap2_selective_hybrid_latent \
  --fixture-count 32 --out-dir outputs/runs/cap2_selective_hybrid_latent
```

Run id `a88bcaf9d8319f29`, disposition **`promote_selective_hybrid`**.

### Results per arm (n=7 held-out)

| arm | overall_strict | soft_strict | precision_strict | binder_reference MSE | soft MSE | soft codec collapsed | precise codec collapsed |
| --- | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| discrete_only | 0.286 | 0.286 | 1.000 | 5.5e-08 | 0.0937 | False | False (no codec) |
| continuous_only | 1.000 | 1.000 | 1.000 | 5.5e-08 | 5.9e-16 | False | False (no codec) |
| **selective_hybrid** | **1.000** | **1.000** | 1.000 | 5.5e-08 | 1.6e-16 | False | False (no codec) |
| all_continuous (negative control) | 1.000 | 1.000 | 1.000 | **1.40e-03** | 1.3e-14 | False | False |
| random_continuous | 0.286 | 0.286 | 1.000 | 5.5e-08 | 0.0927 | False | False (no codec) |
| oracle_continuous | 1.000 | 1.000 | 1.000 | 5.5e-08 | 5.0e-16 | False | False (no codec) |

### Paired comparisons vs. `discrete_only` (1000-resample bootstrap CI)

| arm | overall_strict_rate delta | 95% CI | soft_strict_rate delta | binder_reference MSE delta |
| --- | ---: | ---: | ---: | ---: |
| continuous_only | +0.714 | [0.429, 1.000] | +0.714 | +0.0 |
| **selective_hybrid** | **+0.714** | **[0.429, 1.000]** | **+0.714** | **+0.0** |
| all_continuous | +0.714 | [0.429, 1.000] | +0.714 | **+1.40e-03** |
| random_continuous | +0.000 | [0.000, 0.000] | +0.000 | +0.0 |
| oracle_continuous | +0.714 | [0.429, 1.000] | +0.714 | +0.0 |

### Factor-swap falsification test

Every arm (all six), every perturbation (`shuffled_between_example`,
`zeroed`, `random`): `precision_recon_bit_exact = True`,
`max_abs_precision_delta = 0.0` exactly. 18/18 checks pass. This holds by
construction -- `SelectiveHybridModel.precise_forward` has no code path that
reads `soft_features` in any arm, including `all_continuous` (whose
bottleneck is trained on `precision_features` alone) -- and the test
confirms that guarantee holds in the actually-running trained model, not
merely by reading the source.

## Findings

- **`selective_hybrid` matches `discrete_only`'s precision-critical
  exactness (`precision_strict_rate=1.000` for both) while dramatically
  improving the soft-factor metric**: `soft_strict_rate` 1.000 vs. 0.286,
  `soft_mse_overall` 1.6e-16 vs. 0.0937 (paired 95% CI on the rate delta
  `[0.429, 1.000]`, strictly positive). `overall_strict_rate` (both channels
  must pass) also improves 0.286 -> 1.000, satisfying the acceptance
  criterion "selective hybrid matches/exceeds discrete-only strict
  semantics and improves at least one soft-factor metric without
  binder/reference regression": `binder_reference` MSE is bit-identical to
  `discrete_only`'s (5.475362740980927e-08 for both, to full float64
  precision) since `selective_hybrid`'s precision branch is the same
  unbottlenecked pass-through as `discrete_only`'s.
- **`random_continuous` is indistinguishable from `discrete_only`**
  (identical `overall_strict_rate`/`soft_strict_rate`, CI `[0.000, 0.000]`):
  confirms the improvement above is genuinely driven by the soft-latent
  *content*, not merely by the connector's extra parameters/architecture --
  a content-null control at matched architecture shows zero effect, exactly
  as it should.
- **`all_continuous` (negative control) shows the expected direction of
  binder/reference degradation**: MSE rises from 5.5e-08 (exact/discrete) to
  1.40e-03 (bottlenecked) -- roughly four orders of magnitude worse, though
  still well under `DISTORTION_THRESHOLD` (0.05) at this tiny fixture scale
  and 400-step budget. This is reported honestly as directionally confirming
  but not scary at this scale: giving up exactness for precision-critical
  factors measurably hurts binder/reference reconstruction even in a bounded
  4-latent-dim bottleneck, but a larger corpus/deeper bottleneck sweep would
  be needed to find where it crosses the strict-semantics-proxy threshold.
- **The factor-swap falsification test passes 18/18 (every arm, every
  perturbation) with exact bit-identity, not just a small measured effect.**
  No continuous factor promotes on the back of an unpredictable
  precision-critical coupling -- there is none, by construction and by
  measurement.
- **No encoder collapse.** `soft_codec_collapsed=False` for every arm that
  constructs one; `precise_codec_collapsed=False` for `all_continuous`. The
  collapse check (the same zero-input-must-change-the-code check
  `latent_codec_trainer.audit_no_bypass` uses, applied per-codec since this
  composite model's two independent codecs do not share
  `LatentCodecModel`'s single-codec interface) found no dead/constant
  encoder.

## Disposition

**`promote_selective_hybrid`.** Every hard acceptance-criteria bullet in the
issue is satisfied at this fixture scale:

- "Selective hybrid matches/exceeds discrete-only strict semantics and
  improves at least one soft-factor metric without binder/reference
  regression" -- **met**: `overall_strict_rate` 1.000 >= 0.286,
  `soft_strict_rate`/`soft_mse_overall` both improve, `binder_reference` MSE
  is bit-identical to `discrete_only`'s.
- "Report all-continuous binder/reference results as a negative control" --
  **met and reported** (1.40e-03 vs. 5.5e-08, directionally confirming, not
  overstated as breaching the strict-semantics threshold at this scale).
- "No continuous factor promotes if swaps unpredictably change
  precision-critical state" -- **met**: 18/18 factor-swap checks pass
  bit-exact for every arm.
- "Off/discrete-only mode remains bit-exact" -- **met**: see the Hard gates
  section below.

This is a **fixture-scale, single-seed finding**, not a production or ship
claim (see Honest caveats). The clean separation here (`discrete_only`
0.286 vs. every content-bearing continuous arm at 1.000, `random_continuous`
flat at `discrete_only`'s level) is a strong *directional* signal that the
selective-hybrid routing is sound and not an artifact, but the effect size
at n=7 held-out and a 12-dimensional soft-factor space is not evidence of
production readiness.

## Hard gates

- Encoder-collapse failures: 0/2 constructed codecs (soft codec across 5
  arms; precise codec for `all_continuous`).
- Factor-swap falsification failures: 0/18 (every arm x every perturbation).
- `discrete_only` bit-exactness: enforced by
  `tests/test_harnesses/experiments/test_cap2_08_selective_hybrid_latent.py::test_discrete_only_forward_is_bit_identical_to_a_hand_built_reference_model`
  (hand-built minimal reference module, copied weights, `torch.equal` on
  forward output) and
  `::test_discrete_only_forward_ignores_soft_features_content`
  (output invariant to `soft_features` content, not just its batch size);
  both pass. Also proven at the `DenoiserTower.project` composability layer
  by `tests/test_models/test_selective_latent_connector.py::test_project_is_byte_identical_when_no_selective_connector_attached`.
- `promote_selective_hybrid`'s claim requires `no_bypass`-equivalent
  collapse checks to pass **and** every destructive factor-swap condition to
  be bit-exact **and** the paired `overall_strict_rate` delta's 95% CI lower
  bound to be strictly positive -- enforced in code
  (`run_selective_hybrid`'s disposition logic), not asserted from a single
  point estimate.

## Honest caveats

- **Fixture-scale evidence only.** 25 train / 7 held-out records, single
  seed (0), 400 training steps. Not a production or ship claim; no
  checkpoint was created or promoted, so no `docs/MODEL_CARD.md`/README
  update is required.
- **Strict-semantics proxy, not meaning-v2.** `overall_strict_rate`/
  `soft_strict_rate`/`precision_strict_rate` are AP-030's `DISTORTION_THRESHOLD`
  MSE-threshold proxy over reconstructed program factors, not
  `binding_aware_meaningful_v2`. No claim in this doc should be read as a
  claim about full-program semantic correctness.
- **No real discrete binder/reference symbol-table pipeline exists** in this
  repo (`binder_precision_channel.py`/`program_latent_codec.py`/
  `BinderGraphV1` were not fabricated -- see Scope decision above).
  "Precision-critical stays discrete" is implemented as "not bottlenecked
  through a learned codec," not as a claim about a real discrete AST/binder
  pipeline.
- **`twotower.py`/`blocks.py` were not modified.** Composability with
  `DenoiserTower` is proven structurally (duck-typed drop-in), not by
  editing the production conditioning wiring -- see Scope decision above.
- **`all_continuous`'s binder/reference regression is real but small at this
  scale** (1.40e-03, still under `DISTORTION_THRESHOLD`). A larger
  corpus/narrower bottleneck sweep would be needed to find the point at
  which the strict-semantics-proxy actually fails for that arm.
- **n=7 held-out is small.** The clean 0.286-vs-1.000 separation and the
  `random_continuous` null result both support that this is a genuine
  signal rather than noise, but a larger held-out sample is needed before
  treating the effect size itself as a stable estimate.
