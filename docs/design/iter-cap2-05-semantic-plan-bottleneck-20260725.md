# CAP2-05 — Program-scale SemanticPlanV1 latent-codec bottleneck (2026-07-25)

Implementation wiring for Linear [SLM-325](https://linear.app/quickdeploy-ai/issue/SLM-325) (AP-029).
Bridges the CAP2-02 common `LatentCodec` interface (uniform scalar, mixed-radix
FSQ, binary LFQ, learned VQ, continuous) onto tensorized `SemanticPlanV1`
program factors instead of oracle integer state ids, and extends the
no-bypass audit from `KaryBottleneck` to the generic `LatentCodec` family.
Evidence: [iter-cap2-05-semantic-plan-bottleneck-20260725.json](iter-cap2-05-semantic-plan-bottleneck-20260725.json).

## What was added

- `src/slm_training/models/semantic_plan_factors.py` — deterministic
  tensorization of `SemanticPlanV1` into six fixed-width factor families:
  `inventory`, `cardinality`, `topology`, `binder_reference`,
  `property_role_value`, `style_layout`. Plans are canonicalized first so
  identical structure produces bit-identical tensors.
- `src/slm_training/models/latent_codec_trainer.py` — added
  `audit_no_bypass(model, x)`, generalizing
  `KaryBottleneck.audit_no_bypass` to any `LatentCodec` family: (1) decoder
  output recomputed from `codec.decode_input(encoding)` alone must equal the
  full forward output, and (2) for a learned (`semantic_trace`) encoder,
  zeroing the raw input must change the encoded hard code, ruling out a
  constant/dead encoder acting as an unencoded side channel.
- `src/slm_training/harnesses/experiments/cap2_semantic_plan_bottleneck.py` —
  `ProgramFactorArm`/`ProgramFactorResult`/`ProgramFactorMatrixReport`, a
  `fixture` mode wired against the SLM-144 deterministic fixture corpus, and
  `small_corpus`/`full_corpus` modes that require an immutable
  `SemanticPlanV1` JSONL manifest and fail closed without one.
- `scripts/run_cap2_semantic_plan_bottleneck.py` — CLI following the
  `run_cap2_bottleneck.py` conventions (`--mode`, `--corpus-path`, `--arms`,
  `--seeds`, `--dry-run`, JSON+Markdown report output).
- `tests/test_models/test_semantic_plan_factors.py` — determinism, width,
  and per-factor sensitivity tests for the tensorizer.
- `tests/test_models/test_latent_codec.py` — `audit_no_bypass` regression
  tests: true-positive (trained semantic_trace codec), true-negative
  (oracle_state codec, no learned encoder to audit), and two adversarial
  cases (decoder that leaks raw input; encoder zeroed to a constant).
- `tests/test_harnesses/experiments/test_cap2_05_semantic_plan_bottleneck.py`
  — matrix construction, per-codec reconstruction + no-bypass, corpus-mode
  fail-closed behavior, report versioning, and a regression check that the
  existing CAP2-01/02 oracle-state matrix (`cap2_bottleneck.py`) is
  unchanged.

## Fixture matrix

Run:

```bash
python -m scripts.run_cap2_semantic_plan_bottleneck \
  --mode fixture --fixture-count 12 --seeds 0 \
  --out-dir outputs/runs/cap2_semantic_plan_bottleneck
```

Recipe: CPU; SLM-144 fixture corpus, 12 records before the 80/20 train/val
split (9 train records used here); single seed; wiring-only honesty mode.

| arm | codec | levels | capacity | records | exact_rate | occupied | no_bypass | leakage |
| --- | --- | --- | ---: | ---: | ---: | ---: | :---: | :---: |
| plan_fsq_2_3_3_4_5 | mixed_radix_fsq | [2,3,3,4,5] | 360 | 9 | 1.0000 | 9 | True | False |
| plan_lfq_d6 | binary_lfq | [2,2,2,2,2,2] | 64 | 9 | 1.0000 | 9 | True | False |
| plan_vq_64_d8 | learned_vq | [64] | 64 | 9 | 1.0000 | 9 | True | False |
| plan_continuous_d6 | continuous | [6] | 6* | 9 | 1.0000 | 9 | True | False |
| plan_uniform_b2d6 | uniform_scalar | [2,2,2,2,2,2] | 64 | 9 | 1.0000 | 9 | True | False |

\* `plan_continuous_d6` reports `latent_dim=6` as its nominal "capacity" for
table alignment; it is not a discrete bottleneck and is excluded from
leakage classification, matching CAP2-02 convention.

Every arm's factor-wise distortion (code-distinguishability proxy; see
below) was `0.0000` for all seven measured factor families at this scale.

## Factor-wise distortion proxy

"Factor-wise distortion" is computed purely from each arm's trained hard
codes, without a second regression-style training pass (most factor
families here are not raw scalars, so a reconstruction-error metric would
not be meaningful): for every pair of records whose canonical
`plan_factor_fingerprints` value *differs* in a given factor family, the
metric is the fraction whose learned hard codes *fail* to also differ. It
measures whether the codec's bottleneck preserves each factor family's
distinctions in isolation, not reconstruction fidelity of the factor's raw
values.

The initial implementation used the model's returned `code_index` for this
comparison, which is a dummy all-zero placeholder for the continuous codec
family (it has no discrete flat index) — every pair would have been
misreported as "undistinguished" regardless of the actual latent vectors.
The harness instead compares `encoding.hard` (the real per-family hard code:
discrete symbols for the discrete families, the continuous latent vector
itself for `continuous`), which is meaningful across every codec family.

## No-bypass audit extension

`audit_no_bypass(model, x)` in `latent_codec_trainer.py` generalizes
`KaryBottleneck.audit_no_bypass`'s two-part check to any `LatentCodec`:

1. Recompute the decoder output from `codec.decode_input(encoding)` alone
   and require it equals the full `model(x, hard=True)` forward output —
   proves the decoder never sees `x` or `encoding.metadata` directly, only
   the codec's declared hard-code output.
2. When the codec has a learned (`semantic_trace`) encoder, zeroing the raw
   input must change the encoded hard code — proves the raw program-factor
   tensor actually determines the code, ruling out a constant/dead encoder
   that would let an unencoded gold factor reach the decoder unnoticed. This
   check is skipped for `oracle_state`-mode codecs, which have no learned
   encoder by construction (mirroring `KaryBottleneck`'s own audit).

All five program-factor arms above passed the audit (`no_bypass=True`).
Adversarial regression tests confirm the audit actually discriminates: a
decoder wired to leak raw input past the codec, and an encoder zeroed to a
constant, both correctly fail (`tests/test_models/test_latent_codec.py`).

## Hard gates

- No-bypass audit failures: 0
- Leakage violations: 0
- **PASS** — every arm's decoder input is proven codec-only and no
  below-capacity discrete arm achieved perfect reconstruction by chance.

## Honest caveats

- This is a deterministic CPU fixture run over 9 records, not a
  production-quality or ship claim. `semantic_trace` mode learns a genuine
  feature→code MLP (unlike CAP2-02's `oracle_state` mode, which learns a
  direct per-index embedding), so exact reconstruction is not guaranteed at
  larger record counts within a bounded step budget — the `plan_vq_64_d8`
  and `plan_uniform_b2d6` arms needed materially more training steps
  (`4000`/`3000` vs. the CAP2-02 oracle-state matrix's `2400`/`1200`) to
  reliably reach exact recall even at this small scale, which is expected
  and is not evidence about optimal hyperparameters at production scale.
- `SemanticPlanV1` has no first-class style/layout field today; the
  reference extractor folds layout signal (e.g. a `Stack`'s `direction`)
  into the `archetype.id` string. The `style_layout` tensorizer is a
  faithful bridge onto that existing string field, not a claim that
  style/layout is already first-class structured data in the schema. A
  future SemanticPlanV1 revision that adds a dedicated style/layout field
  would let this tensorizer read it directly instead of string-matching.
- The reference extractor never populates `min_cardinality`/`max_cardinality`
  on extracted role slots, so the `cardinality` factor family is all-zero
  for every fixture record in this run. This honestly reflects the current
  extractor's behavior, not a tensorizer defect; the acceptance criterion
  ("deterministic tensorization for ... cardinality") is satisfied by the
  tensorizer's contract and is exercised directly (with non-zero bounds) in
  `tests/test_models/test_semantic_plan_factors.py`.
- No claim is made about which codec family is best for real OpenUI
  semantic factors, nor about behavior at 100/1000+ record scale;
  `small_corpus`/`full_corpus` modes are wired but intentionally fail
  closed without an externally built immutable manifest, since this
  iteration does not synthesize one.
- Model-card/README updates are not required: no production checkpoint was
  created or promoted.
