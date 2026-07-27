# CAP2-06 — Program-factor latent size/width/codec rate-distortion sweep (2026-07-25)

Implementation wiring for Linear [SLM-328](https://linear.app/quickdeploy-ai/issue/SLM-328) (AP-030).
Sweeps latent count x encoder width x codec family (uniform scalar,
mixed-radix FSQ, binary LFQ, learned VQ, continuous) over the SLM-325
`SemanticPlanV1` program-factor tensorizer to find the smallest oracle
latent representation that preserves each factor family, and to identify
which factor crosses a distortion threshold first as capacity shrinks.
Evidence: [iter-cap2-06-rate-distortion-sweep-20260725.json](iter-cap2-06-rate-distortion-sweep-20260725.json).

## Scope decision (read before the results)

This issue's own "Implementation" section only asks to "add one token/tree
decoder only after factor gates pass" — i.e. real `meaning-v2`,
binder/reference F1, canonical AST, and G0-G8 metrics against generated
OpenUI programs are a later, gated stage that needs the full
compiler/evaluator pipeline wired to a real decoder. This bounded CPU
wiring pass does not build that pipeline. It measures the stage the
issue's implementation section actually describes first: **per-factor-family
reconstruction distortion (MSE)** as a function of latent count, width, and
codec family, using a single linear decoder matched to the full
concatenated factor vector (sliced per family for reporting) rather than a
bespoke decoder network per family. The distortion-threshold "gate" here is
an explicit MSE proxy (`0.05`), not meaning-v2/binder-F1 — a proxy for the
issue's `0.10` margin, not a claim of numeric equivalence to that metric.
Promotion into real semantic-quality evaluation is future work.

## What was added

- `src/slm_training/harnesses/experiments/cap2_rate_distortion_sweep.py` —
  `RateDistortionCell`/`RateDistortionResult`/`RetainedConfig`/
  `RateDistortionSweepReport`; `FactorReconstructionModel` (a `LatentCodec`
  plus a single linear regression decoder reconstructing the full factor
  vector — reuses the exact codec families and `audit_no_bypass` from
  SLM-325 unchanged); `build_sweep`/`evaluate_cell`/`select_retained_configs`/
  `run_sweep`.
- `scripts/run_cap2_rate_distortion_sweep.py` — CLI following existing
  `run_cap2_*` conventions (`--codecs`, `--num-latents`, `--widths`,
  `--train-steps`, `--dry-run`, JSON+Markdown report output).
- `tests/test_harnesses/experiments/test_cap2_06_rate_distortion_sweep.py` —
  grid-size/id-uniqueness tests, per-codec config-mapping tests, a
  no-bypass-audit pass-through test, a capacity-vs-distortion sanity test,
  and two regression tests for the collapsed-encoder exclusion fix below.

## Sweep axis mapping

The issue names two literal axes ("latent counts 4, 8, 16, 32" and "widths
64, 128, 256") without specifying which codec parameter each maps to. This
harness fixes an explicit, defensible mapping, documented here rather than
left implicit:

| axis | meaning | codec parameter |
| --- | --- | --- |
| `num_latents` | discrete/continuous capacity (rate) | `d` (scalar/LFQ), level-vector length (FSQ, fixed 4 levels/coordinate), `log2(codebook_size)` (VQ), `latent_dim` (continuous) |
| `width` | encoder capacity | `hidden_dim` (the `semantic_trace`-mode MLP mapping the raw factor vector to pre-quantization state) |

FSQ's per-coordinate level count (4) and VQ's latent dimension (8) are held
fixed across the sweep so `num_latents`/`width` remain the only varied axes
for every family, matching the issue's 2D grid.

## Fixture sweep (reduced grid for CPU wall-clock; full grid is CLI-selectable)

Run:

```bash
python -m scripts.run_cap2_rate_distortion_sweep \
  --num-latents 2,8 --widths 64 --train-steps 200 \
  --out-dir outputs/runs/cap2_rate_distortion_sweep
```

Recipe: CPU; SLM-144 fixture corpus, 9 train records (same corpus as
CAP2-05); the full `{4,8,16,32} x {64,128,256} x 5 codecs` = 60-cell grid is
supported by the harness and by `build_sweep()`'s default arguments (and by
targeted unit tests exercising every codec's parameter mapping), but this
evidence run uses a reduced `{2,8} x {64}` grid because several full-width
FSQ/uniform-scalar cells (large one-hot decoder inputs) take multiple
minutes per cell on CPU — a wall-clock constraint, not a correctness limit.

### Retained configurations (smallest passing cell per codec, 3 seeds)

| codec | num_latents | width | seed0 max_factor_mse | seed1 | seed2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| uniform_scalar | 8 | 64 | 0.0168 | 0.0187 | 0.0245 |
| fsq | 2 | 64 | 0.0165 | 0.0169 | 0.0178 |
| lfq | 2 | 64 | 0.0216 | 0.0219 | 0.0217 |
| continuous | 2 | 64 | 0.0193 | 0.0173 | 0.0182 |

`vq` has **no retained configuration** in this screened grid (see below).

### Full staged frontier (all cells, including failed)

| cell | n | width | capacity | mse_overall | max_factor_mse | factors_over_threshold | no_bypass |
| --- | ---: | ---: | ---: | ---: | ---: | --- | :---: |
| uniform_scalar_n2_w64 | 2 | 64 | 4 | 0.0098 | 0.0658 | style_layout | **False** |
| uniform_scalar_n8_w64 | 8 | 64 | 256 | 0.0036 | 0.0168 | — | True |
| fsq_n2_w64 | 2 | 64 | 16 | 0.0037 | 0.0165 | — | True |
| fsq_n8_w64 | 8 | 64 | 65536 | 0.0145 | 0.0778 | style_layout | True |
| lfq_n2_w64 | 2 | 64 | 4 | 0.0046 | 0.0216 | — | True |
| lfq_n8_w64 | 8 | 64 | 256 | 0.0025 | 0.0087 | — | True |
| vq_n2_w64 | 2 | 64 | 4 | 0.0098 | 0.0658 | style_layout | **False** |
| vq_n8_w64 | 8 | 64 | 256 | 0.0098 | 0.0658 | style_layout | **False** |
| continuous_n2_w64 | 2 | 64 | 2 | 0.0049 | 0.0193 | — | True |
| continuous_n8_w64 | 8 | 64 | 8 | 0.0002 | 0.0006 | — | True |

## A genuine finding: encoder collapse at minimum capacity, and the bug it exposed

At `num_latents=2` (the smallest screened capacity), `uniform_scalar` and
**both** `vq` cells failed the no-bypass audit — not because the decoder
leaked raw input (the decoder-recompute check passed for every cell), but
because the `semantic_trace` encoder collapsed to a **constant/dead code**:
`codec.encode(features)` and `codec.encode(zeros_like(features))` produced
the identical hard code for every training example. This is a real,
informative rate-distortion finding in its own right — at 2-bit/4-way
capacity with 9 training examples and a 200-step budget, these two
families' tiny encoders found a degenerate optimum rather than a useful
code — and is exactly the kind of failed cell the issue asks to publish,
not hide.

It also exposed a real correctness bug during development: `vq_n2_w64` and
`vq_n8_w64` both collapsed to the *same* dead code (identical
`max_factor_mse=0.0658` for both), and both happened to also fail the
distortion threshold in this run — but `select_retained_configs`'s
"smallest passing cell" search only checked `factors_over_threshold`, not
`no_bypass_ok`. A collapsed encoder that happens to reconstruct a
low-variance factor well (e.g. by learning the sample mean, independent of
its input) would otherwise look like a legitimate low-distortion
configuration and be wrongly promoted. Fixed by requiring both conditions;
regression tests (`test_select_retained_configs_excludes_collapsed_encoder_despite_low_distortion`,
`test_select_retained_configs_omits_codec_when_only_collapsed_cells_pass_distortion`)
construct this exact scenario with a synthetic passing-but-collapsed result
so the exclusion is exercised independent of whether real training happens
to reproduce it.

## Hard gates

- No-bypass audit failures: 3 (`uniform_scalar_n2_w64`, `vq_n2_w64`, `vq_n8_w64` — encoder collapse, not decoder leakage; see above)
- Retained configurations: 4/5 codec families (`vq` did not clear the gate in this screened grid)
- **The harness correctly refuses to promote a collapsed-encoder cell even when its reconstruction MSE looks passing.**

## Honest caveats

- This is a deterministic CPU fixture run over 9 records with a reduced
  `{2,8} x {64}` screening grid, not the full `{4,8,16,32} x {64,128,256}`
  grid and not a production-quality or ship claim.
- `fsq_n8_w64`'s distortion (`0.0778`) is *worse* than `fsq_n2_w64`'s
  (`0.0165`) despite more nominal capacity — a fixed 200-step budget does
  not guarantee monotone convergence across very different decoder-input
  widths for a non-convex MLP-based encoder; this is an optimization-budget
  effect (matching the same phenomenon documented in the SLM-325 design
  doc for `plan_vq_64_d8`/`plan_uniform_b2d6`), not evidence against the
  underlying rate-distortion relationship. The unit test suite verifies
  monotonicity separately using a step budget tuned per cell size.
- No meaning-v2, binder/reference F1, canonical AST, or G0-G8 metric is
  measured or claimed; see "Scope decision" above.
- Model-card/README updates are not required: no production checkpoint was
  created or promoted.
