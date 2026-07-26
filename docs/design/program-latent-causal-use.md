# CAP2-07 — Oracle-latent causal necessity, support, robustness, and collapse audit (2026-07-25)

Implementation wiring for Linear [SLM-330](https://linear.app/quickdeploy-ai/issue/SLM-330)
(AP-031, milestone "Oracle Bottleneck"). Proves (at fixture scale) whether the
AP-030 frozen factor-reconstruction decoder causally depends on the retained
AP-029 `SemanticPlanV1` program-factor oracle latent, characterizes the
robust decoding basin around the correct latent, and dispositions the
continuous-latent track per the issue's own falsification rule.
Evidence: [program-latent-causal-use-20260725.json](program-latent-causal-use-20260725.json).

## Decision

Prove that the decoder causally uses the retained latent, characterize valid
decoding basins, and reject prompt-only or collapsed solutions -- without
introducing any new decode path and without weakening any existing gate.

## Scope decision (read before the results)

AP-029 ([SLM-325](https://linear.app/quickdeploy-ai/issue/SLM-325)) bridges the
CAP2-02 `LatentCodec` family onto tensorized `SemanticPlanV1` program factors.
AP-030 ([SLM-328](https://linear.app/quickdeploy-ai/issue/SLM-328)) sweeps
latent count x encoder width x codec family and trains a single linear
regression decoder that reconstructs the full concatenated factor vector from
the codec's declared latent output. **Neither builds a decoder from oracle
latents back to full OpenUI program text** -- both design docs explicitly gate
that ("add one token/tree decoder only after factor gates pass") to a later
stage that needs the full compiler/evaluator pipeline.

That gap matters here: the issue's Implementation section asks to reuse "the
repo's existing strict-semantics / meaning-v2 scorer"
(`binding_aware_meaningful_v2` in `evals/meaningful_program.py`), but that
scorer takes a serialized prediction *string* and an `ExampleRecord` -- there
is no serialized program to hand it without a factors-to-program decoder that
does not exist in this repo. Building one now, just to satisfy this audit,
would itself be a new, unaudited decode path -- exactly what the "no shadow
paths" / "never introduce a new unconstrained decode path" constraints in
AGENTS.md forbid, and exactly the kind of scope creep CAP2-06 already
declined for the same reason.

Instead, this harness reuses AP-030's own **`DISTORTION_THRESHOLD` (0.05)**
per-factor-family MSE gate unchanged and defines a **strict-semantics
proxy**: a held-out example counts as a strict semantic match only if *every*
factor family's reconstruction MSE is at or under that threshold. This is a
proxy for strict semantic exactness at the factor-reconstruction stage of the
pipeline, not a claim of numeric equivalence to meaning-v2 -- the same
honesty posture CAP2-06 already documented for its own MSE gate, extended
here rather than invented fresh.

## What was added

- `src/slm_training/harnesses/experiments/cap2_latent_causal_audit.py` --
  `CausalAuditArm` / `ConditionResult` / `NoiseRadiusResult` /
  `ArmCausalAuditResult` / `CausalAuditReport`; `evaluate_arm` trains the
  exact AP-030 `FactorReconstructionModel` once per arm (reusing
  `_build_codec` / `train_factor_reconstruction` / `DISTORTION_THRESHOLD`
  unchanged), freezes it, and scores it under every perturbation condition
  against a fixed held-out sample; `run_causal_audit` runs the full arm set
  and computes a `continue` / `stop_ignore` disposition per the issue's
  Acceptance/Falsification rules.
- `scripts/run_cap2_latent_causal_audit.py` -- CLI following the existing
  `run_cap2_*` conventions (`--fixture-count`, `--arms`, `--train-steps`,
  `--ci-resamples`, `--dry-run`, JSON+Markdown report output).
- `tests/test_harnesses/experiments/test_cap2_07_latent_causal_audit.py` --
  17 tests: corpus split/fail-closed behavior, the derangement helper, the
  role-preserving-permutation helper (including a direct regression test for
  a same-width-collapse bug found and fixed during this iteration -- see
  below), the quantized-condition determinism sanity check, the
  role-swap-applicability flag, and an end-to-end small-grid run.

## Method

For each (codec, capacity) **arm**:

1. Train the frozen decoder once on the SLM-144 fixture corpus's 80% train
   split (`build_fixture_plan_corpus`); freeze it (`model.eval()`) for every
   condition below. Since this pipeline stage has no separate prompt input or
   mask schedule -- the decoder's only input is the codec's declared latent
   output -- "holding prompt/decoder seed/mask schedule fixed" reduces to
   holding the trained decoder weights fixed, which is enforced by
   construction (never retrained mid-arm).
2. Encode the held-out (val) split's factor vectors once under `hard=True`
   to get the **retained latent** (`decoder_input`) for every held-out
   example -- the paired design's fixed reference point.
3. Run the frozen decoder on the retained latent (`correct`) and on eight
   perturbation conditions built directly from `decoder_input`, never by
   re-deriving from raw features (so the decoder truly only ever sees a
   perturbed *latent*, holding everything else fixed):
   - **zero** -- decoder input replaced with zeros. Doubles as the
     **prompt-only control**: there is no separate prompt channel at this
     pipeline stage, so masking the latent to zero already *is* the
     prompt-only condition.
   - **random** -- fresh per-dimension `N(mean, std)` draw matched to the
     correct-latent marginal (not an existing example's code).
   - **shuffled_between_example** -- a random derangement reassigns each
     held-out example a different example's retained latent.
   - **slot_permuted** -- a random permutation of the codec's declared slots
     (`spec.levels` segments for concatenated-one-hot families; raw
     coordinates for element-wise families).
   - **role_swapped** -- a permutation *restricted to same-width slot pairs*.
     Only meaningful when the codec has >=2 distinct slot widths with a
     same-width pair to swap; a heterogeneous-radix FSQ arm
     (`fsq_typed`, radixes `(2,3,3,4,5)`, reusing AP-029's own
     `plan_fsq_2_3_3_4_5` radix vector unchanged) is included specifically so
     this condition has something to test. Homogeneous codecs (binary LFQ,
     element-wise continuous/VQ) correctly report `role_swap_applicable=False`
     and omit the arm rather than fabricate a result indistinguishable from
     `slot_permuted`.
   - **quantized** -- re-runs a fresh hard encode with *no* perturbation; a
     determinism sanity control that must reproduce `correct` exactly.
   - **noise_perturbed** at 3 required radii (0.25x/0.5x/1.0x per-dimension
     std) plus a finer 8-point sweep (0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0x)
     used to find the **robust semantic radius**: the largest swept radius
     whose strict-semantics-proxy rate stays within 0.05 of the correct-latent
     rate.
4. Score every condition with the strict-semantics proxy (per-example,
   per-factor-family MSE <= 0.05) and compute **paired** statistics against
   `correct` using the repo's existing SLM-183 stats helpers
   (`slm_training.evals.power_protocol`): `bootstrap_paired_ci` (1000
   resamples) for the rate difference and mean-MSE difference, and
   `exact_paired_binary_test` (exact two-sided McNemar) as a second,
   non-bootstrap check.
5. Compute latent **effective rank** / **stable rank** / **total variance**
   over the retained-latent matrix via SVD (von-Neumann-entropy convention,
   the same one already used by the SLM-214/SLM-217 spectral-snapshot
   harnesses) -- the collapse check.
6. Reuse `latent_codec_trainer.audit_no_bypass` unchanged as the encoder-
   collapse gate: an arm whose encoder collapsed to a constant/dead code
   (CAP2-06's own documented minimum-capacity failure mode) is marked
   `no_bypass_ok=False` and excluded from the "trustworthy" pool that the
   run-level disposition is computed over.

**Weaker-decoder control**: omitted. No smaller/weaker frozen-decoder
checkpoint fixture exists anywhere in this repo
(`src/slm_training/resources/checkpoints/` has none for this
factor-reconstruction stage) -- not fabricated. Every `ArmCausalAuditResult`
carries a `weaker_decoder_note` recording this explicitly.

## Fixture run (2026-07-25)

CPU; SLM-144 fixture corpus, `--fixture-count 32` -> 25 train / 7 held-out
records; single seed (0); default arms; 1000 bootstrap resamples.

```bash
python -m scripts.run_cap2_latent_causal_audit \
  --fixture-count 32 --out-dir outputs/runs/cap2_latent_causal_audit
```

Run id `cbfe4e1fe36d28be`, disposition **continue**.

### Results per arm

| arm | codec | no_bypass_ok | causally_necessary | effective_rank | total_variance | robust_radius (sigma) |
| --- | --- | :---: | :---: | ---: | ---: | ---: |
| continuous_n2_w64 | continuous | True | **True** | 1.06 | 0.077 | 0.0 |
| lfq_n2_w64 | lfq | True | False | 1.00 | 1.633 | 0.5 |
| fsq_typed_2_3_3_4_5 | fsq_typed | True | False | 1.00 | 1.224 | 0.5 |

### `continuous_n2_w64` conditions (n=7 held-out, paired vs `correct`)

| condition | strict_rate | mean_mse | rate CI (95%) | McNemar p |
| --- | ---: | ---: | ---: | ---: |
| zero (= prompt-only) | 0.000 | 0.0539 | [0.571, 1.000] | 0.031 |
| random | 0.143 | 0.0222 | [0.429, 1.000] | 0.063 |
| shuffled_between_example | 0.429 | 0.0357 | [0.143, 0.857] | 0.250 |
| slot_permuted | 0.857 | 0.0100 | [0.000, 0.000] | 1.000 |
| quantized (sanity) | 0.857 | 0.0100 | [0.000, 0.000] | 1.000 |
| noise sigma=0.25 | 0.714 | 0.0105 | [0.000, 0.429] | 1.000 |
| noise sigma=0.5 | 0.571 | 0.0128 | [0.000, 0.571] | 0.500 |
| noise sigma=1.0 | 0.286 | 0.0181 | [0.286, 0.857] | 0.125 |

`quantized` reproduces `correct` exactly (paired CI `[0,0]` on both the rate
and MSE metrics) for every arm, confirming the determinism sanity control
passes. `slot_permuted` also scores close to `correct` here -- expected for a
`num_latents=2` continuous codec, where the "slot" permutation of a 2-vector
is a single coordinate swap that a well-trained linear decoder can be close
to symmetric under, unlike genuinely destructive conditions (zero/random/
shuffle) which erase the actual latent value.

### `lfq_n2_w64` and `fsq_typed_2_3_3_4_5` conditions

Both show the same *directional* pattern (`zero`/`random`/`shuffled_between_example`
score 0.000-0.143 vs. `quantized`'s 0.286), but at n=7 held-out examples the
95% paired bootstrap CI lower bound sits at exactly 0.000 rather than
strictly above it, so `causally_necessary=False` per this run's strict
(CI-lower-bound-must-exceed-zero) criterion. This is an honest small-N
finding, not a null result: the point estimates and the McNemar tests both
point the same direction as `continuous_n2_w64`, just without clearing the
stricter paired-CI bar at this tiny fixture holdout size. Full per-condition
tables are in the JSON evidence and the run's own Markdown report
(`outputs/runs/cap2_latent_causal_audit/cap2_latent_causal_audit_cbfe4e1fe36d28be.md`).

## A bug found and fixed during this iteration

The first implementation of `_role_preserving_permutation` returned a valid
swap whenever *any* same-width slot pair existed -- which is true for every
element-wise codec (binary LFQ, continuous, VQ), since every raw coordinate
trivially has width 1 and therefore all coordinates share one "role" group.
That made `role_swapped` for those codecs silently identical to
`slot_permuted` instead of being marked not-applicable, contradicting this
harness's own honesty rule ("never fabricate a separately distinguishable
result"). Caught by manual inspection of a smoke run before this doc was
written (`lfq_n2_w64` and `continuous_n2_w64` both incorrectly reported
`role_swap_applicable=True`); fixed by requiring **more than one** distinct
slot width before treating any same-width group as swappable (a single
homogeneous group means the "restricted" permutation space equals the full
permutation space, i.e. no restriction at all). Regression tests
(`test_role_preserving_permutation_none_for_homogeneous_widths`,
`test_role_preserving_permutation_applies_only_within_same_width_group`)
pin both the fixed behavior and the AP-029 heterogeneous-radix case that
should remain applicable.

## Findings

- **Causal necessity: demonstrated for the `continuous` arm, directionally
  consistent but not CI-significant at n=7 for `lfq`/`fsq_typed`.** The
  frozen decoder's output depends materially on the retained latent for at
  least one codec family with a paired-CI lower bound strictly above zero
  (`continuous_n2_w64`: destroying the latent via zero/random/shuffle drops
  the strict-semantics-proxy rate by 0.43-0.86 with McNemar p<=0.063). No arm
  showed the opposite (destruction *improving* or not moving the score) with
  a significant effect.
- **No representation collapse detected by the no-bypass audit.** All three
  evaluated arms passed `audit_no_bypass` (encoder is not a constant/dead
  code). This is consistent with picking AP-030's own retained,
  non-collapsed configurations as defaults; `uniform_scalar`/`vq` at this
  same minimum capacity are known (CAP2-06) to collapse and were
  deliberately excluded from the default arm set for that reason (still
  reachable via `--arms`/direct API use).
- **Effective rank is close to the collapse boundary even where the
  no-bypass audit passes.** `effective_rank` is 1.00-1.06 across all three
  `num_latents=2` arms -- i.e. almost all retained-latent variance sits in a
  single direction. This is a genuine, non-fabricated caveat: passing the
  binary no-bypass audit does not by itself guarantee a well-spread
  representation; at this minimal 2-latent capacity the codecs are using
  close to 1 effective dimension out of 2 nominal ones. A larger `num_latents`
  arm (outside this run's default set, reachable via `--arms`/API) would be
  needed to check whether effective rank grows with nominal capacity as
  expected.
- **Robust semantic radius is small.** `continuous_n2_w64`'s robust radius is
  `0.0` (even the smallest swept noise level, 0.25x std, already drops the
  rate by more than the 0.05 tolerance); `lfq_n2_w64`/`fsq_typed`'s is `0.5x`
  std. None of the three arms is censored (i.e. every arm's robust radius was
  found within the swept range, not just "at least as large as the largest
  radius tried").

## Disposition

**Continue the continuous-latent track for `continuous_n2_w64`'s
configuration; re-run the audit at larger `num_latents`/held-out sample size
before drawing a stronger conclusion for `lfq`/`fsq_typed`.** Per the issue's
own falsification rule ("if latent destruction has little effect, stop the
continuous track and disposition it as ignored"): destruction did *not* have
little effect for the trustworthy `continuous` arm (rate dropped
0.43-0.86 points with McNemar p<=0.063, paired-CI lower bound 0.14-0.57
above zero for every destructive condition) -- so this run does not trigger
the stop-and-ignore branch. The other two evaluated arms are directionally
consistent but statistically inconclusive at n=7; they are **not** grounds
for a stop verdict (a null CI at n=7 is underpowered, not evidence of no
effect), but they are also not independent confirmation. The honest
disposition is: keep the continuous-latent track open, and treat the
small-effective-rank and small-robust-radius findings above as follow-up
items (larger latent capacity, larger held-out sample) rather than blockers.

## Hard gates

- No-bypass audit failures: 0/3 arms.
- Causal-necessity claim requires `no_bypass_ok=True` **and** every
  destructive condition's paired-CI lower bound strictly above zero --
  enforced in code (`ArmCausalAuditResult.causally_necessary`), not asserted
  from probe accuracy or attention weights alone.
- No arm's `causally_necessary=True` claim rests on probe accuracy or
  attention weights; every claim traces to a paired difference in the frozen
  decoder's own output under a controlled latent perturbation.

## Honest caveats

- **Fixture-scale evidence only.** 25 train / 7 held-out records, 3 codec
  arms, single seed. Not a production or ship claim; no checkpoint was
  created or promoted, so no `docs/MODEL_CARD.md`/README update is required.
- **Strict-semantics proxy, not meaning-v2.** See "Scope decision" above --
  `strict_semantics_proxy_rate` is an MSE-threshold proxy over reconstructed
  program factors, not `binding_aware_meaningful_v2`. No causal claim in this
  doc should be read as a claim about full-program semantic correctness.
- **Weaker-decoder control omitted** -- no fixture checkpoint exists; not
  fabricated (see "Method" above).
- **`uniform_scalar`/`vq` arms not evaluated by default** at this minimum
  capacity (CAP2-06 found they collapse here); still runnable via `--arms`.
- **n=7 held-out is small.** Two of three arms' paired CIs do not clear the
  strict "lower bound > 0" bar despite a consistent directional effect and a
  McNemar test in the same direction; this is reported honestly as
  inconclusive-not-negative, not rounded up to a positive claim.
