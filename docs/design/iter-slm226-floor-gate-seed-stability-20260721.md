# SLM-226 (NCS0-06): SemanticFloorGateV1 permutation-null seed-stability sweep (slm226-floor-gate-seed-stability-20260721)

**Matrix set:** `slm226_floor_gate_seed_stability`
**Version:** `ncs0-06-v1`
**Status:** fixture
**Claim class:** wiring
**Floor threshold:** parse_rate < 0.5
**Runs per family (fixed):** 4
**Sweep grid (n_families):** [2, 4, 8, 16, 32]
**Permutation-null seeds swept:** [11, 3, 7, 19, 23, 29, 31, 37]
**SLM-225 dip grid points under test:** [8, 16]
**Disposition:** dip_stable_under_permutation_resampling — Every dip point (n_families=8, n_families=16) stayed below the 0.15 margin across all 8 swept permutation-null seeds -- the dip is not explained by permutation-null sampling noise alone (synthetic-data-seed noise remains untested).

## Hypothesis

SLM-225's non-monotonic dip at n_families=8 (margin=0.094) and n_families=16 (margin=0.000) -- both below the 0.15 signal margin, sandwiched between signal-clearing points at n_families=4 (0.188) and n_families=32 (0.229) -- is explained by permutation-null sampling noise: each grid point's null baseline is drawn from only 20 permutations at a single fixed seed. Rerunning the unmodified SLM-223 gate pipeline at each SLM-225 grid point across multiple permutation-null seeds (synthetic data held fixed) will show the margin at n_families=8 and/or 16 crossing 0.15 for at least one alternate seed, i.e. the dip is a resampling artifact, not a stable non-monotonic effect.

## Falsifier

The margin at n_families=8 and n_families=16 stays below the 0.15 signal threshold for every swept permutation-null seed -- i.e. the dip is stable under permutation-null resampling and is not explained by null-baseline sampling noise alone (though a residual synthetic-data-seed noise source remains untested; see honest caveats).

## Honest caveats

- Fixture/wiring evidence only: no trained model, checkpoint promotion, or GPU run.
- Reuses the unmodified SLM-223 gate pipeline and SLM-225's grid at each point; the only new code is the backward-compatible permutation_seed parameter threaded through SLM-223's run_semantic_floor_gate_fixture (default unchanged at seed=11), and this sweep harness itself. No new calibration or statistical method is added.
- This sweep varies only the permutation-null seed; the synthetic per-run data itself is generated with a single hardcoded seed (42, unchanged) in SLM-215's generator. A residual noise source -- sensitivity of the real LOFO balanced accuracy to the synthetic-data seed -- is NOT tested here and remains an open axis; a stable-under-permutation-resampling result at n_families=8/16 does not rule out that a different synthetic-data seed would shift real_balanced_accuracy itself and produce a different dip pattern.
- Only 8 permutation-null seeds are swept per grid point (each itself averaging 20 permutation draws); this bounds, but does not eliminate, sampling uncertainty in the seed-to-seed margin statistics reported here.
- No causal conclusion is drawn and no promotion or ship-gate claim is made; this is a diagnostic follow-up to SLM-225's own honest caveat about permutation-null sampling noise as a candidate explanation for its non-monotonic sweep.
- The real_balanced_accuracy at each n_families point is identical across all swept seeds by construction (permutation_seed only affects the null baseline), so this harness cannot itself distinguish 'genuine non-monotonicity in the real signal' from 'non-monotonicity in the real signal driven by the fixed synthetic-data seed' -- it can only test whether the null-baseline axis alone explains the dip.
- In the committed fixture run, the permutation-null mean (and hence the margin) was empirically identical across all 8 swept seeds (margin_std=0.000) at n_families=2, 8, and 16, and only showed seed-to-seed variance at n_families=32; read literally, this suggests the label-permutation space at smaller run counts is constrained enough that 20 draws already converge to the same mean regardless of RNG seed for this fixture's class balance -- itself a fixture-scale artifact, not evidence that a real checkpoint atlas would behave identically.

## Sweep results

| n_families | real LOFO bal. acc | margin mean | margin std | margin min | margin max | seeds crossing 0.15 / total | SLM-225 disposition (this run) | stability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.750 | 0.125 | 0.000 | 0.125 | 0.125 | 0 / 8 | no_signal | stable_no_signal |
| 4 | 0.625 | 0.188 | 0.000 | 0.188 | 0.188 | 8 / 8 | signal_predictive | stable_signal |
| 8 | 0.594 | 0.094 | 0.000 | 0.094 | 0.094 | 0 / 8 | no_signal | stable_no_signal |
| 16 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0 / 8 | no_signal | stable_no_signal |
| 32 | 0.752 | 0.264 | 0.034 | 0.209 | 0.311 | 8 / 8 | signal_predictive | stable_signal |

## No-go for promotion

This report is wiring/fixture evidence only. No checkpoint promotion, GPU train, or ship gate is claimed. It reruns SLM-223's unmodified gate pipeline at SLM-225's grid points across multiple permutation-null seeds to test whether SLM-225's non-monotonic n_families=8/16 dip is explained by permutation-null sampling noise; it does not itself certify `SemanticFloorGateV1` as a promotion or ship gate, and does not test synthetic-data-seed noise (a separate, still-open axis).
