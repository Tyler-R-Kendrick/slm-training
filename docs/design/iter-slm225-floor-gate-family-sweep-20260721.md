# SLM-225 (NCS0-05): SemanticFloorGateV1 family-count sweep (slm225-floor-gate-family-sweep-20260721)

**Matrix set:** `slm225_floor_gate_family_sweep`
**Version:** `ncs0-05-v1`
**Status:** fixture
**Claim class:** wiring
**Floor threshold:** parse_rate < 0.5
**Runs per family (fixed):** 4
**Sweep grid (n_families):** [2, 4, 8, 16, 32]
**Disposition:** family_count_limited — Signal recovered at n_families=4 (margin=0.188 >= 0.15); SLM-223/SLM-224's no-signal results are consistent with a family-count-limited artifact, not a genuinely absent relationship.

## Hypothesis

SLM-223's SemanticFloorGateV1 'no_signal' disposition, and SLM-224's 'genuinely_no_signal_in_range' finding along the runs-per-family axis, leave the family-count axis untested: the SLM-215 synthetic generator used to always fix families at exactly 2. Holding runs-per-family constant at 4 and sweeping n_families (via the new backward-compatible n_families parameter) will produce a leave-one-family-out balanced accuracy that clears the permutation-null mean by the required 0.15 margin at some larger family count -- i.e. more distinct LOFO folds, not more runs per fold, recovers signal.

## Falsifier

The LOFO-vs-permutation-null margin stays below 0.15 across the full swept grid up to n_families=32 (with runs-per-family held fixed at 4), i.e. more families within the swept range never recovers signal -- suggesting the family-count axis is not the explanation for SLM-223/SLM-224's no-signal results either.

## Honest caveats

- Fixture/wiring evidence only: no trained model, checkpoint promotion, or GPU run.
- Reuses the unmodified SLM-223 gate pipeline at each grid point; the only new code is the backward-compatible n_families parameter threaded through the existing SLM-215 synthetic generator (default unchanged at n_families=2), and this sweep harness itself. No new calibration or statistical method is added.
- Runs-per-family is held fixed at 4 so synthetic_runs scales with n_families (synthetic_runs = n_families * runs_per_family); this isolates the family-count axis from the runs-per-family axis SLM-224 already swept, but the two axes are not fully orthogonal in one combined sweep -- a full 2D grid over both axes independently is out of scope here.
- The synthetic per-matrix signal (0.1 * alpha_z coefficient vs 0.05 noise sd) is a fixture design choice, not a measurement from a real checkpoint; a positive result here shows the gate mechanism *can* detect a real embedded signal given enough families, not that any actual checkpoint atlas has this signal strength.
- No causal conclusion is drawn and no promotion or ship-gate claim is made; this is a diagnostic follow-up to SLM-224's own honest caveat about the untested family-count axis.
- Each grid point's permutation-null baseline is drawn from only 20 random label permutations, and lower n_families points have very few LOFO folds; margins are expected to be noisy from point to point. A single grid point crossing the 0.15 margin is read as an existence result (the mechanism can clear the margin somewhere in range), not as evidence of a clean monotonic family-count effect -- the full swept trend, including any non-monotonic points, should be read together.

## Sweep results

| n_families | runs_per_family | synthetic_runs | n_runs | n_families (actual) | LOFO balanced acc | perm-null mean | margin | disposition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4 | 8 | 8 | 2 | 0.750 | 0.625 | 0.125 | no_signal |
| 4 | 4 | 16 | 16 | 4 | 0.625 | 0.438 | 0.188 | signal_predictive |
| 8 | 4 | 32 | 32 | 8 | 0.594 | 0.500 | 0.094 | no_signal |
| 16 | 4 | 64 | 64 | 16 | 0.500 | 0.500 | 0.000 | no_signal |
| 32 | 4 | 128 | 128 | 32 | 0.752 | 0.523 | 0.229 | signal_predictive |

## No-go for promotion

This report is wiring/fixture evidence only. No checkpoint promotion, GPU train, or ship gate is claimed. It reruns SLM-223's unmodified gate pipeline at increasing n_families values (runs-per-family held fixed) to test whether the family-count axis SLM-224 could not test recovers signal; it does not itself certify `SemanticFloorGateV1` as a promotion or ship gate.
