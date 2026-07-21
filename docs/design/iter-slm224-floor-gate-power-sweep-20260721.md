# SLM-224 (NCS0-04): SemanticFloorGateV1 power sweep (slm224-floor-gate-power-sweep-20260721)

**Matrix set:** `slm224_floor_gate_power_sweep`
**Version:** `ncs0-04-v1`
**Status:** fixture
**Claim class:** wiring
**Floor threshold:** parse_rate < 0.5
**Sweep grid (synthetic_runs):** [4, 8, 16, 32, 64, 128]
**Disposition:** genuinely_no_signal_in_range — No swept point (up to synthetic_runs=128) reached the 0.15 margin; within this range, more fixture size did not recover signal for this LOFO/permutation-null protocol.

## Hypothesis

SLM-223's SemanticFloorGateV1 'no_signal' disposition on the default 4-run/2-family fixture is a statistical-power artifact, not evidence that the underlying alpha_z-vs-parse_rate relationship fails to generalize across families: sweeping the same SLM-215 synthetic generator (which bakes in a real per-matrix parse_rate = 0.4 + 0.1*alpha_z + noise relationship) to larger synthetic_runs values will produce a leave-one-family-out balanced accuracy that clears the permutation-null mean by the required 0.15 margin at some larger sample size.

## Falsifier

The LOFO-vs-permutation-null margin stays below 0.15 across the full swept grid up to 128 synthetic runs, i.e. more fixture size within the swept range never recovers signal -- suggesting SLM-223's no-signal result was not (only) a power problem.

## Honest caveats

- Fixture/wiring evidence only: no trained model, checkpoint promotion, or GPU run.
- Reuses the unmodified SLM-223 gate pipeline and the unmodified SLM-215 synthetic generator at each grid point; no new calibration or statistical method is added.
- The synthetic generator always splits runs into exactly 2 families (run_idx % 2), so scaling synthetic_runs increases runs-per-family, not the number of families; this sweep cannot speak to whether more *families* would help, only whether more *runs per family* helps.
- The synthetic per-matrix signal (0.1 * alpha_z coefficient vs 0.05 noise sd) is a fixture design choice, not a measurement from a real checkpoint; a positive result here shows the gate mechanism *can* detect a real embedded signal given enough samples, not that any actual checkpoint atlas has this signal strength.
- No causal conclusion is drawn and no promotion or ship-gate claim is made; this is a diagnostic follow-up to SLM-223's own honest caveats.

## Sweep results

| synthetic_runs | n_runs | n_families | LOFO balanced acc | perm-null mean | margin | disposition |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 4 | 2 | 0.500 | 0.500 | 0.000 | no_signal |
| 8 | 8 | 2 | 0.750 | 0.625 | 0.125 | no_signal |
| 16 | 16 | 2 | 0.375 | 0.500 | -0.125 | no_signal |
| 32 | 32 | 2 | 0.562 | 0.500 | 0.062 | no_signal |
| 64 | 64 | 2 | 0.531 | 0.516 | 0.016 | no_signal |
| 128 | 128 | 2 | 0.236 | 0.482 | -0.246 | no_signal |

## No-go for promotion

This report is wiring/fixture evidence only. No checkpoint promotion, GPU train, or ship gate is claimed. It reruns SLM-223's unmodified gate pipeline at larger synthetic fixture sizes to test whether SLM-223's no-signal result was a statistical-power artifact of a 4-run fixture; it does not itself certify `SemanticFloorGateV1` as a promotion or ship gate.
