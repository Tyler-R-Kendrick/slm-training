# CAP2-01: strict K-ary bottleneck phase-boundary fixture matrix

**Status:** wiring evidence / fixture CPU run
**Run id:** `cap2-01/2fdaa15e80b7af31`
**Date:** 2026-07-17
**State count:** M=41 synthetic fixture (CAP0-03 robust-code target)

## Honest caveat

This is a fixture CPU run. It verifies the mathematical capacity bound
`K**d >= M` and the no-bypass wiring invariant for the new CAP2-01 harness.
It does **not** train a production model, does **not** run on GPU, and does
**not** make a ship-quality claim.

## Recipe

```bash
python -m scripts.run_cap2_bottleneck \
  --out-dir outputs/runs/cap2_bottleneck \
  --seeds 0
```

- Device: CPU
- State source: verified M=41 synthetic fixture
- Seeds: 0
- Deterministic arms: injective assignment and robust-code controls
- Learned arms: `KaryBottleneck` with straight-through training
- Output JSON: [`cap2-bottleneck-results.json`](./cap2-bottleneck-results.json)

## Results

| arm | K | d | capacity | states | mode | exact_rate | occupied | collisions | leakage | notes |
| --- | - | - | -------- | ------ | ---- | ---------- | -------- | ---------- | ------- | ----- |
| b2d5 | 2 | 5 | 32 | 41 | injective | 0.7805 | 32 | 9 | False | capacity 32 < states 41; 9 collisions |
| b2d6 | 2 | 6 | 64 | 41 | injective | 1.0000 | 41 | 0 | False | injective assignment reconstructed all states |
| t3d3 | 3 | 3 | 27 | 41 | injective | 0.6585 | 27 | 14 | False | capacity 27 < states 41; 14 collisions |
| t3d4 | 3 | 4 | 81 | 41 | injective | 1.0000 | 41 | 0 | False | injective assignment reconstructed all states |
| k2d6 | 2 | 6 | 64 | 41 | injective | 1.0000 | 41 | 0 | False | equal-capacity geometry arm |
| k4d3 | 4 | 3 | 64 | 41 | injective | 1.0000 | 41 | 0 | False | equal-capacity geometry arm |
| k8d2 | 8 | 2 | 64 | 41 | injective | 1.0000 | 41 | 0 | False | equal-capacity geometry arm |
| k7d4_robust | 7 | 4 | 2401 | 41 | robust | 1.0000 | 41 | 0 | False | MDS `[4,2,3]_7` corrects one substitution |
| k3d7_robust | 3 | 7 | 2187 | 41 | robust | 1.0000 | 41 | 0 | False | shortened `[7,4,3]_3` corrects one substitution |
| direct_one_hot | 41 | 1 | 41 | 41 | direct | 1.0000 | 41 | 0 | False | no-bottleneck control |
| learned_b2d6 | 2 | 6 | 64 | 41 | learned | 1.0000 | 41 | 0 | False | learned bottleneck reaches perfect reconstruction |
| learned_t3d4 | 3 | 4 | 81 | 41 | learned | 1.0000 | 41 | 0 | False | learned bottleneck reaches perfect reconstruction |

## Hard gates

- Below-capacity arms: 2 (`b2d5`, `t3d3`)
- Leakage violations (below-capacity exact reconstruction == 1.0): **0**
- Result: **PASS**

## No-bypass audit

`KaryBottleneck.audit_no_bypass` recomputes the decoder output from the integer
code alone and verifies it equals the full forward pass.  For the
`semantic_trace` mode it additionally zeros the upstream encoder latent and
confirms the code changes.  Tests cover both paths.

## Files added

- `src/slm_training/models/kary_bottleneck.py`
- `src/slm_training/harnesses/experiments/cap2_bottleneck.py`
- `scripts/run_cap2_bottleneck.py`
- `tests/test_models/test_kary_bottleneck.py`
- `tests/test_harnesses/experiments/test_cap2_bottleneck.py`

`scripts/run_scaling_ladder.py` was extended with `--family discrete-bottleneck`
to dispatch to the same harness.
