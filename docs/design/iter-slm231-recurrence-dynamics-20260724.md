# SLM-231 residual recurrence dynamics

Verdict: **expansive_unstable**

Report hash: `06bfd4ecd68f936b4c548b2a7a5fad39bbe56e0bdf76dc7308477635bb2e6913`

This is a bounded CPU diagnostic on the non-promotable SLM-230 scratch checkpoint. It changes no training, generation, promotion, or ship default.

## Evidence boundary

- Checkpoint: `outputs/runs/slm230_bounded_recursive_r4_r2/checkpoints/last.pt` (`1604b2cb9282928fa0969ecbbe7d78c9aa4b9907f74d0d58936bfc298a88b28a`)
- State projection: `{"active_positions": [1], "hidden_size": 32, "includes_z": true, "policy": "active_token_subset", "schema": "RecurrenceStateProjectionV1", "sequence_length": 79}`
- AgentV: `{"durationMs": 30, "executionErrors": 0, "failed": 0, "meanScore": 1, "passed": 4, "total": 4}`
- SLM-138 is API/estimator wiring evidence only.
- SLM-220 verifier subspaces were unavailable for this exact checkpoint state and remain censored.

## Residual-correct spectra

| depth | increment top | composite top | identity control |
| ---: | ---: | ---: | ---: |
| 1 | 0.846658 | 1.777733 | 1.000000 |
| 2 | 0.489466 | 1.446661 | 1.000000 |
| 3 | 0.339243 | 1.312100 | 1.000000 |
| 4 | 0.261271 | 1.242097 | 1.000000 |

## Trajectory product

- Exact product top singular value: **4.724291**
- Maximum FTLE: **0.388179**
- JVP/VJP vs exact maximum error: **0.366000**

## Outcome join and disposition

The exact-state profile joins `held_out_form_01` to SLM-230's **stagnant** depth verdict. Dynamics are diagnostic rather than independent evidence of useful reasoning. The resulting gate verdict is **expansive_unstable**.

RSC2/RSC3 must remain blocked unless later independent groups supply non-vacuous semantic outcomes and uncertainty-qualified dynamics.

## Reproduction

```bash
timeout 170s env PYTHONPATH=src .venv/bin/python -m scripts.run_slm231_recurrence_dynamics --check
```
