# Autotrain c1808: container-close alignment is a saturated null

**Verdict:** reject. Both 22-step CPU scratch arms completed at 1,608,962
parameters and seed 101808. Grammar-derived supervision at legal close-versus-
comma decisions ties every smoke quality and decode-work metric, while candidate
training is 3.04x slower.

| Arm | Loss | Train s | Structure | Binder F1 | Recall | MPR | Fidelity | Reward | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weight 0 control | 21.84723 | 3.07 | .3225 | 0 | 0 | 0 | 0 | 0 | 1,900.30 |
| container-close 1 | 24.51254 | 9.35 | .3225 | 0 | 0 | 0 | 0 | 0 | 1,931.99 |

The objective was active, so this is not a missing-data result. It saw 116
eligible gold `)`/`]` rows across 22 steps. Alignment loss fell from `21.7435`
and margin-violation rate `.60` on step one to near zero and zero violations by
step two. The matched control already selected the legal closes: both arms emit
36 tokens with seven neural forwards, 61,368 completion states, and 64,255
parser forks. Candidate compiler time is `4510.16` versus `4419.57` ms.

The result closes the standalone close-loss approach, not the quality goal. The
next size-matched interaction arm combines c1807's typed-family balance quality
objective with this close loss. That directly tests whether close supervision
can retain c1807's structure/binder gains while preventing its 201-token legal
continuation runaway. No parameter growth, decode heuristic, legality change,
or unconstrained fallback is introduced.

The local explicit-no-sync checkpoint SHA-256 values are `d2e8deb7...ccb9`
(control) and `d2fbce14...9d4f` (candidate). Neither is reusable, promotable,
synced, or ship evidence. AgentV bundles are complete, fixture gates fail, and
Lean is `not_applicable:screening`.

Machine evidence:
[`autotrain-cycle-1808-container-close-null.json`](autotrain-cycle-1808-container-close-null.json).
