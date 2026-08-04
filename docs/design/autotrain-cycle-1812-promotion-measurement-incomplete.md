# Autotrain c1812: promotion measurement is incomplete

**Verdict:** do not promote and do not attribute a matched-arm effect. The
balanced container-close candidate completed the fixture promotion suites, but
the exact control completed zero of three smoke documents and zero of five
held-out documents because every decode hit the typed timeout. The frozen pair
must be replayed once before the champion is disposed.

| Arm | Params | Loss | Smoke complete | Smoke structure | Held-out complete | Held-out structure | Held-out MPR | Held-out p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weight 0 control | 1,608,962 | 17.36850 | 0/3 | — | 0/5 | — | — | — |
| balance .25 + close 1 | 1,608,962 | 20.03540 | 3/3 | .264167 | 5/5 | .20316 | .20 | 2613.54 |

The candidate also records smoke MPR `.3333`, binder F1 `.8222`, component
recall `.25`, and fidelity `.7222`; held-out binder F1 is `.7076`, recall is
`.2286`, and fidelity is `.58`. Those are absolute candidate measurements,
not a treatment delta, because the control quality fields are null. The
candidate misses the fixture-scale absolute gates and both suites are below
the required `n=20`; the run is neither reusable checkpoint nor ship evidence.

Formal integration did run: the promotion preflight proved
`metrics.structural_similarity_monotone` in 1.83 seconds and bound the proof to
SHA-256 `404caa09...ce2b`. This proves the registered metric obligation, not
model quality. Both 20-step CPU scratch checkpoints were written locally with
explicit no-sync: control `c91414a1...4aae`, candidate `7b1d0a90...cb4d`.

The next action is a single exact frozen-manifest replay. If the control
timeout reproduces, classify this as a stable runtime-specific candidate
unblock and reject promotion on missing matched evidence plus low absolute MPR.
If it does not reproduce, compare the complete matched pair without changing
the recipe, suites, timeout, seed, gates, or proof obligation.

Machine evidence:
[`autotrain-cycle-1812-promotion-measurement-incomplete.json`](autotrain-cycle-1812-promotion-measurement-incomplete.json).
