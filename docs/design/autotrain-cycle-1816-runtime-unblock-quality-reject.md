# Autotrain c1816: runtime unblock reproduced, quality rejected

**Verdict:** retire the balanced container-close promotion candidate. The exact
frozen replay completed end to end with a fresh Lean proof. The candidate again
completed every fixture document while the size-matched control again timed out
on every document. This establishes a reproducible treatment-specific runtime
unblock, but not a promotable quality result.

| Arm | Params | Smoke complete | Smoke structure | Held-out complete | Held-out structure | Held-out MPR | Held-out p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weight 0 control | 1,608,962 | 0/3 | — | 0/5 | — | — | — |
| balance .25 + close 1 | 1,608,962 | 3/3 | .264167 | 5/5 | .20316 | .20 | 2639.63 |

The candidate also records smoke binder F1 `.8222`, component recall `.25`,
placeholder fidelity `.7222`, and MPR `.3333`; held-out binder F1 is `.7076`,
component recall `.2286`, and placeholder fidelity `.58`. These values reproduce
c1812. They remain below the unchanged fixture gates, and both suites remain
below the required `n=20`. The run is fixture evidence only and does not support
a ship or checkpoint-promotion claim.

The control's eight typed decode timeouts reproduced for the third measured
attempt. Because the candidate has a current complete scoreboard, the runtime
unblock classification is now valid. It still cannot supply a numerical matched
quality delta: the control quality fields are unavailable. The candidate is
rejected on its low absolute quality and the comparison arm is retired.

Formal integration passed. The current campaign proved
`metrics.structural_similarity_monotone` in 1.45 seconds, binding current
obligation `formal-7af3e9686d74ba1e` to artifact `374d981a...23aa`. This proves
the registered metric implication, not model quality.

No checkpoint was created or promoted; both arms reused the content-bound c1812
checkpoints. The next priority is to preregister and implement a genuinely
distinct, zero-parameter or otherwise size-matched quality objective. It should
target exact/canonical AST agreement or component topology directly rather than
another completion-only loss whose principal effect is avoiding the decode wall.
The matched baseline, Lean preflight, honest gates, and parameter accounting stay
unchanged.

Machine evidence:
[`autotrain-cycle-1816-runtime-unblock-quality-reject.json`](autotrain-cycle-1816-runtime-unblock-quality-reject.json).
