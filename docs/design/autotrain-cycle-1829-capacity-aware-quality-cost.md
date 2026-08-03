# Autotrain c1829: capacity-aware quality/cost tradeoff

**Verdict:** the capacity-aware sampler materially improves effective exposure
and every guarded fixture-quality headline, but the candidate is rejected over
the preregistered latency budget. Preserve the sampling signal; repair legal
continuation cost before confirmation.

| Arm | Params | Effective / draws | Unique | Structure | MPR | Recall | Binder F1 | Fidelity | Reward | Tokens | Forwards | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| replacement margin control | 1,608,962 | 24.82 / 44 | 30 | .41973 | .3333 | .1667 | .6333 | .5278 | .8073 | 54 | 9 | 2838.60 |
| capacity-aware margin | 1,608,962 | 32.27 / 44 | 37 | .53000 | .6667 | .4167 | .8222 | .7222 | .8777 | 75 | 13 | 4680.28 |

The sampler raises effective records 30.0%, unique records 23.3%, and lowers
final training loss from `21.5606` to `17.8148`. Structure rises `.11027`, MPR
`.3333`, recall `.25`, binder F1 `.1889`, fidelity `.1944`, and reward `.0703`.
This is the clearest evidence in the current loop that improving the training
distribution can improve meaningful OpenUI programs without buying capacity.

The gain is not yet acceptable end to end. Candidate output grows `54→75`
tokens, forwards `9→13`, compiler time `6927→11585` ms, and p50 latency
`2839→4680` ms. These increases track richer, longer legal programs, but the
latency gate is still authoritative; useful work is not a waiver. The next arm
keeps capacity-aware sampling and the all-family legal margin, then adds
tail-weighted scaffold supervision to test whether explicit late close-token
training preserves quality while reducing runaway legal continuation.

This remains fixture evidence: `n=3<20`, AST and canonical equality are zero,
and held-out, adversarial, OOD, and full Rico were not run. Both arms use seed
101829, 22 CPU scratch steps, batch size 2, and 1,608,962 trainable parameters.
Candidate SHA `da2863d6...429a9c`; control SHA `e284b175...9185333`.
Checkpoints are local no-sync artifacts and are never reusable, promotable,
syncable, or shippable. Lean is `not_applicable:screening`.

Machine evidence:
[`autotrain-cycle-1829-capacity-aware-quality-cost.json`](autotrain-cycle-1829-capacity-aware-quality-cost.json).
