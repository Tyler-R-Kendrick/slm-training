# Autotrain c1847: semantic-exhaustive/compiler-decision successor

**Verdict:** the candidate improved structure and exact compiler work on this
seed, but it lowered binder quality, fidelity, reward, and training loss did
not improve. The fixture is underpowered (`n=3`) and both arms remain rejected;
this is not a learned production capability.

| Arm | Params | Steps | Loss | Structure | MPR | Binder F1 | Fidelity | Reward | Tokens | Forwards | Compiler ms | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exposure-targeted control | 1,608,962 | 22 | 22.6143 | .2383 | 0 | .9524 | .9167 | .9360 | 130 | 28 | 18,539 | 6,886 |
| + semantic exhaustive | 1,608,962 | 22 | 23.4322 | .3225 | 0 | .8222 | .7222 | .8657 | 60 | 11 | 7,455 | 2,811 |

Both arms used CPU scratch TwoTower, one thread, batch size 2, seed 101847,
exposure-targeted sampling, and 1,608,962 trainable parameters. The treatment
cut decode work and p50 latency, and raised structural similarity by `.0842`,
but it regressed binder F1 by `.1302`, fidelity by `.1944`, and reward by
`.0703`; meaningful, AST, and canonical exact metrics were all zero. Training
loss was higher on treatment (`23.4322` vs `22.6143`). The checkpoints are
local explicit no-sync artifacts and are never reusable, promotable, syncable,
or shippable.

Lean is `not_applicable:screening`; the independent LeverProof audit passed in
the same working session. The c1847 result therefore points to an objective
tradeoff, not a proof or runtime failure. Keep the matched control, do not
recycle this arm, and require a distinct preregistered quality objective plus
full held-out suites before interpreting another result as learning.

Machine evidence:
[`autotrain-cycle-1847-semantic-exhaustive-null.json`](autotrain-cycle-1847-semantic-exhaustive-null.json).
