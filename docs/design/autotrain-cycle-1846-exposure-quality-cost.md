# Autotrain c1846: exposure-targeted quality/cost tradeoff

**Verdict:** rare-decision exposure is a real quality-bearing signal, but this
configuration is not promotable. It doubles MPR and substantially improves
recall, binder F1, fidelity, and reward, while lowering structure and more than
doubling decode work and latency.

| Arm | Params | Effective / draws | Unique | Structure | MPR | Recall | Binder F1 | Fidelity | Reward | AST / canonical | Tokens | Forwards | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| capacity-aware | 1,608,962 | 30.41 / 42 | 35 | .46417 | .3333 | .2500 | .6333 | .5278 | .8073 | 0 / 0 | 54 | 10 | 2669.33 |
| exposure-targeted | 1,608,962 | 19.60 / 42 | 26 | .41240 | .6667 | .4167 | 1.0000 | 1.0000 | .9610 | 0 / 0 | 117 | 25 | 6358.19 |

Both arms used CPU scratch TwoTower, 21 steps, batch size 2, one thread, seed
101846, all-family compiler alignment, and 1,608,962 trainable parameters. The
candidate changed only sampling policy. Candidate loss was `20.3384` in 14.86
seconds; control loss was `20.8952` in 12.46 seconds. Candidate SHA is
`a8770114...e8ecb`; control SHA is `f6961900...2bdca`. Both are local explicit
no-sync artifacts and are never reusable, promotable, syncable, or shippable.

The candidate raises MPR `.3333`, component recall `.1667`, binder F1 `.3667`,
fidelity `.4722`, and reward `.1537`, but structure falls `.05177`, tokens rise
117%, forwards rise 150%, and p50 rises 138%. Exact AST and canonical equality
remain zero. Honest ship gates fail on `n=3<20`, exact metrics, and missing
production suites. Lean is `not_applicable:screening`.

The result supports exposure/objective alignment as the leading model-side
direction, not capacity growth. The size-matched successor keeps
`exposure_targeted` in both arms and adds semantic-exhaustive compiler
supervision only to treatment. Prior c1842 evidence showed that mechanism can
cut tokens and forwards; the combination tests whether it can retain this
semantic gain while controlling runaway legal continuation.

Machine evidence:
[`autotrain-cycle-1846-exposure-quality-cost.json`](autotrain-cycle-1846-exposure-quality-cost.json).
