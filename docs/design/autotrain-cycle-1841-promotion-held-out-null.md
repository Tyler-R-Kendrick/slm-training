# Autotrain c1841: Lean-gated promotion held-out null

> **Later audit (c1844): invalid as promotion evidence for c1830.** The queue
> projection dropped `mixture_sampling_policy=capacity_aware` before c1838 and
> c1840/c1841. This held-out null remains valid for the executed replacement-
> sampling tail comparison, but it did not test the source capacity-aware tail
> treatment.

**Verdict:** reject the confirmed tail-loss candidate. The repaired frozen
replay completed both smoke and held-out scoreboards, and candidate and matched
control are identical on every guarded quality and decode-work metric. The
candidate does not improve the promotion primary.

| Suite / arm | n | Structure | MPR | Recall | Binder F1 | Fidelity | Reward | AST / canonical | Tokens | Forwards | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke control | 3 | .46417 | .3333 | .2500 | .6333 | .5278 | .8073 | 0 / 0 | 78 | 17 | 2955.32 |
| smoke candidate | 3 | .46417 | .3333 | .2500 | .6333 | .5278 | .8073 | 0 / 0 | 78 | 17 | 2933.64 |
| held-out control | 5 | .30788 | 0 | .1619 | .4371 | .2900 | .7360 | 0 / 0 | 130 | 17 | 3087.23 |
| held-out candidate | 5 | .30788 | 0 | .1619 | .4371 | .2900 | .7360 | 0 / 0 | 130 | 17 | 3160.20 |

The replay hash-verified and reused the two c1840 CPU scratch TwoTower train
stages: 20 steps, batch size 2, one thread, seed 101840, 1,608,962 trainable
parameters, and tail loss weight 0 versus 1 as the only treatment difference.
It reran evaluation only. The control and candidate checkpoint SHAs remain
`1d0aa0ec...75f5` and `6834a759...4cfa`.

Lean freshly proved the locked
`metrics.structural_similarity_monotone` obligation in 1.61 seconds; replay
artifact SHA `e3741613...71e8`. After the allocator repair, each arm received a
73.1-second symmetric ceiling, both scoreboards and AgentV bundles completed,
and neither arm had a runtime or measurement-integrity failure.

This is evidence of narrow, seed-sensitive learning rather than robust OpenUI
learning. Earlier smoke runs showed a tail-loss structure/binder signal, but it
does not transfer to the promotion endpoint: held-out structure delta is
exactly zero, held-out MPR is zero, and exact AST/canonical equality remain
zero. Candidate held-out p50 is also 2.4% slower. Honest ship gates fail for
quality (MPR, recall, AST, canonical), evidence volume (`n=3/5<20`), and missing
adversarial, OOD, and Rico suites.

The next ranked hypothesis is the already-preregistered, size-matched
capacity-aware semantic-exhaustive compiler-alignment arm. It changes semantic
decision supervision rather than model size, decode legality, deterministic
authority, or gates. RL remains locked.

Machine evidence:
[`autotrain-cycle-1841-promotion-held-out-null.json`](autotrain-cycle-1841-promotion-held-out-null.json).
