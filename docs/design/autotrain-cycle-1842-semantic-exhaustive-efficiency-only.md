# Autotrain c1842: semantic-exhaustive efficiency-only result

**Verdict:** reject semantic-exhaustive compiler alignment as a quality arm.
It substantially reduces decode work, but it slightly lowers structural
similarity and does not move any semantic or exact-program metric.

| Arm | Params | Effective / draws | Unique | Structure | MPR | Recall | Binder F1 | Fidelity | Reward | AST / canonical | Tokens | Forwards | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| semantic-exhaustive off | 1,608,962 | 28.57 / 40 | 32 | .46750 | .3333 | .25 | .6333 | .5278 | .8193 | 0 / 0 | 75 | 12 | 4296.42 |
| semantic-exhaustive on | 1,608,962 | 28.57 / 40 | 32 | .46417 | .3333 | .25 | .6333 | .5278 | .8073 | 0 / 0 | 48 | 8 | 2604.38 |

Both arms used CPU scratch TwoTower, 20 steps, batch size 2, one thread, seed
101842, capacity-aware sampling, and 1,608,962 trainable parameters. Maximum
repeat was 2. Candidate loss was `17.6438` in 14.49 seconds; control loss was
`17.6457` in 11.34 seconds. Candidate SHA is `8eb323b9...3f8f`; control SHA is
`90fa1660...3317`. Both are local explicit no-sync artifacts and are never
reusable, promotable, syncable, or shippable.

The candidate cuts emitted tokens 36%, forwards 33%, compiler time 40%, and
p50 latency 39%, yielding a 65% MPR-per-ms efficiency gain. That is real
learned behavior, but not the requested capability: structure falls `.00333`,
reward falls `.012`, and MPR, component recall, binder F1, fidelity, exact AST,
and canonical equality are unchanged. Honest ship gates also fail on smoke
`n=3<20`, MPR, recall, exact metrics, and all missing production suites. Lean
is `not_applicable:screening`; promotion still requires formal proof.

Campaign v139 preregisters the distinct size-matched successor: retain the
efficient semantic-exhaustive recipe in both arms and add only direct
STRUCT-token reconstruction to treatment. This tests whether scaffold
supervision recovers quality without losing the reduced decode work. Parameters,
legal decoding, deterministic authority, and gates remain unchanged.

Machine evidence:
[`autotrain-cycle-1842-semantic-exhaustive-efficiency-only.json`](autotrain-cycle-1842-semantic-exhaustive-efficiency-only.json).
