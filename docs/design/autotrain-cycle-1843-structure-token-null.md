# Autotrain c1843: structure-token quality null

**Verdict:** reject direct STRUCT-token reconstruction as the quality recovery
for the efficient semantic-exhaustive arm. The treatment is an exact quality
and decode-work null; its small latency delta is below the campaign noise floor.

| Arm | Params | Effective / draws | Unique | Structure | MPR | Recall | Binder F1 | Fidelity | Reward | AST / canonical | Tokens | Forwards | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| structure-token weight 0 | 1,608,962 | 31.5 / 42 | 36 | .46417 | .3333 | .25 | .6333 | .5278 | .8073 | 0 / 0 | 48 | 8 | 2606.55 |
| structure-token weight 1 | 1,608,962 | 31.5 / 42 | 36 | .46417 | .3333 | .25 | .6333 | .5278 | .8073 | 0 / 0 | 48 | 8 | 2550.83 |

Both arms used CPU scratch TwoTower, 21 steps, batch size 2, one thread, seed
101843, capacity-aware sampling, semantic-exhaustive all-family compiler
alignment, and 1,608,962 trainable parameters. Maximum repeat was 3. Candidate
loss was `16.8497` in 18.43 seconds; control loss was `13.8854` in 18.56
seconds. Candidate SHA is `aa032e93...00c8`; control SHA is
`4558d721...d3d1`. Both are local explicit no-sync artifacts and are never
reusable, promotable, syncable, or shippable.

The candidate changes no guarded quality metric and no decode-work counter.
Its 2.18% p50 advantage is below the preregistered 5% minimum effect and is
therefore noise, not a performance win. Honest ship gates fail on smoke
`n=3<20`, MPR, component recall, exact AST/canonical metrics, and all missing
production suites. Lean is `not_applicable:screening`; promotion still requires
formal proof.

This closes another loss-only approach. The distinct size-matched successor
changes online exposure instead: compare default-derived rare-action
`exposure_targeted` sampling against `capacity_aware` sampling while retaining
the same all-family compiler loss, model, and decode authority. This directly
tests whether rare decision coverage, rather than another auxiliary loss, is
the current limitation.

Machine evidence:
[`autotrain-cycle-1843-structure-token-null.json`](autotrain-cycle-1843-structure-token-null.json).
