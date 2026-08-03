# Autotrain c1838: capacity-aware tail fresh-seed confirmation

> **Later audit (c1844): invalid as confirmation of c1830.** The champion
> transition dropped `mixture_sampling_policy=capacity_aware`, so c1838 actually
> measured tail weight 0 versus 1 under replacement sampling. Its measurements
> remain valid for that executed recipe, but the capacity-aware source winner
> must be re-confirmed with its exact recipe.

**Verdict:** the lever is learning a narrow, reproducible fixture signal, but
the model is not good enough. On seed 101836, tail supervision repeats the
structure and binder gains at matched parameter count and exposure. Protected
quality is mixed and all production ship gates remain red.

| Arm | Params | Effective / draws | Unique | Structure | MPR | Recall | Binder F1 | Fidelity | Reward | Tokens | Forwards | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| matched control | 1,608,962 | 20 / 40 | 26 | .40193 | .3333 | .25 | .8000 | .7500 | .8740 | 74 | 16 | 3552.35 |
| + tail supervision | 1,608,962 | 20 / 40 | 26 | .43723 | .3333 | .25 | .8222 | .7222 | .8657 | 78 | 15 | 3733.23 |

Structure rises `.03530` (8.78%) and binder F1 rises `.02222`; MPR and
component recall hold. Fidelity falls `.02778`, reward falls `.00833`, p50
rises 5.09%, compiler time rises 3.83%, and tokens rise 5.41%, although forwards
fall 6.25%. The candidate's larger final training loss (`41.2863` versus
`37.0940`) also shows why training loss alone cannot establish program quality.

Both arms used CPU scratch TwoTower, 20 steps, batch size 2, one thread,
seed 101836, and 1,608,962 trainable parameters. Exposure was symmetric:
20 effective records, 26 unique of 40 draws, maximum repeat 4. Candidate SHA is
`c445ce878821e8fd8e13279d5b919c25742e9a7d45a85076b90597ead29612d5`;
control SHA is
`472b1d203c90e5a8e1e88c097a325fe085da181bdd11684e39ab8c93ab0f5a9b`.
Both checkpoints are explicit local no-sync artifacts and are never reusable,
promotable, syncable, or shippable.

The blockers are concrete: `n=3<20`, MPR `.3333<.66`, recall `.25<.35`, exact
AST and canonical equality are zero, and held-out, adversarial, OOD, and full
Rico evidence is missing. Lean is `not_applicable:screening`; it becomes
mandatory in the promotion preflight. Campaign v136 repairs the queue ledger so
this retry confirms the original c1830 champion and suppresses its duplicate
c1838 screening row.

Next: run the confirmed candidate through promotion cadence with held-out suites
and Lean/formal preflight; prioritize exact AST/canonical completion and protect
fidelity/reward rather than optimizing fixture structure alone.

Machine evidence:
[`autotrain-cycle-1838-capacity-aware-tail-confirmed.json`](autotrain-cycle-1838-capacity-aware-tail-confirmed.json).
