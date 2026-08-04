# Autotrain c1839: capacity-aware tail third-seed null

**Verdict:** reject tail supervision as a robust screening lever. Candidate and
matched control are identical on every guarded quality and decode-work metric;
the candidate is 3.8% slower at p50 and 4.2% higher in compiler time.

| Arm | Params | Effective / draws | Unique | Structure | MPR | Recall | Binder F1 | Fidelity | Reward | Tokens | Forwards | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| capacity-aware margin control | 1,608,962 | 27.59 / 40 | 31 | .4600 | .3333 | .25 | .8222 | .7222 | .8777 | 75 | 13 | 4526.87 |
| + tail supervision | 1,608,962 | 27.59 / 40 | 31 | .4600 | .3333 | .25 | .8222 | .7222 | .8777 | 75 | 13 | 4699.26 |

Both arms used CPU scratch TwoTower, 20 steps, batch size 2, one thread, seed
101839, and 1,608,962 trainable parameters. Maximum repeat was 2. Candidate
loss was `16.3938` in 12.95 seconds; control loss was `13.9927` in 12.58
seconds. Candidate SHA is
`117312ac5ea3e0a13a876b0a0d40e62a219f0b0f30c2179e0900cb01d8781b3b`;
control SHA is
`32f10d7055abd7ee58ef37f30b92ad19d63d324407887b992599b5c034411ea7`.
Both checkpoints are explicit local no-sync artifacts and are never reusable,
promotable, syncable, or shippable.

All honest blockers remain: smoke `n=3<20`, MPR `.3333<.66`, recall
`.25<.35`, exact AST and canonical equality are zero, and held-out,
adversarial, OOD, and full Rico evidence are missing. Lean is
`not_applicable:screening`.

The registered screening bank was exhausted after this null. Campaign v137
therefore preregisters a distinct successor objective: capacity-aware
semantic-exhaustive compiler alignment. Its matched control retains the
capacity-aware all-family margin recipe, while treatment changes only semantic
decision coverage. Parameter count, legal decoding, deterministic authority,
and gates remain unchanged. The confirmed champion still enters promotion only
under cadence with mandatory Lean/formal preflight.

Machine evidence:
[`autotrain-cycle-1839-capacity-aware-tail-null.json`](autotrain-cycle-1839-capacity-aware-tail-null.json).
