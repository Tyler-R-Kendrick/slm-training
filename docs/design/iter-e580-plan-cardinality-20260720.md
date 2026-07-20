# E580 — honest prompt-plan cardinality

Date: 2026-07-20  
Status: honesty correction, quality null; not promotable or ship

E580 preserves repeated component mentions in authored prompt prose while
ignoring generated `Components:` and `Semantic roles:` context lines. The root
closure now waits for each predicted family count, and references only that
many matching generated sections. Candidate legality remains unchanged and
the mechanism stays default-off.

The OOD auth request explicitly asks for a name Input, an email Input, and a
create Button. Its predicted plan is therefore `Button, Input, Input`, not the
de-duplicated `Button, Input` used by E579.

## Matched result

Both arms use clean commit `0b75f907`, E569 checkpoint SHA
`8254fcf7…c6535f73`, CPU, frozen local HF context, OOD `n=4`, honest visible
slot and semantic-role context, choice-codec constrained LTR, 8 generation
steps, 4 attempts, and a 160-token canvas. Both completed under the 170-second
hard cap. Stamps carry eval v16 and TwoTower v17.

| Root weight | Run | meaning-v1 / v2 | structure | recall | reward | AST node / edge | root applications / changes | AgentV |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `e580-e569-cardinality-root0-r1` | 0.25 / 0.00 | 0.1688 | 0.2708 | 0.7345 | 0.2833 / 0.0000 | 0 / 0 | 0/1 |
| 4 | `e580-e569-cardinality-root4-r1` | 0.25 / 0.00 | 0.1688 | 0.2708 | 0.7345 | 0.2833 / 0.0000 | 3 / 0 | 0/1 |

All quality deltas are exactly zero. The auth prediction remains one Input:

```openui
root = Input(":ood.auth.name", ":ood.auth.email", ":ood.auth.create")
```

This means E579's weight-4 AST-edge gain was enabled by an under-counted plan:
it closed a Stack after one Input even though the prompt required two. E580
correctly removes that apparent gain rather than weakening the cardinality
contract.

## Verdict

Retain the repeated-mention cardinality correction. Do not promote or sync a
checkpoint. The next experiment must make component scoring count-aware so the
second required Input can be generated before the verifier-gated root closure.

Machine-readable evidence:
[iter-e580-plan-cardinality-20260720.json](iter-e580-plan-cardinality-20260720.json).
