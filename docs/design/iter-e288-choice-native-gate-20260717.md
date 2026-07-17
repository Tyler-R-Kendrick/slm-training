# E288 — Choice-native constrained decode (2026-07-17)

Frozen-checkpoint eval-only follow-up to the B3 matched five-minute run. Machine
evidence:
[iter-e288-choice-native-gate-20260717.json](iter-e288-choice-native-gate-20260717.json).

## Question

Was the choice arm's all-suite parse 0 caused by undertraining, or by bypassing
the deterministic constrained layer for choice tokens?

## Change

Choice decoding now uses a pushdown state derived from the production codec's
expression/container grammar and the pinned OpenUI library JSON schema. It:

- admits only legal production decisions for the current prefix;
- constrains positional component arguments from generated schema contracts;
- reserves enough remaining positions to complete a schema-valid root;
- rejects unavailable slots and forward references; and
- selects a singleton legal token without running model inference.

There are no component-name conditionals or prompt-specific repairs.

## Frozen-checkpoint result

Weights are byte-identical to B3 (`7cad1431…1c99`). CPU, all 19 committed
remediated records, no DESIGN.md context, 300-second process cap:

| suite | n | parse | meaningful | fidelity | structure | reward | p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | **1.0** | 0 | 0 | 0.3094 | 0 | 6.20 s |
| held_out | 5 | **1.0** | 0 | 0 | 0.2514 | 0 | 6.01 s |
| adversarial | 4 | **1.0** | 0 | 0 | 0.2905 | 0 | 6.15 s |
| ood | 4 | **1.0** | 0 | 0 | 0.2369 | 0 | 6.00 s |
| rico_held | 3 | **1.0** | 0 | 0 | 0.0901 | 0 | 6.07 s |

The same checkpoint moves from 19/19 empty parse failures to 19/19 valid but
trivial layouts. The state forced 8.0–9.6 singleton tokens per example and
recorded zero constrained dead ends. AgentV still passes 0/5 rows.

The first evidence envelope reproduced these scores but mislabeled the loaded
scratch checkpoint as `context_backend=hf` from a CLI construction default.
The evaluator now records effective post-load model settings. The committed
figures point to the regenerated r2 bundle, which reports `scratch`,
`grammar_constrained=true`, and reproduces every quality aggregate exactly.

## Verdict

**The B3 parse 0 was a decoder-path defect, not undertraining.** Structural
adherence is now deterministic across all suites. This does not improve
semantic quality: meaningful-program rate, placeholder fidelity, and reward
remain zero everywhere, so the checkpoint is not promotable and not ship.

The next performance target is the legal-state implementation itself: p50 is
about six seconds because each non-singleton decision currently performs a
full projection and legal candidates are recomputed from cloned states.
Optimization must preserve the exact legal set and singleton bypass.
