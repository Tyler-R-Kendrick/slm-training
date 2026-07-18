# E292 choice loss-suite completeness (2026-07-17)

## Question

The B3 choice arm reported an incomplete frozen loss suite because the binding
category had no masks. Was that a training-data omission, or a measurement
defect in the ChoiceTokenizer path?

It was a measurement defect. `ChoiceTokenizer.id_to_kind` uses the generic
`sym`, `bind`, `state`, `struct`, `component`, and `builtin` kinds, while the
loss-suite classifier only understood lexer-native kinds and surface spellings.
Choice binding tokens such as `&0` and `@0` therefore fell through. E292 maps
generic choice kinds into the existing binding/structural categories and adds a
regression test. This changes loss accounting only; it does not change labels,
training, decoding, or gates.

## Runs

The first ladder invocation divided a 5,000-token base budget across the four
capacity cells and produced a 1,263-token calibration checkpoint. It is retained
as evidence, but is not the matched B3 comparison. The corrected invocation used
a 20,000-token base budget, which assigned 5,000 tokens to the selected arm.

| Run | Recipe | Result |
| --- | --- | --- |
| budget calibration | CPU scratch, choice codec, d64/h2, seed 0, batch 2, 26 steps / 1,263 target tokens | complete weighted NLL 18.5150; checkpoint SHA `78334790da71535f1b65edd7073b8c66a21bda35d3da87cfd6d33ab1ece11211`; not matched or promotable |
| matched rerun | CPU scratch, choice codec, d64/h2, seed 0, batch 2, 107 steps / 5,022 target tokens, 35.19 s | complete weighted NLL 7.2265; checkpoint SHA `7cad143139f91369b4780878f634e7fa24434d7ccb8f9813a00f5412f3051c99`; local scratch, no bucket sync |

The matched checkpoint is byte-identical to E288-E291. That is expected and
confirms that the patch repairs measurement rather than model behavior.

Post-run provenance audit in E293 found that the ladder's summary mislabeled
the training recipe as `no-design-md-context`: the serialized model config and
54,476 prompt tokens prove DESIGN context was enabled. The summary writer
treated an unset outer config as false while the model factory correctly
treated it as the TwoTower default (true); E293 fixes that reporting defect.
The standalone E292 ship-gate evaluation below did disable DESIGN context, so
its evaluation policy and scores remain valid. The training checkpoint is a
DESIGN-context scratch diagnostic, not a no-DESIGN model claim.

## Complete frozen loss suite

| Category | NLL | Masked tokens |
| --- | ---: | ---: |
| binding | 8.0201 | 112 |
| structural | 5.6419 | 210 |
| repair | 7.6943 | frozen repair objective |
| schema OOD | 7.0693 | 265 |
| broad | 8.1075 | 385 |
| **weighted** | **7.2265** | all five categories present |

Weighted NLL fell monotonically at the frozen checkpoints:
20.4595 → 14.5726 → 10.6305 → 8.6687 → 7.4410 → 7.2265 at steps
20/40/60/80/100/107. The loss-suite AgentEvals record passed 1/1 through the
pinned AgentV SDK with zero execution errors.

## Frozen honest ship-gate evaluation

The standalone evaluation used scratch context, grammar constraints,
prompt-derived slot-contract constrained decoding, `honest_slot_contract=true`,
no DESIGN.md context, and no unconstrained fallback. This is distinct from the
ladder's inline diagnostic board, which did not enable honest slot-contract
constrained decoding and therefore reported zero fidelity.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Component recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 0.0 | 0.7222 | 0.2958 | 0.00 | 0.0000 |
| held_out | 5 | 1.0 | 0.0 | 0.4800 | 0.2784 | 0.04 | 0.1414 |
| adversarial | 4 | 1.0 | 0.0 | 0.4167 | 0.2330 | 0.00 | 0.0000 |
| ood | 4 | 1.0 | 0.0 | 0.1667 | 0.2731 | 0.00 | 0.0000 |
| rico_held | 3 | 1.0 | 0.0 | 0.0000 | 0.0901 | 0.00 | 0.0000 |

All five AgentV suite rows failed (zero execution errors), and 15 frozen gates
failed. The honest prompt inventory recovers placeholder fidelity on four small
suites, but it does not recover meaningful programs or component selection.
This is fixture-scale diagnostic evidence, not a production readiness claim.

## Verdict

E292 closes the missing-category measurement defect. Binding is now the hardest
targeted denoising category, while the frozen board shows that syntax and
prompt-grounded placeholders are no longer the primary blocker. The next
bounded quality iteration should isolate component/inventory selection
supervision against this complete loss-suite baseline. Do not spend another
iteration on choice-decoder syntax or runtime unless that semantic experiment
reveals a constrained-path defect.

Artifacts:

- `outputs/ladders/e292-choice-loss-suite-complete/`
- `outputs/ladders/e292-choice-loss-suite-complete-r2/`
- `outputs/runs/e292-choice-loss-suite-frozen-r1/`
- [machine-readable result](choice-loss-suite-results-iter-e292-20260717.json)
