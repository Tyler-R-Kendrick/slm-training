# E293 choice-native component plan (2026-07-17)

## Hypothesis and harness repair

E292 isolated component selection as the next semantic bottleneck. E293 tests
the existing grammar-role component-plan objective on the matched choice-codec
arm: root classification plus bound-component counts, with no prompt-specific
cases and no change to grammar legality.

The first preflight (`e293-choice-component-plan-r1`) failed before writing a
checkpoint. `gold_compiler_decisions` routed ChoiceTokenizer through the surface
compiler mapper, which tried to parse individual production choices as complete
OpenUI. The repair replays the choice codec's own pushdown state, classifies its
dependency-first structural stream as bound components followed by the root,
and applies component inventory/plan scores only within the state's legal
candidate set. Structural streams expose an honest `root_or_bound` ambiguity at
decode time; the bias marginalizes the learned root and remaining-count scores
instead of inventing a hidden marker. Focused verification passes 66 tests.

## Recipe audit

The first completed train (`r2`) retained DESIGN.md context and exactly matches
E292's actual input policy and 54,476 prompt-token count. E292's train summary
had incorrectly labeled that default as no-DESIGN; its checkpoint config proves
otherwise. The summary writer now reads the effective model config so future
evidence cannot make that mismatch.

`r2` is therefore the matched E292 comparison. Its bias-off honest evaluation
reaches adversarial meaningful 0.5 versus E292's 0.0, but the direct plan bias
erases the gain. Because both checkpoints were trained with DESIGN context,
this is evidence for plan training only in that regime—not no-DESIGN transfer.

`e293-choice-component-plan-r3` is the policy-correct no-DESIGN follow-up:

- CPU scratch, choice codec, d64/h2, seed 0, batch 2;
- 107 steps / 5,022 target tokens, no DESIGN.md context;
- component-plan loss/decode weights 1.0; no checkpoint sync.

| Metric | Step 1 | Step 107 |
| --- | ---: | ---: |
| component-plan loss | 5.6761 | 3.2616 |
| root accuracy | 0.0000 | 0.5000 |
| bound top-k recall | 0.0000 | 0.5000 |
| bound count MAE | 0.6807 | 0.3907 |

Complete weighted NLL is 7.5550 versus E292's 7.2265; binding NLL is 8.2260
versus 8.0201. Checkpoint SHA-256:
`78b70c81bd16395e22718baa91b50427c205f38136269c6248b85562cdec5308`.

## Frozen honest evaluation and causal ablation

Both evaluations use the matched `r3` checkpoint, scratch context, grammar
constraints, prompt-derived honest slot contracts, no DESIGN.md context, and no
unconstrained fallback. The only difference is component-plan decode weight 1
versus 0.

| Suite | n | Meaningful on/off | Fidelity on/off | Structure on/off | Component recall on/off | Reward on/off |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 0.0 / 0.0 | 0.7222 / 0.3333 | 0.2681 / 0.3500 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| held_out | 5 | 0.0 / 0.0 | 0.5600 / 0.0000 | 0.3328 / 0.2514 | 0.0400 / 0.0000 | 0.1474 / 0.0000 |
| adversarial | 4 | 0.0 / 0.0 | 0.8333 / 0.2500 | 0.2843 / 0.2363 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| ood | 4 | 0.0 / 0.0 | 0.5167 / 0.0000 | 0.3617 / 0.2369 | 0.0625 / 0.0000 | 0.1893 / 0.0000 |
| rico_held | 3 | 0.0 / 0.0 | 0.2500 / 0.0000 | 0.1381 / 0.0901 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |

Decode weight 1 applies the plan 752 times and changes 38 component choices
across 19 examples. It improves most fidelity/structure cells and reduces
frozen gate failures from 17 to 13 versus bias-off, proving the choice-native
integration is active. However, both settings retain meaningful rate 0.0 on all
five suites and AgentV 0/5 with zero execution errors. Parse stays 1.0.

The actual E292-matched DESIGN-context `r2` arm reached adversarial meaningful
0.5 only with decode bias off (AgentV 1/5, 11 gate failures). The no-DESIGN
follow-up does not reproduce it. There is no separately trained no-DESIGN
control checkpoint, so `r3` is a policy transfer check, not an isolated
plan-training comparison.

## Verdict

E293 repairs and validates the choice-native plan harness. Plan training helps
one suite in the actual E292-matched DESIGN-context regime, while direct decode
bias is harmful there. The gain does not survive policy-correct no-DESIGN
training, where bias improves only secondary metrics and AgentV remains 0/5.
Keep the generalized harness, do not promote either checkpoint, and isolate
context grounding with a true no-DESIGN control before tuning this bias further.

Artifacts:

- `outputs/runs/e293-choice-component-plan-r2/` (context calibration)
- `outputs/runs/e293-choice-component-plan-r3/` (matched train)
- `outputs/runs/e293-choice-component-plan-honest-r2/`
- `outputs/runs/e293-choice-component-plan-off-r2/`
- [machine-readable result](choice-component-plan-results-iter-e293-20260717.json)
