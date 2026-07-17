# E249 — exact-event CE plus margin

Date: 2026-07-16
Status: completed; hypothesis falsified; eight ship thresholds failed; checkpoint rejected

E249 applied 30 CPU updates at learning rate `5e-5` to the unchanged E228
parent using the committed `e249_constraint_shadows_v1` corpus. The run used
seed 0, frozen local HF context, strict compiler-tree decoding, all five
committed remediated suites, and unchanged ship gates. The parent SHA is
`7a9be4a665e216d7f7e73883ad74ad972bbf30846896d0c29188d6482f5b093a`;
the resulting checkpoint SHA is
`24285bd447158dd8ba51ebc7d005babd142b223e8f0fca880deefb61d264f32c`.

## Exact-state result

The intervention strongly generalized its lexical objective to all 319 held-out
constraint shadows:

| Metric | Parent | E249 | Delta |
| --- | ---: | ---: | ---: |
| Chosen win | 0.0000 | 0.7649 | +0.7649 |
| Margin win | 0.0000 | 0.6489 | +0.6489 |
| Mean good-minus-bad margin | -3.4834 | 4.2292 | +7.7126 |
| Good probability mass | 0.0420 | 0.1874 | +0.1454 |
| Bad probability mass | 0.4042 | 0.0116 | -0.3925 |
| CE-plus-margin loss | 9.0930 | 2.1644 | -6.9286 |

All events have `constraint_shadow` evidence. The result therefore shows that
E249 learned legal-over-illegal token ranking; it does not show semantic
preference learning.

## Full evaluation

| Suite | n | Syntax | Meaningful | Fidelity | Structure | Reward | E248 structure delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.3333 | 0.5278 | 0.1742 | 0.7653 | -0.2900 |
| held_out | 5 | 1.0000 | 0 | 0.2800 | 0.1088 | 0.6910 | -0.2281 |
| adversarial | 4 | 1.0000 | 0.5000 | 0.5417 | 0.1927 | 0.7695 | -0.2817 |
| ood | 4 | 1.0000 | 0 | 0.2167 | 0.1469 | 0.6720 | -0.2281 |
| rico_held | 3 | 1.0000 | 0.6667 | 0.1250 | 0.0727 | 0.6445 | -0.0901 |

Syntax remained deterministically perfect and meaningful rates did not improve,
while structure and reward regressed on every suite. Eight ship thresholds
failed and AgentV passed 0/5 with zero execution errors. There were zero
compiler fallbacks and zero decode timeouts.

This falsifies the E249 hypothesis. Constraint-shadow supervision is actively
misaligned with semantic quality even though it succeeds on held-out exact
states. Keep lexical legality in deterministic constrained decoding; require
same-state counterfactual semantic evidence before another local preference
train. E250 and E251 should not consume this corpus as quality labels, and
E252-E254 remain fail-closed until counterfactual set-valued evidence exists.

The first evaluation publication attempt could not resolve the pinned AgentV
SDK from the isolated worktree and was invalidated. Re-running with the shared
tracked AgentV runner produced the same checkpoint hash and the canonical
AgentEvals/AgentV bundle. Training trace:
`a46fab75177ec35be1434ea4769fb434`; evaluation trace:
`b89a2fccda220887c3f3923da91db580`; canonical evaluation time:
`2026-07-16T23:08:32.076565+00:00`.

Machine-readable evidence:
[`quality-matrix-v10-e249-results.json`](quality-matrix-v10-e249-results.json).
This was a scratch matrix run, so the rejected checkpoint was not synced to the
HF bucket and was not promoted.
