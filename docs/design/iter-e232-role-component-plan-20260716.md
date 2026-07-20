# E232 — grammar-role component plan

Date: 2026-07-16
Status: completed; checkpoint rejected; not promotable or ship

E232 replaces E231's flat component set with targets read from compiler decision
kinds: one `component_root` class and `component_bound` multiplicities. Root CE
and a positive/negative-balanced Poisson count objective train a pooled prompt
plan. Decode applies root logits or remaining-count logits only to component
candidates already admitted by the compiler forest. It cannot alter legality and
contains no prompt, output, component-arrangement, or literal cases.

Immediately before training, the isolated branch fetched and rebased onto
`origin/main` twice, was zero commits behind, and was clean. Ruff and 83 focused
tests passed after reconciliation.

The matched E231 recipe used the published 126-row
`e230_diverse_judged_roots_v2` corpus, CPU, 32 steps, batch 4, learning rate
0.0003, seed 0, frozen local SmolLM2-135M, lexer output, compiler CE/margin 1.0,
schema and honest slot context, no DESIGN context, capacity-aware sampling, and
no checkpoint sync. In 154.70 s, 128 draws covered 81 rows including 30 RICO and
25 human-curated draws. Root accuracy rose 0 → 1.0, bound top-k recall 0 →
0.7083, and count MAE fell 0.6974 → 0.4355. Training trace:
`902e7e50a085e33630b398a0ffaebd11`; checkpoint SHA:
`da42b9eab705c2b4659daea7677dca6fb35e5efc63dafb8787b1d0e6a4be208e`.

The run's Poisson telemetry omitted the target-only full term, so its final
bound loss is numerically negative although its gradients are valid. Future runs
include that constant and report a nonnegative value; checkpoint gradients and
the E232 evaluation are unaffected.

| Suite | n | syntax | meaningful | structure | component recall | fidelity | reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.3333 | 0.4642 | 0.2500 | 0.5278 | 0.8073 |
| held_out | 5 | 1.0000 | 0.0000 | 0.3335 | 0.1567 | 0.1800 | 0.5732 |
| adversarial | 4 | 1.0000 | 0.5000 | 0.4744 | 0.4583 | 0.5417 | 0.8115 |
| ood | 4 | 1.0000 | 0.0000 | 0.3469 | 0.1458 | 0.2083 | 0.5493 |
| rico_held | 3 | 1.0000 | 0.6667 | 0.1628 | 0.4444 | 0.1250 | 0.6865 |

Strict five-suite compiler-tree evaluation keeps syntax 1.0 and fails the same
four thresholds as the E228/E230 frontier. AgentV passes 1/5 with zero execution
errors. Trace: `6c3216f0886aaaebb923d2ce099fd3c3`.

The plan-off ablation reduces adversarial recall 0.4583 → 0.3750, fidelity
0.5417 → 0.4167, and reward 0.8115 → 0.6118; other suites are unchanged. At
weight 1, only 3 of 137 planner applications change a choice. Weight 4 increases
changes to 19 but leaves every aggregate metric identical to weight 1. Traces:
off `b08e82ba2dff83cdfe811f70956c5844`; weight 4
`ba2639571da1a8799e5f08277c3ffb56`.

Retain the generalized role/count mechanism, telemetry, and causal override,
but reject the checkpoint. The next semantic lever must model component
relationships or hierarchy, not merely stronger planner calibration.

Machine-readable evidence:
[iter-e232-role-component-plan-20260716.json](iter-e232-role-component-plan-20260716.json).
