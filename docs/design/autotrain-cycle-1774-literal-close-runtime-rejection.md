# Autotrain c1774: literal-close runtime unblock reproduced, quality rejected

**Verdict:** the exact frozen replay reproduced a model-runtime difference, but
the candidate is rejected on absolute quality. The tail-supervised candidate
completed all three smoke documents again; the matched control timed out on all
three again. The candidate still produced zero meaningful programs, binder
reference F1, placeholder fidelity, and reward.

| Arm | Frozen artifact | Smoke | Decode work | Decision |
| --- | --- | --- | --- | --- |
| literal-close (`ltr_tail_loss_weight=2`) | c1773 checkpoint and recipe reused exactly | 3/3 complete; parse 1; meaning 0; structure .09640; binder F1/fidelity/reward 0; p50 1,060.83 ms | 12 forwards; 31,111 unique states; 857 witness expansions; 32,115 parser forks | runtime unblock reproduced; absolute quality rejected |
| matched control (tail weight 0) | c1773 checkpoint and recipe reused exactly | 0/3 complete; three typed timeouts; quality unavailable; p50 including incomplete 11,244.87 ms | 342 forwards; 87,454 unique states; 8,526 witness expansions; 93,070 parser forks | model-runtime rejection |

The replay answers the registered runtime question without inventing a quality
comparison. Termination supervision changed this frozen candidate's executable
behavior, but syntax alone is insufficient: meaningful-program, binder,
fidelity, and reward evidence all reject the arm. `literal-close` is retired,
and the loop must select a distinct, non-terminal hypothesis.

Campaign orchestration v81 projects both sides of a reproduced control-only
timeout as `model_runtime`, labels the completing arm as
`runtime unblock; quality reject`, suppresses stale infrastructure repair
advice, and excludes runtime-terminal arms from near-term selection. This also
prevents the already runtime-rejected `component-plan` arm from being proposed
again.

Both arms emitted canonical AgentV bundles. Ship gates fail; this is local CPU
fixture evidence, not a ship result. The replay created no checkpoint: it reused
the exact c1773 candidate and control checkpoints, so no model-card or README
checkpoint entry is required. Lean is `not_applicable:retry_measurement` because
there is no confirmed or promotable champion; promotion formal preflight
remains mandatory when a champion exists.

Machine evidence:
[`autotrain-cycle-1774-literal-close-runtime-rejection.json`](autotrain-cycle-1774-literal-close-runtime-rejection.json).
