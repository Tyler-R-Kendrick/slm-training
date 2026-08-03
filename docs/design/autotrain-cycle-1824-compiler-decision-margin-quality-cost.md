# Autotrain c1824: compiler-decision margin quality/cost tradeoff

**Verdict:** retain the all-family compiler-decision margin as a quality signal,
but reject this arm for the champion queue because its end-to-end latency is
4.01× the matched control. The candidate is substantially better on every
measured non-equality quality metric, yet the gain is bought with longer output
and 3.75× as many neural forwards.

| Arm | Params | Structure | MPR | Recall | Binder F1 | Fidelity | Reward | Tokens | Forwards | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,608,962 | .13527 | .3333 | .1667 | .6333 | .5278 | .7653 | 21 | 4 | 973.41 |
| compiler-decision margin | 1,608,962 | .48110 | .6667 | .4167 | .8222 | .7222 | .8597 | 61 | 15 | 3901.53 |

The candidate improves structure by `.34583`, MPR by `.3333`, component recall
by `.25`, binder F1 by `.18889`, fidelity by `.19444`, and reward by `.09433`.
It clears the fixture thresholds for MPR, structure, recall, fidelity, reward,
and parse. However, p50 rises by 300.8%; emitted tokens rise `21→61`, forwards
`4→15`, compiler prefill tokens `2816→8448`, and canvas tokens `1024→3840`.
Training also takes `11.22` seconds versus `2.68` seconds. The unchanged
quality-primary latency budget therefore correctly rejects the candidate rather
than letting a larger generated program mint a free quality win.

This remains a three-document fixture result, not ship evidence. AST and
canonical equality stay at zero, `n=3<20`, and held-out, adversarial, OOD, and
full Rico were not run. Both arms use seed 101824, 20 CPU scratch steps, batch
size 2, and exactly 1,608,962 trainable parameters. Candidate SHA
`524a7938...be6605`; control SHA `01f3d6ab...1158a5`. The checkpoints are local
no-sync evidence only and are never reusable, promotable, syncable, or
shippable. Lean is `not_applicable:screening`; no theorem or promotion claim is
made.

The next experiment preserves the all-family margin recipe in both arms and
changes only `grammar_completion_bounds`. That isolates whether deterministic
completion can reduce forwards and latency while retaining the observed quality,
without changing model size, learned scores, grammar authority, or legal domains.
Campaign v122 also promotes emitted-token, forward, prefill, and canvas counters
into the terminal result matrix so future cost rejections explain themselves.

Machine evidence:
[`autotrain-cycle-1824-compiler-decision-margin-quality-cost.json`](autotrain-cycle-1824-compiler-decision-margin-quality-cost.json).
