# Autotrain c1823: compiler-decision token fresh-seed null

**Verdict:** reject `compiler-decision-token` as a reproducible quality lever.
The same size-matched arm that was positive in the c1822 frozen replay produces
the same outputs as its control at seed 101823: structural similarity `.0575`,
meaningful-program rate `0`, component recall `0`, binder F1 `.48889`, fidelity
`.38889`, reward `0`, and AST/canonical equality `0` in both arms.

| Arm | Params | Loss | Smoke structure | MPR | Recall | Binder F1 | Fidelity | Reward | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,608,962 | 15.62064 | .0575 | 0 | 0 | .48889 | .38889 | 0 | 1765.12 |
| compiler-decision token | 1,608,962 | 21.14347 | .0575 | 0 | 0 | .48889 | .38889 | 0 | 1781.41 |

Both CPU scratch arms trained for 22 steps with batch size 2 and seed 101823.
The only treatment difference is compiler-decision reconstruction weight `0`
versus `1`; trainable parameter count is unchanged. All three documents finish,
all parse, and neither arm reports a decode timeout. The candidate takes 3.79
seconds longer to train and is 0.92% slower at p50, so neither quality nor
efficiency supports the c1822 effect.

This is a three-document fixture screen, not ship evidence. It was selected as
the next fresh-seed instance of the same registered arm, but the driver labeled
it `screening` rather than consuming a champion-queue confirmation. That
governance mismatch does not change the causal null; campaign v121 preserves
screening queue semantics across exact retries and avoids claiming
`candidate_queued` without a real queue entry.

The unchanged ship gates fail: `n=3<20`, MPR, structure, recall, reward, and
AST/canonical equality are below thresholds, while held-out, adversarial, OOD,
and full Rico were not run. Both checkpoints are local no-sync scratch evidence
only. Candidate SHA `de76bb72...2c2e2c`; control SHA
`56af2560...1d37aef`. They are never reusable, promotable, syncable, or
shippable. Lean is `not_applicable:screening`; no theorem or promotion claim is
made.

The next size-matched hypothesis is `compiler-decision-margin`: apply the
existing grammar-oracle alignment loss across every compiler decision family,
stratified by family, so the gold legal branch must outrank legal siblings.
Unlike c1823's extra full-vocabulary CE, it directly optimizes the constrained
choice the production compiler makes and adds no parameters or decode authority.

Machine evidence:
[`autotrain-cycle-1823-compiler-decision-token-fresh-seed-null.json`](autotrain-cycle-1823-compiler-decision-token-fresh-seed-null.json).
