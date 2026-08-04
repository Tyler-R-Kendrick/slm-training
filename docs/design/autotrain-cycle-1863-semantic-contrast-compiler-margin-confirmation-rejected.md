# Autotrain c1863: semantic-contrast/compiler-margin confirmation rejected

**Verdict:** the fresh-seed arm does not confirm a promotable learning win.

Unlike c1862, c1863 produced a complete matched control and candidate. The
candidate improved structural similarity `.1742 → .2899` and binder/fidelity,
but meaningful-program rate stayed `.333`, component recall stayed `.25`, and
exact AST/canonical agreement stayed `0`. Decode cost exploded: p50
`909 → 4875 ms`, tokens `30 → 137`, forwards `5 → 35`, and compiler time
`2293 → 12193 ms`. Both arms are size-matched and timeout-free, but the smoke
subset is only `n=3` and all production suites are absent.

| Arm | Params | Parse | Struct | MPR | Recall | Binder F1 | Fidelity | Exact AST / canonical | p50 ms | Tokens / forwards | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control | 1,608,962 | 1.000 | .1742 | .333 | .250 | .633 | .528 | 0 / 0 | 909 | 30 / 5 | gate reject |
| confirmation | 1,608,962 | 1.000 | .2899 | .333 | .250 | 1.000 | 1.000 | 0 / 0 | 4875 | 137 / 35 | rejected: cost |

This is evidence of a narrow structural/binding response, not high-quality
OpenUI learning. The remaining blockers are weak semantic/exact targets,
tiny smoke evidence, and an unacceptable cost tradeoff—not Lean or model
capacity. The fingerprint is exhausted; the next run must use a distinct,
size-matched objective and broader evaluation.

Machine evidence: [`autotrain-cycle-1863-semantic-contrast-compiler-margin-confirmation-rejected.json`](autotrain-cycle-1863-semantic-contrast-compiler-margin-confirmation-rejected.json).
