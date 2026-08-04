# Autotrain c1862: semantic-contrast/compiler-margin confirmation incomplete

**Verdict:** no confirmation claim; the matched control timed out before its
scoreboard.

c1862 ran the c1861 candidate on a fresh seed. The candidate completed and
reached structural similarity `.4197`, but meaningful-program rate remained
`.333`, component recall fell to `.167`, and exact AST/canonical agreement was
`0`. Its three-record smoke result is below the evidence and quality gates.
The matched control was interrupted before producing a scoreboard, so the
fresh-seed treatment effect is unavailable.

| Arm | Params | Parse | Struct | MPR | Recall | Binder F1 | Fidelity | Exact AST / canonical | p50 ms | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control | 1,608,962 | — | — | — | — | — | — | — | — | incomplete: missing scoreboard |
| confirmation candidate | 1,608,962 | 1.000 | .4197 | .333 | .167 | .633 | .528 | 0 / 0 | 1821 | gate reject |

This does not show that the model generalized or that c1861 confirmed: a
candidate-only score cannot establish attribution. The c1861 compiler deadline
repair remains effective for the completed candidate (`decode_timeout_count=0`),
but the bounded control wall is still the immediate measurement blocker. The
loop must replay the exact c1862 control and candidate before changing the
hypothesis. Lean remains `not_applicable:confirmation` and is not implicated.

Machine evidence: [`autotrain-cycle-1862-semantic-contrast-compiler-margin-confirmation-incomplete.json`](autotrain-cycle-1862-semantic-contrast-compiler-margin-confirmation-incomplete.json).
