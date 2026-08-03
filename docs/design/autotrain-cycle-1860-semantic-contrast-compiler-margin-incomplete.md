# Autotrain c1860: semantic-contrast/compiler-margin measurement incomplete

**Verdict:** no learning claim; the matched comparison is incomplete and the
candidate fails the honest smoke gates.

The candidate trained and decoded through the constrained production path, but
the outer hard cap interrupted the matched control during evaluation before it
produced a scoreboard. Therefore no treatment effect can be attributed. The
candidate's standalone smoke result is also below every meaningful quality
threshold that matters here: MPR `.333`, structural similarity `.274`, exact
AST/canonical agreement `0`, and smoke `n=3` instead of the required `n≥20`.

| Arm | Params | Loss | Parse | Struct | MPR | Recall | Binder F1 | Fidelity | p50 ms | Tokens | Forwards | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control | 1,608,962 | 15.0883 | — | — | — | — | — | — | — | — | — | incomplete: evaluation interrupted |
| semantic-contrast + compiler-margin | 1,608,962 | 17.9803 | 1.000 | .2742 | .333 | .333 | .633 | .528 | 3606 | 126 | 52 | gate reject |

The candidate added no parameters, so capacity growth is not the blocker. The
observed blockers are (1) incomplete control measurement, (2) too-small smoke
evidence, and (3) insufficient semantic supervision/transfer: the new losses
increased training loss and decode work without producing exact AST agreement.
Lean is not implicated because this was a screening arm (`formal:
not_applicable:screening`); Lean promotion gates remain locked.

Machine evidence: [`autotrain-cycle-1860-semantic-contrast-compiler-margin-incomplete.json`](autotrain-cycle-1860-semantic-contrast-compiler-margin-incomplete.json).

Next run: replay the exact frozen candidate and control to complete the
comparison before steering to another hypothesis. Do not promote, sync, or
serve either scratch checkpoint.
