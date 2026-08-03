# Autotrain c1861: semantic-contrast/compiler-margin frozen replay

**Verdict:** the model shows a narrow matched fixture signal, but it is not
learning high-quality OpenUI programs at ship confidence.

The exact c1860 control and candidate were replayed after compiler-tree
deadline hardening. Both arms now produced complete scoreboards with zero
decode timeouts, so the prior measurement failure was repaired. The candidate
beats the control on this three-record fixture while remaining the same size:
structural similarity `.0575 → .2742`, meaningful-program rate `0 → .333`,
component recall `0 → .333`, and p50 latency `16759 → 3626 ms`. This is useful
evidence that the objective can move a narrow supervised signal, not evidence
of general OpenUI learning.

| Arm | Params | Parse | Struct | MPR | Recall | Binder F1 | Fidelity | Exact AST / canonical | p50 ms | Tokens | Forwards | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control | 1,608,962 | 1.000 | .0575 | 0 | 0 | 0 | 0 | 0 / 0 | 16759 | 765 | 243 | gate reject |
| semantic-contrast + compiler-margin | 1,608,962 | 1.000 | .2742 | .333 | .333 | .633 | .528 | 0 / 0 | 3626 | 126 | 52 | gate reject |

The result is not promotable: smoke `n=3` is below the required `n≥20`, MPR
and component recall remain below thresholds, exact AST/canonical agreement is
zero, and held-out, adversarial, OOD, and RICO suites were not run. The
candidate also reuses the c1860 checkpoint, so this replay adds no new
capacity or checkpoint claim. Lean remains correctly `not_applicable:screening`;
there is no evidence that Lean or training execution blocked this result.

The remaining prevention is supervision and evaluation breadth: the fixture
does not reward exact structure/meaning strongly enough to transfer, and the
smoke-only sample cannot distinguish generalization. Next, keep size fixed,
train against a larger meaning/exact-target corpus, and require the complete
evaluation ladder before promotion.

Machine evidence: [`autotrain-cycle-1861-semantic-contrast-compiler-margin-replay.json`](autotrain-cycle-1861-semantic-contrast-compiler-margin-replay.json).
