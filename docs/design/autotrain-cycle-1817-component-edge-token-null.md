# Autotrain c1817: component-edge token weighting is null

**Verdict:** reject the zero-parameter `component-edge-token` approach at
fixture scale. The candidate and matched control produced identical programs
and identical quality metrics. A 0.25% MPR-per-millisecond change is below the
5% efficiency floor and is timing noise, not a win.

| Arm | Params | Loss | Edge rows at final step | Smoke structure | MPR | Binder F1 | AST / canonical | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weight 0 control | 1,608,962 | 10.64503 | 0 | .174167 | .3333 | .6333 | 0 / 0 | 962.44 |
| edge-token weight 1 | 1,608,962 | 11.44708 | 3 | .174167 | .3333 | .6333 | 0 / 0 | 960.07 |

The treatment objective was active: at step 22 the deterministic compiler
identified three masked non-root component positions with mean CE `11.7851`.
Across earlier batches the count ranged with available component edges. The
candidate changed no parameters, decoder score, grammar domain, or deterministic
authority. It nevertheless changed no smoke prediction after 22 CPU scratch
steps and 2,004 target tokens.

Both arms parse all three documents, with component recall `.25`, placeholder
fidelity `.5278`, and reward `.7653`. They fail the unchanged MPR, structure,
component recall, AST equality, canonical equality, and evidence-volume gates.
This is fixture evidence only; Lean is `not_applicable:screening` and no theorem
or ship claim is made.

Both local checkpoints are explicit no-sync scratch artifacts: control
`5a349a4b...d0c1`, candidate `d68ff571...27fe`. They are not reusable,
promotable, syncable, or shippable.

The next hypothesis should change the source of topology information rather
than increase this scalar. Prioritize topology-error mining or a structured
parent/child/span consistency target with substantially denser, measured
coverage, while retaining the same parameter budget, constrained decoder,
matched control, and honest gates.

Machine evidence:
[`autotrain-cycle-1817-component-edge-token-null.json`](autotrain-cycle-1817-component-edge-token-null.json).
