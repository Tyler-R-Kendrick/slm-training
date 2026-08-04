# Autotrain c1818: component-edge margin harness failure

**Verdict:** infrastructure-incomplete; replay the exact frozen candidate and
control after repairing `ModelBuildConfig`. This cycle does not measure the
component-edge margin hypothesis.

| Arm | Params | Train | Smoke n | Structure | MPR | Binder F1 | AST / canonical | p50 ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| margin candidate | — | rejected before model construction | 0 | — | — | — | — | — |
| matched control | 1,608,962 | 20 CPU steps | 3 | .404433 | .6667 | .95238 | 0 / 0 | 3797.15 |

The candidate CLI reached `scripts.train_model`, but the fail-closed
`ModelBuildConfig` allowlist had not been extended with `component-edge` even
though the campaign schema, CLI parser, model config, and model objective had
been wired. It raised before model construction, so there is no candidate
checkpoint, scoreboard, parameter count, or causal comparison. The completed
control cannot be interpreted as a treatment result.

The control used seed 101818, batch 2, 20 steps, 1,767 target tokens, and an
explicit no-sync scratch checkpoint (`b97e7424...b502215`). It parses all three
smoke documents, but fails the unchanged evidence-volume, component recall,
AST equality, and canonical equality gates. It is not reusable, promotable,
syncable, or shippable.

Lean is `not_applicable:screening`; no theorem or promotion claim is made. The
next run must replay the exact frozen c1818 recipes after the config-owner fix,
not advance to a new model hypothesis.

Machine evidence:
[`autotrain-cycle-1818-component-edge-margin-harness-failure.json`](autotrain-cycle-1818-component-edge-margin-harness-failure.json).
