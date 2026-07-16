# E214–E216 — generated-schema role judging

Status: **diagnostic only; no checkpoint promoted; ship gates not run**.

## Hypothesis and data repair

The E213 failure could come from contradictory gold data rather than insufficient
training: the corpus contained values that the generated OpenUI schema rejects.
G11 now parses each output with the official parser, walks the resolved AST, and
checks required and present properties against generated schema roles. It handles
`anyOf`, references, arrays, objects, and scalar types without component-specific
branches.

The generalized test uses one generated `FormControl.input` role: an `Input`
reference passes, while a placeholder string and a list fail. Auditing all 496 E177
records rejected 49 unique records. Reason occurrences overlap:

| Generated-schema reason | Occurrences |
| --- | ---: |
| `FormControl.input` value-role mismatch | 16 |
| `Modal.open` value-role mismatch | 11 |
| `Slider.defaultValue` value-role mismatch | 6 |
| `SwitchItem.description` value-role mismatch | 16 |
| `SwitchItem.label` value-role mismatch | 16 |

E214 is an immutable, committed 447-record derivative at
`src/slm_training/resources/train_data/e214_schema_role_judge_v3/`. It has zero
post-judge rejects or build errors, mean quality 0.9648, manifest fingerprint
`d3ad058f…e27ad`, records SHA `47867699…c458b`, and synthesis telemetry SHA
`41c636fe…c22d`. The build used `--source existing`, `--derive-from` E177,
`--synthesizer none`, and disabled new derivatives; it is the filtered matched
control, not newly synthesized evidence.

## Matched train and strict diagnostic

E215 uses the E212 recipe except for E214 data: CPU, 32 steps, batch 4, seed 0,
frozen local SmolLM2-135M context, lexer output, schema and slot context, stratified
compiler-alignment weight 1.0, no DESIGN.md context, and no checkpoint sync. Last
loss is 12.4024 over 18,035 prompt and 5,455 target tokens; wall time is 110.40 s.
Alignment loss falls 75.1520 → 3.5080 and root-child reference loss falls
68.6239 → 0.7868. Checkpoint SHA is `8f700fa7…a3eb37`.

E216 is a one-example smoke diagnostic with compiler tree decode and no
unconstrained fallback:

| Syntax | Meaningful parse | Structure | Component recall | Normalized fidelity | Placeholder validity | Fallback | p50 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 0.0 | 0.3458 | 0.25 | 0.25 | 0.55 | 0 | 11187.37 |

The emitted program is valid but incomplete:

```openui
root = Stack([b1], "column")
b1 = TextContent(":hero.body")
```

The prior invalid `FormControl.input` failure is gone, with zero compiler fallback
and zero constrained dead ends. Meaningful parse correctly remains zero because
component recall is only 0.25. AgentV reports 0/5 checks passed. This separates
deterministic syntax from semantic adequacy and is a negative ship result.
Complete recipes, hashes, and artifact paths are in
[the result JSON](iter-e214-e216-schema-role-judge-20260716.json).

## Next hypothesis

Generate schema-valid positives and nearby invalid counterexamples from every
required and `anyOf` generated-schema path, route both through G11, and test whether
broader reusable value-role supervision improves component recall without weakening
the deterministic syntax guarantee. Do not add component-name, prompt-string, or
observed-output cases.
