# E218–E220 — corrected generated-schema admission

Status: **diagnostic only; no checkpoint promoted; ship gates not run**.

## Correction to E214

E214 overfiltered the corpus. OpenUI uses `null` as the positional omission
sentinel for an optional property before a later required property. G11 treated
those legal omissions as typed values, falsely rejecting 27 records: 11 Modal
records and 16 SwitchItem records. The remaining 22 records were genuinely
invalid: 16 had a non-component `FormControl.input`, and six inherited an invalid
Slider enum and scalar `defaultValue` from one stale fixture.

The E215/E216 metrics remain reproducible evidence for the 447-record corpus, but
its data-quality conclusion is superseded. An untracked 474-record E217 preflight
was also invalidated and deleted before training because normalization had not yet
run before G11.

## Generalized repair and E218 data

The corrected path is generated-schema driven:

- optional `null` passes only when the property is absent from generated
  `required` metadata;
- parser semantic errors, enum membership, `anyOf`, references, arrays, objects,
  and scalar roles remain enforced;
- canonical schema normalization now runs before verification for every training
  candidate;
- Slider enum and array shapes come from `library_schema`, and the language-contract
  builder recursively selects a generated `anyOf` branch. A FormControl positive
  now creates and references an `Input` binder.
- canonical form and gallery prompts now match their component structures, so
  future train and test seeds pass the independent judge instead of relying on
  downstream filtering.

E218 is an immutable 480-record derivative. It restores all 33 records missing
from E214: 27 legal optional omissions and six normalized Slider descendants. It
has zero post-build judge rejects, mean quality 0.9654, manifest fingerprint
`f23ac31e…eb0f7`, records SHA `a52b8ac9…a5e7`, and synthesis telemetry SHA
`4dbb4471…0b51`.

## Matched train and strict diagnostic

E219 changes only corrected admission relative to E215: CPU, 32 steps, batch 4,
seed 0, frozen local SmolLM2-135M context, lexer output, schema and slot context,
stratified compiler-alignment weight 1.0, no DESIGN.md context, and no checkpoint
sync. Last loss is 13.2406 over 17,674 prompt and 6,236 target tokens; wall time is
116.24 s. Alignment loss falls 74.2605 → 3.9909. Checkpoint SHA is
`48ba5a7a…9e112`.

E220 is the same one-example smoke diagnostic as E216, with compiler tree decode
and no unconstrained fallback:

| Syntax | Meaningful parse | Structure | Component recall | Normalized fidelity | Placeholder validity | Tokens | Fallback | p50 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 0.0 | 0.3458 | 0.25 | 0.25 | 0.55 | 18 | 0 | 4376.31 |

The valid output is still semantically incomplete:

```openui
root = Stack([b1], "column")
b1 = TextContent(":hero.body")
```

E220 exactly matches E216's semantic metrics, with zero compiler fallback and zero
constrained dead ends. AgentV reports 0/5 checks passed. Corrected admission is
retained because it fixes future data, but it does not improve component coverage
at this training budget. This is a negative ship result. Full recipes and hashes
are in [the result JSON](iter-e218-e220-schema-normalization-20260716.json).

## Next hypothesis

Measure prompt-requested generated-schema role coverage across admitted records,
then derive schema-valid positives for underrepresented requested component-role
sets. Keep a frozen nearby counterexample set for G11, but never use invalid
outputs as training targets.
