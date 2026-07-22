# E926-E927: strict direct component types

E926-E927 tighten the default-off official-schema path so a component-valued
property admits only a compatible component or binder, not arbitrary grammar
expressions, and carries a direct property's type to a later binder declaration.
The E891 checkpoint, E842 held-out rows, canvas cap 192, 12-second deadline,
strict compiler-tree policy, plan weights 4/2, and honest opaque slot contract
are fixed. E926 is the same-revision default-off control.

| Run | Treatment | parse | meaning-v1 | strict-v2 | slot fidelity | gold structure | gold type recall | reward | timeout / fallback | AgentV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E926 | v246 control | 1.0000 | 0.6000 | 0.2000 | 0.6800 | 0.3419 | 0.5810 | 0.8488 | 0 / 2 | 0/1 |
| E927 | strict direct types | 0.8000 | 0.6000 | 0.6000 | 0.6286 | 0.4405 | 0.6286 | 0.6994 | 1 / 2 | 0/1 |

The local constraint is correct: the form attempt now selects forward binders
for `Form.buttons` and `FormControl.input`, later constraining the first to
`Buttons`, instead of admitting `(null)`. The aggregate is unchanged from E924,
however. While building `Buttons`, the decoder reuses the input-owned binder as
a button item. Its accumulated `Input` and `Button` requirements have empty
intersection, so no later declaration can satisfy both and the attempt falls
back to `TextContent(":slot_0")`.

Retain the precision improvement only behind the existing default-off
`compiler_schema_component_types` capability and reject E927. Do not run smoke,
promote, sync, or change defaults. The next generalized step is to exclude
unresolved binders whose existing component requirement conflicts with the
active typed use site.

Both evals emitted AgentEvals JSONL and AgentV bundles; AgentV is 0/2. No ship
gates ran and no checkpoint was created. Result stamps are dirty because they
include the intended v246 implementation and unrelated concurrent experiment
files; those unrelated files remain untouched.
