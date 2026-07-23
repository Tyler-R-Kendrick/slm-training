# E921-E922: schema-aware recursive inline items

E921-E922 test whether official schema typing should allocate a unique recursive
typed-array item inline before forward binders consume its slot resources. The
E891 checkpoint, E842 held-out rows, canvas cap 192, 12-second record deadline,
strict compiler-tree policy, plan weights 4/2, and typed-item margin 0 are fixed.
E921 is the same-revision default-off control.

| Run | Treatment | parse | meaning-v1 | strict-v2 | slot fidelity | gold structure | gold type recall | reward | timeout / fallback | AgentV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E921 | v244 control | 1.0000 | 0.4000 | 0.2000 | 0.5133 | 0.2945 | 0.4476 | 0.7826 | 0 / 4 | 0/1 |
| E922 | recursive inline item | 0.6000 | 0.4000 | 0.4000 | 0.4333 | 0.2993 | 0.4667 | 0.5110 | 2 / 2 | 0/1 |

The local mechanism works: settings emits one compact schema-correct
`SwitchItem`, and settings plus input are strict-v2 valid with slot fidelity
and gold type recall 1.0. This confirms that minification itself is not being
penalized by slot fidelity; the aggregate loss comes from the other rows
omitting required slots or timing out.

The treatment is not an overall quality win. Form and dual-card still hit the
12-second deadline. Tabs chooses the correct recursive `TabItem`, fills all six
slots, then takes an optional-argument comma for which the compiler has an empty
completion forest. The retry falls back to `TextContent(":slot_0")`. Relative
to E921, parse falls 1.0→0.6, slot fidelity 0.5133→0.4333, and reward
0.7826→0.5110 despite strict-v2 rising 0.2→0.4.

Retain the generalized behavior only behind the existing default-off
`compiler_schema_component_types` capability and reject E922. Do not run smoke,
promote, sync, or change defaults. The next lever should make optional recursive
component continuations completion-aware, rather than increasing timeout or
weakening schema and semantic gates.

Both evals emitted AgentEvals JSONL and AgentV bundles; AgentV is 0/2. No ship
gates ran and no checkpoint was created. Result stamps are dirty because they
include the intended v244 implementation and unrelated concurrent experiment
files; those unrelated files remain untouched.
