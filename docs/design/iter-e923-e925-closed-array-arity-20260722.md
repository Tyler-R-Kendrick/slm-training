# E923-E925: closed-array schema arity

E923-E925 repair the exact compiler contradiction exposed by E922. After a
recursive component's final array-valued argument closed, the schema arity
filter still treated the array as active and admitted a trailing comma. The
next compiler state had no completion. The fix exempts comma filtering only
while the schema array is actually open.

The E891 checkpoint, E842 data, canvas cap 192, 12-second record deadline,
strict compiler-tree policy, plan weights 4/2, typed-item margin 0, and honest
opaque slot contract are fixed. E923 is the v245 default path, E924 enables the
default-off schema-inline treatment, and E925 guards smoke on the retained
default path.

| Run | Suite / treatment | parse | meaning-v1 | strict-v2 | slot fidelity | gold structure | gold type recall | reward | timeout / fallback | AgentV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E923 | held-out v245 default | 1.0000 | 0.6000 | 0.2000 | 0.6800 | 0.3419 | 0.5810 | 0.8488 | 0 / 2 | 0/1 |
| E924 | held-out schema inline | 0.8000 | 0.6000 | 0.6000 | 0.6286 | 0.4405 | 0.6286 | 0.6994 | 1 / 1 | 0/1 |
| E925 | smoke v245 default | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5783 | 0.6667 | 0.9450 | 0 / 0 | 0/1 |

The compiler correction is a default-path quality win. Against v244 E921,
E923 keeps parse at 1.0 while improving meaning-v1 0.4→0.6, slot fidelity
0.5133→0.6800, gold structure 0.2945→0.3419, gold type recall
0.4476→0.5810, and reward 0.7826→0.8488. Fallback-marked rows fall from four
to two. E925 exactly matches E907's smoke quality aggregate with no timeout or
fallback.

E924 confirms the local schema-inline mechanism more strongly: input, tabs,
and settings are compact strict-v2-valid programs with slot fidelity and gold
type recall 1.0, raising strict-v2 to 0.6. It is still rejected because
dual-card times out, form collapses, parse falls to 0.8, and reward falls to
0.6994. This is omitted-content loss, not a penalty for minification.

Retain the closed-array arity repair as the canonical compiler behavior. Keep
`compiler_schema_component_types` default off and reject E924. E906 remains
the strongest historical E891 diagnostic aggregate; rows across decoder
versions are not interchangeable. Next target whole-path feasibility for form
and repeated-card plans without increasing timeout or weakening gates.

All three evals emitted AgentEvals JSONL and AgentV bundles; AgentV is 0/3. No
ship gates ran and no checkpoint was created. Result stamps are dirty because
they include the intended v245 implementation and unrelated concurrent
experiment files; those unrelated files remain untouched.
