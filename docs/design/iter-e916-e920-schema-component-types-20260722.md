# E916-E920: official component-schema constraints

E916-E920 test whether the lexer compiler should propagate official-schema
component types from typed-array forward-reference use sites to later binder
declarations, then whether the same opt-in path should constrain singular
component-valued properties. The E891 checkpoint, E842 held-out rows, canvas
cap 192, 12-second record deadline, strict compiler-tree policy, and plan
weights 4/2 are fixed. E918 is the same-revision margin-0 control for the
isolated E919 treatment.

| Run | Treatment | parse | meaning-v1 | strict-v2 | slot fidelity | gold structure | gold type recall | reward | timeout / fallback | AgentV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E916 | item-margin control | 1.0000 | 0.4000 | 0.2000 | 0.5133 | 0.2945 | 0.4476 | 0.7826 | 0 / 4 | 0/1 |
| E917 | item margin + forward types | 0.8000 | 0.4000 | 0.4000 | 0.4619 | 0.3223 | 0.4952 | 0.6410 | 1 / 4 | 0/1 |
| E918 | isolated control | 1.0000 | 0.4000 | 0.2000 | 0.5133 | 0.2945 | 0.4476 | 0.7826 | 0 / 4 | 0/1 |
| E919 | forward types only | 0.8000 | 0.6000 | 0.4000 | 0.6333 | 0.4452 | 0.5810 | 0.6960 | 1 / 4 | 0/1 |
| E920 | direct + forward types | 0.6000 | 0.4000 | 0.4000 | 0.4333 | 0.3323 | 0.4667 | 0.5086 | 2 / 3 | 0/1 |

E919 proves the generalized path is active and semantically useful: invalid
`Separator` declarations disappear from typed arrays, settings becomes a clean
compact strict-v2 pass with slot fidelity and gold type recall 1.0, and the
aggregate v1/v2, fidelity, structure, and recall signals all improve. It is not
a quality win because dual-card hits the 12-second deadline, parse falls to
0.8, and reward falls 0.7826→0.6960.

E920 additionally prevents direct `Button`/`TextContent` values where the
official schema requires `Buttons`/`Input`. That is locally correct but exposes
the unresolved planner problem: after unique slots are consumed, some required
component declarations have no legal content-bearing completion. Form and
dual-card time out, so parse falls to 0.6 and reward to 0.5086.

Retain the official-schema implementation behind the explicit
`compiler_schema_component_types` capability, default off. Reject all E891
treatments; do not run smoke, promote, sync, or change global defaults. E906
remains the best E891 diagnostic recipe. The next useful lever must allocate
typed forward declarations before their unique slot resources are consumed,
not increase timeout or weaken schema/semantic gates.

All five evals emitted AgentEvals JSONL and AgentV bundles; AgentV is 0/5. No
ship gates ran and no checkpoint was created. Result stamps are dirty because
they include the intended v243 implementation and unrelated concurrent
experiment files; those unrelated files remain untouched.
