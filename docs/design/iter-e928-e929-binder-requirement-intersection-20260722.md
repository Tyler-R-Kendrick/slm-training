# E928-E929: binder requirement intersection

E928-E929 prevent an unresolved binder from being reused at a typed component
use site when its accumulated official-schema requirements do not overlap the
new site's allowed component family. The check is structural: it reads binder
identity and schema component sets only, never marker spelling or prompt text.

The E891 checkpoint, E842 held-out rows, canvas cap 192, 12-second deadline,
strict compiler-tree policy, plan weights 4/2, and honest opaque slot contract
are fixed. E928 is the same-revision default-off control.

| Run | Treatment | parse | meaning-v1 | strict-v2 | slot fidelity | gold structure | gold type recall | reward | timeout / fallback | AgentV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E928 | v247 control | 1.0000 | 0.6000 | 0.2000 | 0.6800 | 0.3419 | 0.5810 | 0.8488 | 0 / 2 | 0/1 |
| E929 | intersect pending types | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.5430 | 0.7429 | 0.7640 | 1 / 1 | 0/1 |

The mechanism succeeds on the previously failing Form row. `Form.buttons`
resolves through `b1 = Buttons([b3])`, while the input-owned binder resolves
separately as `b2 = RadioGroup(...)`; the incompatible Input/Button alias is
gone. Four of five treatment rows are parseable and strict-v2 meaningful with
perfect slot fidelity and type recall on each accepted row. The remaining
dual-card row times out and emits an empty prediction.

Retain the generalized intersection check behind the existing default-off
`compiler_schema_component_types` capability. Reject default enablement and do
not promote: E929 materially improves strict fidelity, structure, and type
recall, but parse falls 1.0 to 0.8, reward remains below E928, and AgentV fails.
The next action is to diagnose the dual-card timeout under the now-satisfiable
typed-binder path. No ship gates ran and no checkpoint was created.

Both evals emitted AgentEvals JSONL and AgentV bundles. Result stamps are dirty
because they include the intended v247 implementation and unrelated concurrent
experiment files; those unrelated files remain untouched.
