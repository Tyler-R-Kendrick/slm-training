# E781-E790 — numeric literal planning closure

**Date:** 2026-07-22  
**Decision:** retain model v224; reject margin policy promotion  
**Evidence:** [`iter-e781-e790-numeric-literal-planning-20260722.json`](iter-e781-e790-numeric-literal-planning-20260722.json)

E780's weakest honest metric was component recall. Every held-out miss omitted
an explicitly requested schema family (`Form`, `Card`, `Input`, `Tabs`, or
`Slider`), while semantic-plan telemetry recorded zero applications. E781
activated the existing soft plan score at weight 1; it applied five times but
changed no choices. E782 used the previously retained missing-family margin 2.
It correctly selected `Slider`, then exposed a compiler dead end at the first
numeric property.

E783-E787 repaired that failure at its source. Lexer-native numeric literals
are framed as `LIT_NUM`, schema-valid byte tokens, and `LIT_END`; the compiler
previously treated the frame markers as source lexemes, allowed arbitrary
bytes, crossed both frame boundaries with grammar-only forced suffixes, and
used a second divergent token-to-source conversion in persistent grammar
state. Model v224 now uses one shared token-surface conversion, a numeric
prefix automaton for sign/digits/decimal/exponent, forbids empty numeric
closure, and makes both literal markers hard draft boundaries. Regression
coverage proves incremental grammar state and full tokenizer decode agree and
that schema arity resumes after `LIT_END`.

The repaired E788 settings row emits `Slider` and `SwitchGroup` with no
fallback: meaningful-v1 rises 0→1, structure 0.1978→0.4133, and component
recall 0→0.5. Marker fidelity is 0.8, so this is not a complete repair.

The E789 local CPU held-out n=5 replay is a structural Pareto versus E780:
meaningful-v1 rises 0.2→0.6, structure 0.3921→0.5622, component recall
0.2571→0.5857, and p95 falls 3543.57→3085.20 ms, with parse 1.0 and zero
fallbacks/timeouts. But marker fidelity and recall fall from 1.0 to 0.7076,
reward falls 0.9466→0.8613, strict-v2 remains 0, and AgentV remains 0/1.
Therefore the numeric/compiler repair is retained while margin 2 remains an
experimental lever, not the canonical policy.

E790 composed the previously accepted visible-role, schema-role, and
coverage-close owners on the targeted settings row. It changed which markers
owned planned families but produced identical headline metrics and still
omitted one declared marker, so the composition is rejected as no-effect.

Config-levers v16 also moves changed-test parallelism out of the CI helper and
into `slm_training.levers`, beside the hard run cap, and publishes it through
the canonical lever catalog. The workflow and test assertions are unchanged.

Every accepted run used local CPU under the 110-second command cap and a
12-second per-record decode guard. The first E781 attempt failed mandatory
AgentV dependency resolution and is explicitly invalid; the replay and every
later arm emitted AgentEvals plus an AgentV bundle. No remote experiment
workflow ran. No checkpoint was created, synced, or promoted, so the model card
is unchanged. All predictions contain only grammar/AST symbols, schema enum
literals, and request-declared template markers; no free-form completion text
was introduced.

Next, make the declared marker inventory authoritative within planned-family
construction so the structural gain does not trade away marker coverage.
