# E208–E213 — contextual grammar-decision alignment

Status: **diagnostic only; no checkpoint promoted; ship gates not run**.

## Corpus evidence

The committed 496-record E177 corpus has 496 populated root containers and no
empty roots. Bound containers are populated in 156 declarations and empty in
12; there are 1,002 bound leaf declarations. Empty lists are therefore legal
but strongly role-dependent. The audit uses lexer-native statements and
generated-schema `children` roles, not prompt or component-name cases.

## Matched experiments

All three trains use the exact E205 corpus and mixture hashes, CPU, 32 steps,
batch 4, seed 0, frozen SmolLM2-135M context, lexer output, schema/slot context,
no DESIGN.md context, alignment weight 1.0, and no checkpoint sync.

| Train | Parser/schema decision signature | Last loss | Wall s | Aligned rows | Alignment loss first → last |
| --- | --- | ---: | ---: | ---: | ---: |
| E208 | `RSQB` empty vs populated | 7.4938 | 97.21 | 1,141 | 70.5770 → 2.8706 |
| E210 | `RSQB` + root/bound scope + occupancy | 7.5847 | 96.21 | 1,191 | 71.2849 → 2.2678 |
| E212 | declaration/reference + scope + generated slot | 7.5117 | 114.65 | 1,466 | 69.8059 → 2.3547 |

The harness now emits row counts and cross-entropy loss for every contextual
decision kind without an extra model forward. E212 sees 82
`bind_reference_root_children` rows; that loss falls from 61.4179 to 1.0285.
Hashes and complete telemetry are in
[the result JSON](iter-e208-e213-contextual-decisions-20260716.json).

## Strict diagnostics

Every evaluation is a one-example smoke diagnostic with no unconstrained
fallback. Each emitted AgentEvals JSONL and an AgentV SDK bundle.

| Eval | Checkpoint | Syntax | Meaningful parse | Structure | Component recall | Normalized fidelity | Placeholder validity | Fallback | p50 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E209 | E208 | 1.0 | 0.0 | 0.1917 | 0.0 | 0.0 | 0.0 | 0 | 4374.25 |
| E211 | E210 | 1.0 | 0.0 | 0.1917 | 0.0 | 0.0 | 0.0 | 0 | 9995.95 |
| E213 | E212 | 0.0 | 0.0 | 0.1333 | 0.0 | 0.50 | 0.70 | 0 | 10727.10 |

E209 shows that occupancy-only stratification leaks the 12 legitimate bound
empty examples into the root decision: it emits `root = Stack([])`. E210 adds
typed declaration scope, but E211 still emits an empty root because the first
child binder reference remains mixed with every declaration/reference binder
decision.

E212 replaces that generic bucket with a contextual signature derived from
semantic token kind, declaration/reference role, typed root/bound identity, and
the active generated-schema property. E213 then chooses the first root child
binder over `]` by 1.82 log-score and reaches normalized placeholder fidelity
0.50. It still fails: a generated `FormControl` required `input` resolves to
null, so syntax and meaningful parse are zero. This is a negative ship result;
none of the checkpoints is promotable.

## Next hypothesis

Derive component-choice signatures from generated schema value roles and make
required non-null object/reference slots reachable through generated schema
paths. Do not add component names, prompt strings, or observed-output cases.
