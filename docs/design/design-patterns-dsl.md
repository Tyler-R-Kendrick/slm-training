# Design-patterns DSL — language design (F3)

Design document only. No grammar, corpus, model, or ship claim exists yet; this
scopes the third DSL target of the program (OpenUI → GraphQL → **patterns**) and
must be committed and reviewed before any pack code (F1 contract) or training
data is generated. Linear SLM-44.

## Why this DSL is different (the oracle problem)

The first two targets have a *free* validity oracle:

| DSL | Grammar | Validity oracle |
| --- | --- | --- |
| OpenUI | `@openuidev/lang-core` | parser + renderer (a program renders or it does not) |
| GraphQL | graphql-core (F2, landed) | parser + schema validation against the introspection schema (`packs/graphql.py`; graphql-js byte parity is a non-goal) |
| **Patterns** | (to design) | **none free — must be defined** |

A "software design pattern / algorithm / data-structure" DSL describes an
*intent and structure*, not a runnable artifact whose success is observable.
"Correct" is therefore a modeling decision, not a given. Choosing it wrong makes
every downstream reward, ship gate, and autoresearch feedback meaningless (the
same failure mode the OpenUI E225→E226 honesty fix corrected). This doc's main
job is to pick an oracle we can defend.

## Candidate oracles (ranked)

1. **Typed AST + structural type checker (recommended first cut).** The DSL is a
   typed algebra of pattern *roles* (participant, collaboration, constraint) and
   relations (implements, delegates, composes, notifies). Validity = the AST
   type-checks against the role/relation schema (arities, allowed edge types,
   acyclicity where required). This is a *free* oracle once the schema exists —
   the same shape as OpenUI's typed-AST generator (`data/progspec/`), and it
   reuses the whole existing pack machinery. It proves *well-formedness*, not
   *semantic correctness*, and we say so.
2. **Property tests over generated reference implementations.** Each pattern
   instance carries executable properties (e.g. Observer: "every attached
   observer receives every notification"; Iterator: "traverses each element once
   and terminates"). A generated implementation must pass them. Strong oracle,
   but expensive and language-bound (needs a runtime); a second-phase upgrade,
   not the first cut.
3. **Bisimulation / refinement against a reference implementation.** Strongest,
   heaviest; out of scope until (1)+(2) are established.

**Decision:** ship the pack on oracle (1) — typed well-formedness — and label
its reward `well_formed_not_behavioral` (analogous to OpenUI's
`syntax != meaningful`). Oracle (2) is a documented follow-up that upgrades the
gate; it is never silently conflated with (1).

## Language sketch

A pattern program is a set of typed declarations over a fixed vocabulary of
roles and relations. Illustrative (subject to the schema, not final syntax):

```
pattern Observer
  role Subject      { holds: ObserverSet; emits: StateChange }
  role Observer     { reacts_to: StateChange }
  relation notifies : Subject -> Observer   [fan_out, on = StateChange]
  invariant every(Observer).receives(every StateChange)   # oracle-2 property
```

Design commitments that keep it inside the existing program machinery:

- **Typed AST**, so `data/progspec/`-style coverage-guided generation applies.
- **Placeholders** for concrete names (`:subject.name`) exactly like OpenUI, so
  identifiers stay content-routed and the D2 canonicalizer / D1 forward
  simplification carry over unchanged.
- **Scope rules**: role names are De Bruijn-style relative indices (Track C1),
  so alpha-equivalent patterns share one canonical form.
- **Canonical form**: role declaration order fixed topologically; relations
  sorted; the D2 canonicalizer generalizes with a pattern-specific rule set.

## Pack contract mapping (F1)

F1 (SLM-34) landed the contract: `DSLPack` in
[`src/slm_training/dsl/packs/types.py`](../../src/slm_training/dsl/packs/types.py)
with builtin `openui` and `toy-layout` instances and contract-invariant tests
(`tests/test_dsl/test_packs.py`). The patterns pack supplies:

| Pack slot | Patterns DSL |
| --- | --- |
| grammar | new Lark grammar for the role/relation algebra |
| validity oracle | typed well-formedness checker (oracle 1); property runner (oracle 2, later) |
| typed-AST generator | coverage-guided over the role/relation/invariant space |
| canonicalizer | D2 + pattern rewrite rules (relation sort, role reindex) |
| scope rules | De Bruijn role indices |
| placeholder policy | concrete names → `:role.field` placeholders |

## Open questions (resolve before F1 pack work)

- Vocabulary scope: GoF-only, or algorithms + data structures too? (Affects
  schema size and whether one grammar or a family is needed.)
- Is oracle (2) worth a sandboxed runtime, or do we stay at well-formedness for
  the whole first pack cycle?
- Does the pattern algebra need first-class *generics/type parameters*? If so,
  the scope model must handle bound type variables, not just role names — this
  is where the context-sensitive α-equivalence caveat (2401.02948) bites.

## Verification (when built)

- Design: this doc committed and reviewed (no code yet).
- Pack: the standard end-to-end fixture run (generate → scratch train → eval →
  document) green under the F1 contract, with the reward honestly labeled
  `well_formed_not_behavioral` until oracle (2) exists.

## Honesty

No grammar, corpus, oracle, model, or metric exists yet. This is scope + an
oracle decision, deliberately committed before any training so the reward is
defined up front rather than reverse-engineered from a run.
