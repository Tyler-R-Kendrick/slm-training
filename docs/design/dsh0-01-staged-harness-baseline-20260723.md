# DSH0-01 staged-harness baseline (SLM-345)

**Disposition:** adopt the pinned vocabulary and reuse map. This is a
contract/identity audit (`fixture_demo`), not a train, eval, benchmark, model
quality result, checkpoint, certificate, or ship claim.

Machine-readable evidence:
[`dsh0-01-staged-harness-baseline-20260723.json`](dsh0-01-staged-harness-baseline-20260723.json).

## Pinned baseline

| Field | Identity |
| --- | --- |
| repository | clean `820e5f2ba9054d286395980473b6dd66831a228b` |
| quality frontier | DSH3-17 terminal disposition: `CERT_CAP2` rejected |
| output contract | `output_contract/v2` (symbol-only) |
| checkpoint implementation | `model.twotower/v227` |
| staged contract | `staged_harness_baseline/v1`, component `harness.staged/v1` |
| baseline SHA | `724bf511770017c51a4404e5cfede0d3ef68615787a0932a0401d6996fcd08fa` |
| run class | `fixture_demo` |

The baseline records the repository state before this DSH0 contract lands.
Every cited repository artifact has an exact SHA-256 in the JSON. The current
quality frontier remains negative: DSH3-17 rejected learned CAP2 and closed
DSH4. It does not imply that the newly defined staged CAP0 or CAP1 certificates
exist.

## Frozen orthogonal axes

```text
CAP0_GRAMMAR → CERT_CAP0
CAP1_SEMANTICS → CERT_CAP1
CAP2_TRANSFORM → CERT_CAP2

SUP_COMPILER | SUP_PARAPHRASE | SUP_DISTILL
EVAL_STATIC  | EVAL_PROPERTY   | EVAL_TRACE

difficulty: atomic | compositional | contextual | adversarial
```

Capability answers what the model has demonstrated. Supervision names how
examples were produced. Evaluation source names where evidence came from.
Difficulty describes task complexity. Distillation is a supervision process
and trace mining is an evidence process; neither can promote capability.

These names do not replace the repository's L0-L5 abstraction ladder. L0-L5
continues to describe how authored requests are abstracted and resolved.
CAP0-CAP2 describes certified model capability. No `Lv0`-style alias is
introduced.

## Canonical reuse map

| Planned seam | Existing owner | Relation |
| --- | --- | --- |
| language bundle and missing-slot behavior | `dsl/pack.py::DslPack` | reuse/extend |
| materialized rows and accepted outputs | `dsl/schema.py::ExampleRecord` | reuse |
| visible runtime symbol authority | `data/contract.py::RuntimeSymbol` | reuse; never restore hidden gold |
| legal prefix state | `grammar/fastpath/compiler_draft.py::CompletionForest` | reuse |
| exact support evidence | `dsl/solver/support.py` | reuse |
| topology state/action machinery | `models/grammar_diffusion.py` | reuse; default-off until certified |
| template ownership boundary | `models/template_fill.py` | reuse |
| legal operator enumeration | `dsl/operators/legal_set.py` | reuse |
| frozen legal-action fixture | `resources/evals/cap2_operator_v1.json` | reuse as eval only |
| authored-request abstraction | `abstraction-house-style.md` L0-L5 | separate axis |
| generic CAP1 authority | SLM-343 | dependency after `CERT_CAP0` |
| ambiguity-aware CAP1 data | SLM-344 | dependency after SLM-343 |

This map supersedes any proposal to build a second pack registry, row schema,
runtime-symbol table, legal-prefix representation, topology engine, template
filler, or legal-action enumerator for the staged curriculum.

## Fail-closed evidence semantics

`StagedHarnessBaselineV1` serializes deterministically and exposes
`blocking_reasons()` / `require_reusable()`. A reusable baseline requires a
known clean commit, known frontier/contract/checkpoint generations, and
verified artifact hashes. Missing or unusable evidence is explicitly
`unknown` or `invalid`, with a reason. It is never represented by numeric zero.

## Validation

Focused contract tests cover axis separation, byte-stable serialization/hash,
explicit missing evidence, verified digest enforcement, schema mismatch, and
run-class rejection. Repository policy, component-version checks, Ruff,
compileall, JSON parsing, and changed-test selection are part of the delivery
gate.

## Next disposition

Proceed to SLM-346 by extending these types with
`SymbolicSurfacePolicyV1`. CAP1 training remains blocked on a valid
`CERT_CAP0`; CAP2/DSH4 remains closed under DSH3-17.
