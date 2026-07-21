# SPV0-01 (SLM-141): semantic planning and valid-state learning contract

**Status:** architecture contract defined (SPV0-01). No plan predictor, decode-path integration, experiment, or checkpoint is added, and no model or ship claim is made. Existing checkpoints and decode behavior are unchanged until a later SPV issue enables new flags behind a feature gate.

**Code:** `SemanticPlanV1` skeleton in [`src/slm_training/data/progspec/semantic_plan.py`](../../src/slm_training/data/progspec/semantic_plan.py); source manifest in [`src/slm_training/resources/autoresearch/semantic-planning-sources.json`](../../src/slm_training/resources/autoresearch/semantic-planning-sources.json).

**Reader contract:** `docs/design/` is the source of truth for coding agents and experiment reviewers; the Linear project document is planning context only. The terms below are meant to be precise enough that subsequent issues can implement modules, metrics, and experiment arms without inventing semantics.

## The distinction this contract adds

The repository already keeps compiler-owned legality authoritative and lets soft scores only *rank* it — see [`verified-scope-solver.md`](verified-scope-solver.md), whose subordination principle is that logits "cannot create a legal branch, discard all legal branches, or bypass final OpenUI validation."

What is missing is a contract distinguishing two properties that are currently conflated in discussion of "plans":

- **compiler-owned legality** — a candidate is admissible as the next action at the current decode position because the grammar, schema, binder, and slot constraints allow it. This already exists and remains the only hard authority.
- **learned semantic plan** — a predicted, retrieved, or merged soft structure that suggests roles, components, topology, symbols, and bindings without proving them.

> **Disambiguation (do not drift).** A "plan" here is a *soft structured hypothesis* about the desired program. It is distinct from the existing exact-state and verified-scope work of VSS/CAP, which deals with hard membership and proof-backed elimination. A predicted plan is **never** a proof object.

## Core division of labor

```text
learned semantic plan / legal-action scores / valid edits / program energy
                        !=
compiler-owned legality / exact state / hard membership / proof-backed elimination
```

The left-hand side may suggest, rank, seed, or score. The right-hand side may admit, reject, or certify. A plan-bearing system must keep these separable at every interface:

1. The compiler exposes `A_G(s)` — the exact legal action set at state `s`.
2. The plan predicts `plan(s)` — a structured hypothesis about the desired program.
3. The model scores `score_theta(x, plan, s, a)` for `a in A_G(s)`.
4. The decoder selects an action from `A_G(s)`; it never adds an illegal action or removes a legal one because of the plan.
5. Final verification remains authoritative.

## `SemanticPlanV1`

`SemanticPlanV1` is a pack-neutral, versioned IR with explicit provenance and optional/unknown fields. Every field belongs to one of the provenance classes below.

```text
SemanticPlanV1
  plan_version: "1"                        # schema version
  identity:
    pack_id / contract_hash: str            # pack contract this plan is for
    source_program_fingerprint: str | None  # canonical fingerprint of the source program, if any
    prompt_context_hash: str | None         # hash of prompt + context
    provenance: "gold" | "predicted" | "retrieved" | "merged" | "oracle_override"
  archetype:
    id: str | None
    distribution: dict[str, float] | None  # softmax over known archetypes
    confidence: float | None               # [0, 1]
  role_slots[]:
    role_id: str
    component_family: str | None           # e.g. "Card", "Text", "Button"
    candidate_distribution: dict[str, float] | None
    min_cardinality: int | None
    max_cardinality: int | None
    required: bool | None
    evidence_spans: list[str] | None       # opaque span references
  topology:
    parent_relation_candidates: list[dict] | None
    sibling_order_groups: list[list[str]] | None
    depth_bounds: tuple[int, int] | None
    cardinality_bounds: dict[str, tuple[int, int]] | None
    partial_order_constraints: list[dict] | None
  symbols[]:
    symbol_id: str
    semantic_role: str | None
    allowed_pointer_targets: list[str] | None
  bindings[]:
    role_slot_id: str
    candidate_symbols: list[str] | None     # live symbol/placeholder pointer candidates
    placeholder_fallback: bool | None
  coverage:
    named_requirements_accounted_for: list[str] | None
    unresolved_requirements: list[str] | None
  confidence_calibration:
    per_factor_confidence: dict[str, float] | None
    abstention_reason: str | None           # e.g. "ambiguous", "missing_pack", "low_confidence"
```

### Provenance classes

Every field in a plan carries one of the following provenance classes:

- **exact compiler fact** — independently derivable from authored/pack data or certified by an existing exact owner (e.g., the set of legal actions `A_G(s)`).
- **authored request fact** — explicit in the user prompt or context.
- **deterministic pack-derived fact** — derivable from the pack contract without model inference.
- **predicted soft preference** — output of a learned model; may be wrong and is never hard.
- **retrieved prototype evidence** — copied from a similar existing program; may be wrong and is never hard.
- **gold / oracle-only field** — available only in named oracle experiment arms; must be stripped from production manifests.

A predicted plan is **never** a proof object. A retrieved prototype is **never** a proof object. A merged plan inherits the weakest provenance of its inputs.

### `UNKNOWN` and abstention

`UNKNOWN` remains live. Absence of a plan prediction is not rejection. A plan with `abstention_reason` set must compile to unchanged baseline behavior: no hard restrictions, no candidate deletion, no override of compiler admissibility.

## Guarantee boundary

The following operations may be hard versus soft:

| Operation | Hard (compiler-owned) | Soft (plan-bearing) |
| --- | --- | --- |
| Enforce a plan fact | Only when independently derivable from authored/pack data or certified by an exact owner | Never for predicted/retrieved/merged fields |
| Use plan as seed | Never | Yes, for any provenance |
| Use plan as scorer feature | Never | Yes, via `score_theta(x, plan, s, a)` |
| Use plan to delete legal candidates | **Never** | No plan field authorizes this |
| Use certified plan-derived restrictions | Yes, if certified by an exact owner | No predicted/retrieved field |
| Gold/oracle fields in production | **Never** | Stripped by `to_production_dict()` unless `honesty_mode="oracle_diagnostic"` |

## Operational math

The model is conditional structured prediction over live compiler state:

```text
s_{t+1} = T_G(s_t, a_t)
a_t in A_G(s_t)
score_theta(x, plan, s_t, a_t)
```

Where:

- `T_G` is the exact compiler transition function.
- `A_G(s_t)` is the exact legal action set at state `s_t`.
- `x` is the input prompt/context.
- `plan` is a `SemanticPlanV1` hypothesis (possibly `None` or abstained).
- `score_theta` is the learned scoring function.

When stochastic search or remasking is used, the plan-bearing experiment must declare:

- the random variables and seeds;
- the proposal distribution over plans or plan-conditioned actions;
- the acceptance / ranking policy;
- the replay identity (how to reproduce the exact trajectory).

Do not describe sampling as a deterministic fixed-point equation.

## Experiment taxonomy

Canonical matched arm identifiers for plan-bearing experiments:

| Arm id | Description |
| --- | --- |
| `no_plan` | Baseline decoder with no plan input. |
| `gold_plan` | Oracle gold plan injected; fixture-only, never promotable. |
| `predicted_plan` | Learned plan predictor output as full plan input. |
| `retrieved_prototype` | Plan built only from retrieved prototype. |
| `retrieved_plus_predicted` | Prototype plan merged with predicted refinements. |
| `plan_as_seed` | Plan used only to seed the initial canvas; decoder is unchanged afterwards. |
| `plan_as_soft_scorer` | Plan features added to `score_theta`; no hard restrictions. |
| `certified_restrictions` | Only plan facts certified by an exact owner are enforced. |
| `unsafe_predicted_hard_control` | Predicted plan wrongly treated as hard; fixture-only benchmark control, never promotable. |

The `unsafe_predicted_hard_control` arm exists only to measure the damage of violating the guarantee boundary. It must never enter a champion or production manifest.

## Required diagnostics

Every plan-bearing experiment must emit:

- plan factor metrics (archetype accuracy, role-slot precision/recall, topology edge recall, symbol binding accuracy);
- gold-oracle substitution delta (`gold_plan` vs `no_plan`);
- plan-on versus plan-off choice changes at shared decode states;
- candidate coverage and false hard-prune counts (must be zero for soft arms);
- seed-to-target tree-edit distance;
- local, search, and global-ranker regret where available;
- plan confidence/calibration and abstention rate;
- verifier and latency cost;
- binding-aware meaningful and AgentV outcomes.

## Evidence and source manifest

The source manifest pins titles, authors, URLs/arXiv IDs, relevance, fidelity tags, and issue mappings. See [`src/slm_training/resources/autoresearch/semantic-planning-sources.json`](../../src/slm_training/resources/autoresearch/semantic-planning-sources.json). Tags follow the existing research-lineage policy: `Faithful`, `Adapted`, `Adjacent`, or `Diagnostic`.

## Migration and versioning

`plan_version` is `"1"` for this contract. Future schema changes must:

1. Introduce a new `plan_version` value.
2. Provide a `from_dict` migration path from the previous version.
3. Reject unknown versions fail-closed.
4. Update this document and the source manifest.

## Relation to existing contracts

This contract consumes the existing exact-state and evidence owners rather than recreating or weakening them:

- VSS/CAP exact state remains the hard carrier; see [`verified-scope-solver.md`](verified-scope-solver.md).
- EFS evaluation remains the primary outcome signal; see [`agentv-evaluation.md`](agentv-evaluation.md) and [`meaningful_program.py`](../../src/slm_training/evals/meaningful_program.py).
- LDI local-decision interventions remain the preferred fine-grained training signal; see [`local-decision-interventions.md`](local-decision-interventions.md).
- ProgramSpec and scope contracts remain the pack-derived fact owners; see [`src/slm_training/data/progspec/`](../../src/slm_training/data/progspec/).

## Non-goals

- Do not implement a plan predictor, X22 change, energy model, or long training run here.
- Do not claim formal abstract-interpretation or lattice theorems unless the exact formal objects and proofs are supplied.
- Do not replace EFS/VSS/CAP/LDI contracts.

## Implementation (SPV0-02, SLM-142)

**Status:** extraction/canonicalization/oracle-substitution/seed-construction harness wired and unit-tested. No predictor, no decode-path integration, no training run, and no ship claim is made.

**Code:**

- `src/slm_training/data/semantic_plan/extract.py` — `OpenUISemanticPlanExtractor` derives archetype, role slots, topology, symbols, and bindings from an OpenUI `ProgramSpec` AST. Gold provenance is preserved.
- `src/slm_training/data/semantic_plan/canonicalize.py` — `canonicalize_plan` normalizes role/symbol IDs so alpha-renamed or sibling-permuted plans share fingerprints. Symbol ordinals follow declared structural order, never lexical or semantic-role sorting. `plan_factor_fingerprints` emits SHA-256 hashes for `exact`, `archetype`, `role_set`, `topology`, and `bindings`.
- `src/slm_training/data/semantic_plan/oracle.py` — `PlanOracleSubstitutor` performs fail-closed factor-wise substitution (`archetype`, `roles`, `topology`, `bindings`). Gold/oracle plans are rejected for production manifests unless `honesty_mode="oracle_diagnostic"`; a contamination banner is available for diagnostic artifacts.
- `src/slm_training/data/semantic_plan/seed.py` — `PlanSeedBuilder` constructs a valid OpenUI seed from a plan using pack-owned constructors. It is fail-closed: multiple roots, missing content-prop mappings, or validation failures return `ok=False` with a reason rather than an illegal seed.
- `scripts/extract_semantic_plans.py` — CLI to emit `SemanticPlanV1` gold plans and factor fingerprints from a records JSONL file.

**Tests:** `tests/test_data/test_semantic_plan_extraction/` covers extraction structure, canonical idempotence, fingerprint stability across alpha-renaming, oracle fail-closed behavior, and seed validity via `validate()`.

**Honesty caveats:**

- Seed reconstruction maps each bound symbol to an opaque placeholder of the form
  `:slot_N.{property}` in declared symbol order. The semantic role may select the
  typed component property, but neither the external symbol spelling nor that role
  becomes marker identity. Caller-facing names require an explicit late binding map.
- Factor-wise oracle substitution is implemented only for the four factors listed above; `symbols` and `coverage` are carried implicitly through the baseline plan. A future issue can add explicit symbol/coverage oracle arms if diagnostics require them.
- Archetype inference is deterministic and pack-derived (root component + optional `direction`); no learned archetype predictor exists yet.
- Full plan-bearing eval matrix (gold seed, oracle factors, predicted-plan surrogate) remains blocked until frontier checkpoints and GPU compute are available.
