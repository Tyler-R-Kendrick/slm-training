# PGS-I01 (SLM-512): proof-carrying goal support — design, authority, and threat model

**Status:** integrated architecture documented against landed code (PGS-A through PGS-H).
No new production code, campaign, checkpoint, or ship claim is made here.

**Code owners:** see the architecture path below; symbol ownership is recorded in
[`repository-ownership-map.md`](repository-ownership-map.md) (maintained by SLM-513).

**Reader contract:** `docs/design/` is the source of truth for coding agents and
experiment reviewers; Linear planning text is context only. Every claim below
carries an explicit **proof/evidence class** and **bounded scope**. Fixture
evidence is diagnostic wiring only — not production validation, not MODEL_CARD,
not ship.

**Related measured evidence:** [`proof-carrying-goal-support-fixture-results.md`](proof-carrying-goal-support-fixture-results.md)
(PGS-H02 / SLM-511). **Related structural laws:** [`proof-carrying-goal-support-proofs.md`](proof-carrying-goal-support-proofs.md)
(PGS-F01 / SLM-506).

---

## The distinction this contract adds

The repository already separates several properties that review discussion often
conflates. Proof-carrying goal support (PGS) adds a **goal-aware support**
layer on top of existing compiler legality and VSS structural support — without
replacing either.

| Property | Meaning | Hard authority? | Owner |
| --- | --- | --- | --- |
| **Grammar legality** | Next token/action is compiler-admissible at the current decode position (Lark CFG, binders, schema). | Yes — always | Completion forest / `CompletionDomainV1` |
| **Structural support (VSS)** | Candidate participates in at least one bounded, fully verified completion under structural/oracle verification. | Yes — when replay-valid and complete coverage | `EnumerativeSupportOracle`, `SupportCertificate`, `exact_closure` |
| **Goal support (PGS)** | Candidate participates in at least one bounded completion that also satisfies **compiled goal constraints** under a pinned `GoalVerifierProfileV1`. | Yes — only under `production_exact` + hard authority + replay | `GoalSupportProvider`, `GoalSupportResultV1`, `exact_goal_closure` |
| **Learned / advisory** | Predicted plans, prompt-derived requirements, retrieved prototypes, rankers. | Never — suggest/rank only | `SemanticPlanV1`, advisory constraints, preference materializers |
| **Evaluation-only** | Frozen eval-oracle fixtures and diagnostic arms. | Never for prune — label/diagnose only | `evaluation_oracle` profile, evaluation constraint partition |
| **Certified pruning** | Removing a candidate from the live decode forest because replay proves it goal-`UNSUPPORTED` under production-exact bounds. | Yes — `goal_support_mode=certified` only | `goal_support_certified_prune` |
| **Model quality / ship** | Multi-suite `--ship-gates`, promotion, MODEL_CARD roster. | Separate ladder | Honest ship eval — never inferred from support alone |

> **Disambiguation (do not drift).** Goal support is *participation in a
> goal-verified completion*, not grammar legality alone and not the E283
> decode-coverage **support signature** telemetry
> ([`iter-e283-signature-support-repair-20260717.md`](iter-e283-signature-support-repair-20260717.md)).
> Terminal pruning uses `GoalTerminalStatus` + `SupportVerdict`, not
> `VerificationReport.ok` from the gate stack summary.

---

## Architecture path (integrated stack)

End-to-end data flow through **named, landed owners** — no parallel search engine,
verifier stack, synthesis envelope, semantic-plan schema, decision schema, or
evidence root:

```text
GenerationRequest ──┐
VerifiedSynthesisProblemV1 ──┼──► compile_goal_constraints (requirements_compile.py)
                             │         │
                             │         ▼
                             │    CompiledGoalConstraintSetV1
                             │         │
                             │         ▼
                             │    GoalVerifierProfileV1  (mode + required ids + authority)
                             │         │
                             │         ▼
                             │    OpenUIGoalVerifier (openui_support.py)
                             │      structural → ProgramSpec → SemanticPlanV1
                             │      → evaluate_goal_constraints → G0–G12 gates → evaluators
                             │         │
                             │         ▼
                             │    GoalTerminalEvidenceV1  (redacted, digest-bound)
                             │         │
                             ▼         ▼
                    GoalSupportProvider.check_with_sidecar()
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
   EnumerativeSupportOracle          GoalSupportResultV1
   → SupportCertificate              (sidecar keyed by base_certificate_digest)
              │                             │
              └──────────┬──────────────────┘
                         ▼
              replay_goal_support_result / exact_goal_closure
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
 DomainObstructionCoreV1   analyze_goal_domain   decode seam
 (obstruction/core)         (domain adequacy)      off | diagnostic | certified
         │               │               │
         └───────────────┴───────────────┘
                         ▼
              materialize_goal_support (decision_events_v2.py)
                         ▼
              DecisionEventV2 / ObjectiveView
                         ▼
              governed evidence (trace store, campaign JSON, Lean mapping)
```

| Stage | Primary symbols | File |
| --- | --- | --- |
| Request / synthesis envelope | `GenerationRequest`, `VerifiedSynthesisProblemV1` | `data/contract.py`, `data/progspec/synthesis_problem.py` |
| Compiled typed constraints | `CompiledGoalConstraintSetV1`, `GoalConstraintV1`, `compile_goal_constraints`, `evaluate_goal_constraints` | `data/progspec/goal_constraints.py`, `data/semantic_plan/requirements_compile.py` |
| Profile | `GoalVerifierProfileV1`, `validate_profile_against_constraint_set`, `profile_mode_authority_table` | `dsl/solver/goal_support.py` |
| Terminal verifier | `OpenUIGoalVerifier`, `GoalTerminalEvidenceTrace`, `GoalTerminalEvidenceV1` | `dsl/solver/openui_support.py` |
| Structural oracle | `EnumerativeSupportOracle`, `SupportQuery`, `SupportCertificate`, `SupportVerdict` | `dsl/solver/support.py` |
| Goal-aware wrapper | `GoalSupportProvider`, `GoalSupportResultV1`, `replay_goal_support_result`, `exact_goal_closure` | `dsl/solver/goal_support.py` |
| Obstruction | `DomainObstructionCoreV1`, `compute_domain_obstruction_core`, `obstruction_core_algorithm_table` | `dsl/solver/goal_support_obstruction.py` |
| Domain adequacy | `GoalActionEvidenceV1`, `GoalDomainAdequacyReportV1`, `analyze_goal_domain`, `domain_adequacy_classification_table` | `dsl/solver/goal_support_domain_adequacy.py` |
| Decision materialization | `DecisionEventV2`, `GoalSupportStateBinding`, `materialize_goal_support` | `harnesses/preference/local_decisions.py`, `decision_events_v2.py` |
| Decode integration | `GOAL_SUPPORT_MODES`, `build_goal_support_decode_binding`, `goal_support_certified_prune`, `goal_support_diagnostic_observe` | `dsl/solver/decode.py` |
| Formal drift guard | `goal_support_mapping.py`, Lean `GoalSupport.lean` | `formal/`, `src/leverproof_lean/` |

**Why no new stack components:** PGS is a thin composition layer. Search remains
`EnumerativeSupportOracle` + `exact_closure` (VSS). Closure authority is guarded
by `exact_goal_closure`, not a second engine. Verification remains
`OpenUIGoalVerifier` over the existing gate/evaluator stack. Constraints compile
into the existing `CompiledGoalConstraintSetV1` contract. Synthesis problems
stay `VerifiedSynthesisProblemV1`. Semantic plans stay `SemanticPlanV1` inputs
to the terminal verifier — not a new plan schema. Decision events stay
`DecisionEventV2`; PGS adds `materialize_goal_support` mapping only. Evidence
stays digest-bound sidecars + redacted terminal records — not a new evidence root.

Implementation pin: `GOAL_SUPPORT_IMPLEMENTATION_VERSION = "goal_support/v1"`.

---

## Semantic distinctions — verdicts and classifications

### Four action partitions (`GoalActionPartition`)

Assigned by `_assign_partition()` over one bounded legal set:

| Partition | Condition | Prune-eligible? |
| --- | --- | --- |
| `supported` | `SupportVerdict.SUPPORTED` and sidecar replay ok | No (keep) |
| `unsupported` | `SupportVerdict.UNSUPPORTED`, replay ok, **hard profile** | Yes — certified mode only |
| `unknown` | Observed but not above (soft unsupported, replay fail, `UNKNOWN` verdict) | **Never** |
| `unobserved` | Not queried (cap-truncated actions); no certificate/sidecar digests | **Never** — inference forbidden |

Cap policy (`domain_adequacy_cap_policy_table`): above `exact_action_cap`, first
N `action_id`s in lexicographic order are queried; remainder → `unobserved`.

### Three support verdicts (`SupportVerdict`)

| Verdict | Meaning | Removal authority |
| --- | --- | --- |
| `SUPPORTED` | Replay-valid witness completion under profile | None |
| `UNSUPPORTED` | Complete bounded search proves no goal-verified terminal | `production_exact` + hard replay only |
| `UNKNOWN` | Partial coverage, budget stop, stale identity, replay failure | **Never** |

### Five domain-adequacy classifications (`DomainAdequacyClassification`)

Closed table in `domain_adequacy_classification_table()`:

| Classification | Meaning |
| --- | --- |
| `domain_adequate_selected_supported` | `supported` nonempty and selected action ∈ supported |
| `selection_regret` | `supported` nonempty and selected ∈ unsupported |
| `selection_unresolved` | `supported` nonempty and selected ∈ unknown ∪ unobserved |
| `domain_inadequate_under_bounds` | No supported/unknown/unobserved actions; every legal action replay-valid `UNSUPPORTED` under hard profile; cap not applied; obstruction summary present |
| `coverage_unknown` | All other cases (partial forest, bound exhaustion, cap exclusion, advisory/eval modes, etc.) |

> **Never use global UNSAT language.** Inadequacy is always
> **`domain_inadequate_under_bounds`**: relative to the pinned profile, state
> fingerprint, and declared bounds — not a claim that no solution exists in the
> full OpenUI semantic space.

### Obstruction cores (`DomainObstructionCoreV1`)

An obstruction core is a **profile/state/bounds-relative hitting set** over
exact mandatory failure atoms — not a causal or globally minimum explanation
unless the recorded `algorithm_mode` explicitly claims finite search and
`minimum_cardinality_exact` (otherwise `subset_minimal_exact` or
`sound_overapprox`). Lean proves subset-minimal deletion only; Python
`sound_overapprox` is explicitly **not** claimed subset-minimal
([`proof-carrying-goal-support-proofs.md`](proof-carrying-goal-support-proofs.md)).

Emission requires replay-valid hard `UNSUPPORTED`, complete action coverage, no
cap, and no budget-stop classification — see `obstruction_core_emission_allowed`.

---

## Authority matrix — sources, profiles, and operations

### A. Compile-time source → authority (`source_to_authority_matrix`)

`goal_support_mode` is `off`, `diagnostic`, or `certified`; the default is
`off`. `goal_support_query_cap` defaults to 32. Request-local
`goal_support_contexts` supply the pinned profile and constraint set and are not
stored in checkpoints. These controls are available only through the
programmatic `TwoTowerConfig` plus `generate_batch_requests` context API. They
are deliberately absent from ModelBuild, CLI, and OpenFeature surfaces: those
surfaces cannot construct or bind a request-local `VerifiedSynthesisProblemV1`,
so exposing an enable switch there would be unusable or invite invented
authority.

| Source kind | Authority tier | Completeness | `may_prune` | Partition |
| --- | --- | --- | --- | --- |
| `generation_request` | compiler-hard | EXACT | **False** | hard |
| `pack_contract` | compiler-hard | EXACT | **True** | hard |
| `verification_requirement` | verifier-hard | EXACT | **True** | hard |
| `prompt_requirement` | advisory-learned | HEURISTIC | False | advisory |
| `oracle_diagnostic` | oracle-diagnostic | HEURISTIC | False | advisory |
| `evaluation_fixture` | evaluation-only | EXACT | False | evaluation |

Prune law: `may_prune=True` requires hard authority + EXACT + source ∈
`{pack_contract, verification_requirement}` only (`GoalConstraintV1`).

### B. Profile mode → authority (`profile_mode_authority_table`)

| | `production_exact` | `evaluation_oracle` | `advisory_diagnostic` |
| --- | --- | --- | --- |
| Allowed tiers | compiler-hard, verifier-hard | evaluation-only | advisory-learned, oracle-diagnostic |
| Constraint partition | `hard_constraint_ids` | `evaluation_constraint_ids` | `advisory_constraint_ids` |
| Completeness required | EXACT | none | none |
| Heuristic gates forbidden | G11, G12, independent_judge, human_audit | — | — |
| May authorize pruning | **True** | False | False |
| Unresolved hard ambiguity forbidden | **True** | False | False |

Incomplete forests are unchanged. UNKNOWN and unobserved candidates stay live.
No path widens the grammar or adds a full-vocabulary fallback. If certified
closure leaves one live candidate, the existing singleton path commits it
without a neural forward or learned ranking. The programmatic configuration
fields have behavior-preserving defaults and add no model parameters.

## Persisted schemas and invalidation

All persisted contracts are exact-version only; there is no implicit migration
or permissive compatibility path.

| Contract | Accepted schema version |
| --- | --- |
| `GoalConstraintV1` | `goal_constraint/v1` |
| `CompiledGoalConstraintSetV1` | `compiled_goal_constraint_set/v1` |
| `GoalConstraintEvaluationV1` | `goal_constraint_evaluation/v1` |
| `GoalVerifierProfileV1` | `goal_verifier_profile/v1` |
| `GoalTerminalEvidenceV1` | `goal_terminal_evidence/v1` |
| `GoalSupportResultV1` | `goal_support_result/v1` |
| `GoalActionEvidenceV1` | `goal_action_evidence/v1` |
| `DomainObstructionCoreV1` | `domain_obstruction_core/v1` |
| `GoalDomainAdequacyReportV1` | `goal_domain_adequacy_report/v1` |

An unknown version, unknown field, digest mismatch, or identity mismatch is
rejected; old records must be replayed and re-emitted under the new pinned
profile rather than silently upgraded.

Generated strict schemas are under `src/slm_training/resources/`:

- `compiled_goal_constraint_set.schema.json`;
- `goal_verifier_profile.schema.json`;
- `goal_terminal_evidence.schema.json`;
- `goal_support_result.schema.json`;
- `goal_action_evidence.schema.json`;
- `domain_obstruction_core.schema.json`;
- `goal_domain_adequacy_report.schema.json`.

Digests bind the problem/request, constraint payload, pack/contract/grammar/
tokenizer/canonicalizer identities, required gate/evaluator/metric identities,
solver bounds, and implementation/schema versions. Any change prevents stale
cache or replay reuse.

### C. Operation permissions (closed)

Legend: ✅ allowed · ⚠ diagnostic only · ❌ forbidden · — not applicable

| Evidence class | Compile | Diagnose | Rank | Train | Evaluate | Prune |
| --- | --- | --- | --- | --- | --- | --- |
| Structured exact facts (request/pack/verification) | ✅ | ✅ | — | — | ✅ | ✅ under `production_exact` |
| Prompt-derived requirements | ✅ advisory | ✅ | ✅ soft | ⚠ features only | ✅ eval | ❌ |
| Predicted / retrieved plans | — | ✅ | ✅ soft | ⚠ never hard labels | ✅ eval | ❌ |
| Eval / oracle fixtures | ✅ eval partition | ✅ | — | ❌ | ✅ | ❌ |
| G0–G12 gates (deterministic) | — | ✅ | — | — | ✅ | ✅ when exact + hard profile |
| G11 / G12 / judge / human | — | ⚠ advisory | ⚠ | ❌ hard labels | ✅ eval | ❌ |
| Aggregate meaningful metrics | — | ⚠ telemetry | ⚠ | ❌ sole label | ✅ ship ladder | ❌ |
| Learned rankers / preferences | — | ⚠ | ✅ | ✅ | ✅ | ❌ |
| VSS structural `SupportCertificate` | — | ✅ | ✅ soft | — | ✅ | ✅ structural closure only |
| PGS goal sidecar | — | ✅ | — | ⚠ `production_exact` only | ✅ | ✅ `certified` + `production_exact` |

### D. Decode modes (`goal_support_mode`)

| Mode | Default | Mutates forest? | Closure invoked? | Profile required |
| --- | --- | --- | --- | --- |
| `off` | **Yes** | No | No | — |
| `diagnostic` | No | **No** — decode-identical to off | Yes — counters only | `production_exact` binding; eval/advisory **forbidden** |
| `certified` | No | **Yes** — `goal_support_certified_prune` | Yes | `goal_support_profile_mode=production_exact`, `compiler_decode_mode ≠ off` |

`goal_support_mode` is **not** in `CONSTRAINT_WEAKENING_LEVERS` (`levers.py`).

### E. Trainability (`goal_support_trainability_table`)

| Profile | Trainable labels | Promotion |
| --- | --- | --- |
| `production_exact` | Only with hard-tier replay evidence | Allowed when materializer gates pass |
| `evaluation_oracle` | Never | Never |
| `advisory_diagnostic` | Never (`hard_good_bad_labels_forbidden`) | Never |

---

## No-escalation laws and fail-closed paths

1. **Unknown never prunes.** `SupportVerdict.UNKNOWN`, partition `unknown`, and
   `unobserved` never authorize candidate removal — including replay failure
   (sidecar replay does not upgrade to `UNSUPPORTED`).
2. **Advisory never prunes.** Prompt/oracle/advisory constraints and
   `advisory_diagnostic` / `evaluation_oracle` profiles cannot call
   `exact_goal_closure` for forest mutation.
3. **Profile staleness fails closed.** `validate_profile_against_constraint_set`
   rejects digest mismatch between profile and compiled constraint set.
4. **Identity binding.** `request_production_digest`, `pack_identity_digest`, and
   constraint-set digest must match across compile → verify → replay → decode binding.
5. **Mandatory SKIP.** Required constraints marked unavailable → terminal
   `UNAVAILABLE` → no prune authority.
6. **Certified singleton bypass (I2).** When certified closure leaves exactly one
   survivor, commit with `forwards_count == 0` — no ranking downgrade.
7. **Diagnostic drift guard.** `diagnostic` mode must preserve forest identity
   (`diagnostic_domain_digest_changes=0` in H02 fixtures).
8. **Privacy/redaction.** `GoalTerminalEvidenceV1` forbids raw prompts, OpenUI
   source, witness text, secrets, timestamps, PIDs, model scores;
   `redact_bounded_string` bounds field length.

---

## Threat model

| Threat | Mitigation | Residual risk |
| --- | --- | --- |
| **Authority injection** — soft source compiled as hard | Closed `source_to_authority_matrix`; `may_prune` law; profile partition enforcement | New source kinds require compiler table extension + review |
| **Free-text masquerading** — NL in evidence fields | Forbidden field set on `GoalTerminalEvidenceV1`; secret patterns stripped | Heuristic redaction ≠ cryptographic erasure |
| **Gold leakage** — target AST in compile inputs | `GoldTargetAccessError`; `_GOLD_LEAK_KEYS` guard in compiler | Mis-labeled fields in new callers |
| **Mandatory SKIP bypass** — omitting required checks | Terminal `UNAVAILABLE` blocks prune; unresolved hard ambiguity forbidden under `production_exact` | Caller skips profile validation |
| **Stale identity / cache** — replay under wrong digest | Digest-bound profiles, sidecars, certificates; replay validators | Shared cache key omitting profile digest |
| **Shared mutable traces** — cross-request contamination | Query-local `GoalTerminalEvidenceTrace`; immutable frozen models | Future mutable trace stores |
| **Tampered cert / sidecar / core** — forged replay artifacts | SHA-256 digests; `replay_goal_support_result`; Lean/Python mapping tests | Out-of-band artifact substitution |
| **Omitted action evidence** — cap hides support | `unobserved` partition; cap → `coverage_unknown`, not inadequacy | Operators misread partial coverage |
| **Timeout-as-unsupported** — budget stop treated as proof | Budget exhaustion → `UNKNOWN` / `coverage_unknown`; obstruction emission blocked | Misconfigured bounds in production |
| **Raw-content exposure** — prompts in logs | Redaction + forbidden fields; H02 `raw_content_redaction` fixture | Downstream loggers bypass redaction |
| **Diagnostic drift** — telemetry path mutates decode | Tests pin forest parity for `diagnostic`; certified isolated | ONNX/backend divergence |
| **Certified misuse** — prune without production profile | `build_goal_support_decode_binding` hardcodes production profile; mode validation in decode | Manual API calls bypassing decode seam |

---

## Profile, compatibility, and performance bounds

| Concern | Policy |
| --- | --- |
| **Production profile** | `production_exact` + `compiler-hard` or `verifier-hard`; required constraint ids ⊆ hard partition |
| **Eval profile** | `evaluation_oracle` + `evaluation-only`; materialize only with `allow_evaluation_diagnostic=True` |
| **Advisory profile** | `advisory_diagnostic`; scoring/diagnostics only |
| **Schema migration** | Versioned schemas (`goal_verifier_profile/v1`, `goal_support_result/v1`, …); `scripts/sync_goal_support_schema.py` |
| **Checkpoint defaults** | `goal_support_mode=off`; certified requires explicit opt-in |
| **Backend limits** | Bounded by `SolverBounds` (tokens, nodes, depth, backtracks, verifier calls) — same VSS carrier |
| **Performance** | H02 fixture wall time ≈30 ms / 15 fixtures CPU — not a latency SLA |
| **Privacy** | Digest-bound evidence; bounded strings; no raw OpenUI in sidecars |
| **Zero-forward singleton** | Certified arm fixture `certified_singleton_zero_forward`: `forwards=0`, replay ok |

---

## Research lineage and source fidelity

PGS composes prior art **without** reproducing Astra-scale theorem proving or
global semantic synthesis. Fidelity labels match
[`research-lineage.md`](research-lineage.md).

| Source cluster | Fidelity | PGS use | Boundary |
| --- | --- | --- | --- |
| **SyGuS / SemGuS** (Alur et al.; SemGuS 2021+) | **Adjacent** | Capability reports (`SyGuSCapabilityReportV1`) inspire interoperability vocabulary; constraints compile to typed predicates, not SyGuS-IF emission | No `(check-synth)` conformance; see SGS-008 notes |
| **CEGIS / LLM-Modulo** | **Adapted boundary** | Counterexample → refinement loop informs obstruction cores and domain adequacy diagnosis; neural CEGIS training not implemented | Deduction vs decision split only |
| **PCC / certifying algorithms** (Necula; Appel) | **Adapted** | `SupportCertificate` + sidecar replay as proof-carrying evidence; not a general PCC compiler | Finite-domain, bounded witnesses only |
| **VSS** ([`verified-scope-solver.md`](verified-scope-solver.md)) | **Faithful (mechanism)** | `EnumerativeSupportOracle`, `exact_closure`, verdict algebra | Goal layer adds constraint-aware terminal verifier |
| **SPV / semantic planning** ([`semantic-planning-valid-state.md`](semantic-planning-valid-state.md)) | **Adapted** | `SemanticPlanV1` input to terminal verifier; plans never proof objects | Compiler legality remains sole hard authority |
| **LDI / DecisionEventV2** ([`local-decision-interventions.md`](local-decision-interventions.md)) | **Adapted** | `materialize_goal_support` maps partitions → objective views | No new decision schema |
| **OpenAI reasoning walkthroughs** (o-series chain-of-thought releases) | **Adjacent** | Motivation for bounded, replayable step evidence | No API reasoning trace import; no hidden CoT authority |
| **Formal proof manuscripts / math overview** (community LLM+Lean lines) | **Adjacent** | Lean `GoalSupport.lean` proves finite partition laws only | Not Astra theorem reproduction; no NL→Lean pipeline |
| **G0–G12 verifier stack** ([`verifier-stack.md`](verifier-stack.md)) | **Adapted** | Terminal composition in `OpenUIGoalVerifier` | Gates are necessary, not sufficient for global semantic proof |

**Positive fidelity claims (what we do reproduce):**

- Replayable witness / obstruction certificates over a **finite** declared domain.
- Representation-bound evidence (digests, redaction, forbidden raw fields).
- Bounded search with explicit dependence on profile, bounds, and compiler versions.
- Explicit non-dependence on unconstrained NL semantics for prune authority.

**Explicit non-goals:** Astra-scale theorem reproduction; global synthesis;
neural residual heads; symmetry breaking; ECC; quantization; unconstrained
semantic planning as prune authority.

---

## Evidence interpretation (PGS-H02)

Measured fixture campaign (diagnostic claim class):

| Artifact | Path |
| --- | --- |
| Markdown summary | [`proof-carrying-goal-support-fixture-results.md`](proof-carrying-goal-support-fixture-results.md) |
| Machine-readable | [`proof-carrying-goal-support-fixture-results.json`](proof-carrying-goal-support-fixture-results.json) |

Headline counters (production-exact arm): `false_hard_prune_count=0`,
`support_certificate_replay_failure_count=0`,
`evaluation_oracle_called_certified_closure=False`,
`deterministic_rerun_digest_match=True`, primary metric
`exact_action_coverage_rate≈0.978`.

**Why no MODEL_CARD / ship:** H02 is a 15-fixture word-tree wiring campaign on
CPU with `claim_class=diagnostic` and `promotion=false`. It validates
classifier tables, replay, obstruction emission, arm isolation, and decode
parity — not model quality, semantic generalization, production latency, or
ship gates.

**Honest limitations recorded in H02:**

- `bound_exhaustion` certified arm may forward without removal under tight bounds
  (`certified_singleton_forward_violations=1`) while falsifiers stay zero.
- Partial forests, cap exclusion, and advisory/eval fixtures correctly land in
  `coverage_unknown` — not retuned.
- Aggregate `falsifier_holds=True` reflects bounded-classification honesty, not
  a production defect.

**Successor hypotheses (documentation only — not expanded here):**

- Frontier-scale domain adequacy under real pack/compiler coverage (not word-tree fixtures).
- Matched production-exact vs structural-only arms on trained checkpoints.
- Tighter bound-exhaustion policy for certified decode (without weakening UNKNOWN law).

---

## Implementation ↔ design notes for reviewers

1. **`verified-scope-solver.md`** predates PGS-G03 decode integration; its header
   still states support oracle is not wired into decode by default. PGS adds an
   **opt-in** certified/diagnostic seam (`goal_support_mode`, default `off`).
2. **`generation_request` constraints are hard but not prune-eligible** — only
   `pack_contract` and `verification_requirement` satisfy independent exact prune
   sources.
3. **Dual status vocabularies** — terminal `ACCEPT/REJECT/UNAVAILABLE` vs
   constraint `PASS/FAIL/UNKNOWN/NOT_APPLICABLE`; do not collapse to gate OK.
4. **Lean vs Python obstruction modes** — subset-minimal vs sound overapprox; see
   PGS-F01 proof boundary.

---

## Verification

```bash
python -m scripts.repo_policy
python -m scripts.verify_decode_invariants
python -m scripts.verify_version_stamps --check
.githooks/check-changed
```

Structural drift:

```bash
make -C src/leverproof_lean test
pytest -q tests/test_formal/test_goal_support_mapping.py
pytest -q tests/test_dsl/test_goal_support_adversarial.py tests/test_dsl/test_goal_support_decode_g03.py
```

Fixture evidence regeneration (diagnostic only):

```bash
python -m scripts.run_goal_support_domain_adequacy --mode fixture \
  --out-dir outputs/runs/pgs_h02_goal_support_domain_adequacy \
  --docs-out docs/design/proof-carrying-goal-support-fixture-results.json \
  --claim-class diagnostic
```
