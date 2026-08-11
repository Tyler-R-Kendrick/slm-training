# Reverse mathematics, computability, and canonical evidence ownership

**Status:** INTEG-04 harness four-axis ledger adapters (SLM-529); INTEG-03 campaign formal/empirical linkage (SLM-556); KERN-11 fixture-level proof-trace refinement (SLM-539); INTEG-01 canonical proof-trace projection (SLM-528); HARN-09 proposition-preserving self-healing (SLM-560); HARN-08 conservative RM labeling / anti-overclaim (SLM-537); HARN-06 constructivization + counterexample validators landed (SLM-546); HARN-07 quantitative-bound extraction (SLM-547); HARN-05 reversal/bidirectional validation (SLM-545); HARN-04 assumption ablation (SLM-544); EVID-03 four-axis ledger (SLM-519); HARN-03 runner/replay/report (SLM-536); HARN-02 schemas (SLM-527); profile + parity (HARN-01 / SLM-520); owner map (EVID-01 / SLM-515)  
**Base SHA:** `980aa465223adf7822f82930550c92b9240333ca` (`origin/main` at audit)  
**Machine-readable map:** [`src/slm_training/resources/revmath_owner_map.json`](../../src/slm_training/resources/revmath_owner_map.json)  
**Harness parity matrix:** [`src/slm_training/resources/revmath_harness_parity.json`](../../src/slm_training/resources/revmath_harness_parity.json)  
**ADR:** [adr-revmath-reasoning-profile.md](adr-revmath-reasoning-profile.md)  
**Verified by:** `python -m scripts.verify_revmath_owners` · `python -m scripts.verify_revmath_harness_parity`

> **Goal law:** bound by [decode-invariants.md](decode-invariants.md). Reverse mathematics here means *which assumptions suffice for which conclusion* — not a second training stack.

## Decision summary

Reverse mathematics in this repository is a typed **`reasoning/revmath` profile** over the existing campaign, formal-preflight, formal-object, telemetry, witness, disposition, replay, repair, version, and agent-surface owners. It is **not**:

- a parallel trainer or model-build path;
- a parallel proof stack or Lean orchestrator;
- a parallel evidence store or campaign registry;
- a parallel subprocess runner or version registry.

Every revmath run **must** bind to `ExperimentCampaignV1`, emit or consume evidence only through the canonical envelopes below, and respect bounded execution via `run_bounded_process` / `run_formal_process`. Timeout, skipped replay, incomplete coverage, and missing tools map to **unknown/inconclusive** — never semantic refutation.

## HARN-05 bidirectional reversal

Owner: `harnesses/reasoning/revmath/reversal.py` (+ `ReversalPlugin` in `plugins.py`).
Given base `B`, principle `A`, theorem `T`, both `B+A ⊢ T` and `B+T ⊢ A` are
generated and checked with separate evidence. Reports label **equivalent** only
when both directions witness over the same base theory and explicit
`RmInterpretationStatus` / `InterpretationStatus` (`explicit_reversal` or
`explicit_interpretation`). One-way implications, timeouts/unsupported
(unknown), strength mismatches, hidden stronger assumptions, and changed
propositions are preserved explicitly and never promoted to equivalence.
Toy fixtures live under `resources/revmath/fixtures/reversal_*.{task,meta}.json`.


## HARN-06 constructivization and formal counterexamples

Owners: `harnesses/reasoning/revmath/constructivization.py` +
`counterexample.py` (plugins in `plugins.py`).

**Constructivization modes:** `bounded_finite_analogue` (EVID-04 `bound.*` AST
id placeholder), `explicit_witness_producing`, `oracle_relative`, and
`documented_nonconstructive_remainder`. Every constructivized claim records an
explicit `weakening_description` and a distinct `constructivized_statement_sha256`
— it must not masquerade as the original classical theorem.

**Counterexamples:** validators check the exact weakened proposition +
`allowed_assumption_ids` and require a replayable `finite_model` or
`computable_trace`. Proof-search failure / missing model stays `unknown`
(KERN-01) — never inferred impossibility. Refutation only when
`independently_checked` against the weakened theorem.

Toy fixtures: `resources/revmath/fixtures/constructivization_*.{task,meta}.json`
and `counterexample_*.{task,meta}.json`.

## HARN-08 conservative labeling and anti-overclaim

Owner: `harnesses/reasoning/revmath/labeling.py` (+ `labeling_notes_for_tasks` in
`report.py`).

**Labeling states:** `practical_computability_only`, `candidate_upper_bound`,
`interpreted`, `reversed_equivalent`, `counterexample_known`, `unclassified`.

**Big-Five equivalence** (`RCA0` / `WKL0` / `ACA0` / … as `reversed_equivalent`)
requires an explicit package: `base_theory_id`, `coding_id`,
`interpretation_status ∈ {explicit_interpretation, explicit_reversal}`,
`forward_evidence_sha256`, `reversal_evidence_sha256`, and evidence kinds
`interpretation_package` + `forward_implication` + `reversal`.

**Anti-overclaim:** `#print axioms` alone, Lean compile success / no
project-specific axioms, finite bounded analogues without a real interpretation,
classical existence, and empirical claims cannot yield Big-Five labels.
Assumption ablation stays `analysis_kind=rm_inspired_assumption_minimization`
and is never upgraded to genuine reverse mathematics without the package above.

**Reports** must distinguish RM-inspired assumption minimization from genuine
reverse mathematics (`report_labeling_disclaimer` /
`default_analysis_kind_for_task`).

Fixtures: `resources/revmath/fixtures/labeling_*.claim.json` (positive +
negative).

## INTEG-01 canonical proof-trace projection

Owner: `formal/proof_trace.py` (design:
[integ-01-canonical-proof-trace.md](integ-01-canonical-proof-trace.md)).

Projects `ReplayBundleV1`, decode stats / mechanism activation, and optional
verifier witnesses into `canonical_proof_trace/v1`, reusing KERN-06 Event/cost
models for observed work. Unobserved fields stay explicit. Sealed traces never
set `claims_abstract_refinement` — KERN-11 proves refinement separately. No
second runtime recorder.

## KERN-11 fixture-level proof-trace refinement

Owner: `formal/trace_refinement.py` (+ Lean `LeverProofLean.ProofTraceRefinement`).

Sealed INTEG-01 `CanonicalProofTraceV1` events are checked against abstract
Judgment / CompleteDomain / EventTrace / DecodeInvariants owners for the
supported fixture subset (`bundle_stats_mechanism`). Successful checks bind
`formal_authority/v2`. Fixture/subset-scoped only — not full PyTorch semantics
or physical latency equivalence. Design:
[kern-11-proof-trace-refinement.md](kern-11-proof-trace-refinement.md).

## KERN-12 practical computability vocabulary

Owner: `formal/computability_classification.py` (+ Lean
`LeverProofLean.ComputabilityClassification`).

Production classes (`finite_decidable`, `primitive_recursive`, `bounded_search`,
`total_recursive`, `semidecidable`, `co_semidecidable`, `oracle_relative`,
`classical_propositional`, `noncomputable_existence`, `unclassified`) each have
an evidence contract, permitted implications, and non-implications. HARN-08
`practical_class` and KERN-10 / EVID-03 ledger labels consume this vocabulary
(legacy alias `classical_noncomputable_existence` → `noncomputable_existence`).
`rm_research:*` requires an explicit interpretation package — `#print axioms`
alone never classifies a theorem as `RCA0` / `WKL0` / ….

Design: [kern-12-computability-classification.md](kern-12-computability-classification.md).

## HARN-09 proposition-preserving self-healing

Owner: `harnesses/reasoning/revmath/self_healing.py` (extends `semantic_repair`
lineage shape; does not fork distill/OpenUI repair).

**May modify** (parity `self_healing.may_modify` /
`REVMATH_MUTABLE_REPAIR_KNOBS`): tactic sequence, proof-search strategy,
retrieval choices, type-equivalent theorem selection, serialization/bridge
generation, temporary decomposition, and resource allocation *inside* the
locked budget envelope.

**Must freeze** (`REVMATH_FROZEN_REPAIR_FIELDS`): proposition, base theory /
allowed assumptions, theorem direction, corpus membership, experiment arms,
budgets/stopping rules, judge definitions, promotion gates, authority policy,
expected counterexample semantics.

The controller consumes typed failure/witness data, proposes bounded knob
deltas, re-runs the *exact* locked task via the HARN-03 runner after every
attempt, records immutable `RevmathRepairAttemptV1` lineage (before/after
artifacts, knob digests, search budget, checker/evidence refs, parent id),
invalidates stale proof digests on re-check, and terminates with
`cycle_detected` / `budget_exhausted` → unknown rather than looping.
Successful repair proves the original `task_identity_digest`; mutation tests
reject weaken-theorem / add-axiom / increase-budget / change-corpus paths.

## Owner map (one canonical owner per surface)

| Surface id | Kind | Canonical owner | Key symbols | Tier / role |
| --- | --- | --- | --- | --- |
| `formal_preflight_schema` | schema | `autoresearch/schemas.py` | `FormalPreflightV1`, `FormalClaimV1`, `FormalObligationV1` | compiler-hard contract |
| `formal_preflight_execution` | formal preflight | `autoresearch/formal.py` | `run_formal_preflight`, `validate_formal_preflight_artifact` | bounded Lean + audit |
| `formal_preflight_seam` | command | `scripts/autoresearch.py` | `cmd_formalize` | campaign-bound artifact write |
| `formal_object_schema` | schema | `formal/objects.py` | `FormalObjectV1`, exporters | portable object contract |
| `formal_object_loop` | formal object | `formal/loop.py` | `close_formal_loop`, `loop_requires_multi_backend` | multi-prover close |
| `formal_checkers` | checker | `formal/checkers.py` | `run_checkers`, `FormalProjectLock` | independent backends |
| `formal_objects_cli` | command | `scripts/verify_formal_objects.py` | `main` | export/verify CLI |
| `bounded_process` | execution | `harness_core/bounded_process.py` | `run_bounded_process` | hard run-cap subprocess |
| `experiment_campaign` | campaign | `autoresearch/experiment_campaign.py` | `ExperimentCampaignV1` | preregistered manifest |
| `decode_stats_record` | telemetry | `models/decode_stats.py` | `DecodeStatsRecordV1` | persisted decode evidence |
| `mechanism_activation` | activation | `models/decode_stats.py` | `MechanismActivationV1` | mechanism envelope |
| `replay_bundle` | replay | `runtime/telemetry/replay_bundle.py` | `ReplayBundleV1` | offline replay bundle |
| `solver_replay` | replay | `dsl/solver/replay.py` | `solver_replay_violations` | trace validation |
| `verifier_witness` | witness | `evals/semantic_failure.py` | `VerifierWitnessV1` | typed verifier witness |
| `mechanism_disposition` | disposition | `harnesses/experiments/mechanism_disposition_report.py` | `MechanismDispositionRecordV1` | adopt/supersede |
| `semantic_repair` | repair | `harnesses/distill/semantic_repair.py` | `SemanticRepairRecordV1`, `RepairResidualV1` | repair + residual |
| `version_registry` | version | `resources/versions.json` | (registry file) | component versions |
| `agent_surface_parity` | agent surface | `scripts/verify_agent_surfaces.py` | `Obligation`, `check` | harness law matrix |
| `reasoning_harness_parent` | profile parent | `harnesses/reasoning/__init__.py` | `run_reasoning_bench`, `AbstractWarmupCampaignV1` | G4 reasoning harness |
| `revmath_schemas` | schema | `harnesses/reasoning/revmath/schemas.py` | `RevmathTaskV1`, `RevmathResultV1`, `RevmathRepairRecordV1`, … | HARN-02 typed contracts |
| `revmath_runner` | runner | `harnesses/reasoning/revmath/runner.py` | `run_revmath_task`, `RevmathRunRecord` | HARN-03 bounded deterministic runner |
| `revmath_plugins` | plugin | `harnesses/reasoning/revmath/plugins.py` | `RevmathTaskPlugin`, `resolve_plugin` | HARN-03 task-kind seam |
| `revmath_assumption_ablation` | task_validator | `harnesses/reasoning/revmath/assumption_ablation.py` | `generate_ablation_candidates`, `evaluate_ablation_lattice`, `audit_hidden_reintroduction` | HARN-04 finite-lattice ablation; minimality scoped to explored set |
| `revmath_constructivization` | task_validator | `harnesses/reasoning/revmath/constructivization.py` | `generate_constructivization_variant`, `evaluate_constructivization`, `assert_not_masquerading` | HARN-06 constructivization modes; no masquerade |
| `revmath_counterexample` | task_validator | `harnesses/reasoning/revmath/counterexample.py` | `check_counterexample_against_theorem`, `evaluate_counterexample` | HARN-06 checked finite/computable counterexamples; search≠refutation |
| `revmath_quantitative_bound` | task_validator | `harnesses/reasoning/revmath/quantitative_bound.py` | `extract_quantitative_bound`, `QuantitativeBoundReportV1` | HARN-07 theorem-derived bounds via EVID-04; no invented rates |
| `revmath_labeling` | task_validator | `harnesses/reasoning/revmath/labeling.py` | `validate_label_claim`, `anti_overclaim_reasons`, `RmLabelClaimV1` | HARN-08 conservative Big-Five labeling; anti-overclaim |
| `proof_trace_refinement` | refinement | `formal/trace_refinement.py` | `check_trace_refinement`, `emit_refinement_certificate`, `bind_refinement_authority` | KERN-11 fixture-subset refinement to abstract semantics |
| `canonical_proof_trace` | projection | `formal/proof_trace.py` | `CanonicalProofTraceV1`, `project_runtime_evidence`, `match_proof_trace_to_evidence` | INTEG-01 project existing evidence into canonical proof trace |
| `formal_computability_classification` | schema | `formal/computability_classification.py` | `CLASS_CONTRACTS`, `validate_classification_claim`, `ClassificationClaimV1` | KERN-12 practical computability vocabulary + evidence contracts |
| `revmath_self_healing` | repair | `harnesses/reasoning/revmath/self_healing.py` | `run_self_healing`, `propose_repair_attempts`, `RevmathRepairSessionV1` | HARN-09 proposition-preserving self-healing |
| `revmath_failure_witness` | witness_adapter | `harnesses/reasoning/revmath/failure_witness.py` | `route_revmath_failure`, `RevmathFailureEvidenceV1`, `feed_repair_session_to_disposition` | INTEG-05 failure→VerifierWitnessV1 + disposition feedback |
| `revmath_replay` | replay | `harnesses/reasoning/revmath/replay.py` | `build_revmath_replay_bundle` | HARN-03 ReplayBundleV1 composition |
| `revmath_report` | report | `harnesses/reasoning/revmath/report.py` | `build_revmath_report` | HARN-03 typed reports |
| `campaign_formal_evidence` | campaign_evidence_link | `autoresearch/campaign_formal_evidence.py` | `CampaignFormalEmpiricalSplitV1`, `link_four_axis_and_revmath`, `decision_support_report` | INTEG-03 optional formal↔empirical refs on CampaignResultV1 |
| `revmath_profile_binding` | campaign_binding | `harnesses/reasoning/revmath/profile_binding.py` | `run_revmath_profile` | HARN-10 ExperimentCampaignV1 lock + evidence |
| `revmath_profile_cli` | command | `scripts/run_revmath_profile.py` | `main` | HARN-10 locked profile CLI |
| `continuous_formal_promote` | command | `scripts/run_autotrain_continuous.py` | `ensure_promote_formal_preflight` | promotion gate |

Design references: [formal-autoresearch.md](formal-autoresearch.md), [formal-objects-multi-prover.md](formal-objects-multi-prover.md), [experiment-campaign-governance.md](experiment-campaign-governance.md), [repository-ownership-map.md](repository-ownership-map.md).

Cross-links to the global ownership map (`ownership_map.json` subsystems): where a row names `ownership_map_subsystem`, downstream work must extend that subsystem rather than introduce a shadow module.

## Extension seams (downstream work)

| Issue | Extends | New owner? | Rule |
| --- | --- | --- | --- |
| EVID-03 / SLM-519 | `formal_preflight_schema` | no | **done** — `FormalPreflightV1.four_axis_ledger` + `FormalPreflightFourAxisLedgerV1` |
| EVID-04 / SLM-525 | `formal_preflight_schema` → `bound_ast_schema` | yes (`formal/bound_ast.py`) | **done** — safe symbolic bound AST + exact Fraction evaluator; registry ids cited by ledger |
| EVID-06 / SLM-526 | `formal_object_schema` | no | **done** — `formal_authority/v2` adapts `FormalObjectV1` + `FormalPreflightV1` |
| HARN-01 / SLM-520 | `reasoning_harness_parent` | no | register `reasoning/revmath` profile + parity matrix |
| HARN-09 / SLM-560 | `semantic_repair` | yes (`self_healing.py`) | **done** — proposition-preserving repair controller; lineage extends SemanticRepairRecordV1 without forking distill |
| INTEG-03 / SLM-556 | `experiment_campaign` | yes (`campaign_formal_evidence.py` + profile_binding) | **done** — optional formal/empirical split on CampaignResultV1; decision-support report; locks unchanged |
| HARN-10 / SLM-548 | `experiment_campaign` | yes (`profile_binding.py`) | **done** — lock ExperimentCampaignV1 via CampaignStore; formal/evidence/disposition/replay through canonical owners; `run_revmath_profile` CLI |

## Prohibited duplicate owners

Agents **must not** add modules that claim semantic authority for:

| Concern | Canonical owner | Blocked shadow patterns |
| --- | --- | --- |
| Campaign manifest / arms / gates | `experiment_campaign.py` | `revmath.*campaign`, `RevMathCampaign` |
| Evidence persistence | `decode_stats.py` (`DecodeStatsRecordV1`) | `revmath.*evidence`, `RevMathEvidenceStore` |
| Proof preflight execution | `autoresearch/formal.py` | `revmath.*proof`, `RevMathProofStack` |
| Orchestration / trainer | `harnesses/reasoning/` + campaign owners | `revmath_trainer`, `revmath_orchestrator` |

`verify_revmath_owners` scans `src/` and `scripts/` for these shadow patterns and fails closed.

## INTEG-03 — Campaign formal/empirical linkage (SLM-556)

Optional `CampaignResultV1.formal_empirical` (`campaign_formal_empirical_split/v1`)
links FormalAuthorityV2, four-axis ledgers, revmath reports/results, bound ASTs,
and an optional KERN-11 refinement-trace digest slot. Formal and empirical
statuses are independent: proved preflight + failed empirical (or weak formal +
successful empirical) must not conflate. Query via `decision_support_report`.
Design note: [`integ-03-campaign-formal-evidence.md`](integ-03-campaign-formal-evidence.md).


## INTEG-06 — Adversarial end-to-end acceptance (SLM-573)

Release-blocking matrix over existing EVID-11 / KERN-07/08/11 / HARN-09/11 /
INTEG-01/02/03/05 gates. Orchestrator:
`formal/integ06_acceptance.py`; verify:
`scripts/verify_integ06_adversarial_acceptance.py`; design:
[`integ-06-adversarial-acceptance.md`](integ-06-adversarial-acceptance.md).
Positive controls pass; adversarial mutations fail at the named gate; failures
never become destructive semantic authority.

## INTEG-05 — Failure/repair routing through witness & disposition (SLM-571)

Adapter owner: [`harnesses/reasoning/revmath/failure_witness.py`](../../src/slm_training/harnesses/reasoning/revmath/failure_witness.py).

Revmath-localized failures (missing/necessary assumption candidate, failed reversal direction, nonconstructive remainder, quantitative bound unavailable, checked counterexample, checker/tool/environment error, blocked semantic mutation) are sealed into the existing [`VerifierWitnessV1`](../../src/slm_training/evals/semantic_failure.py) taxonomy. Unknown/unlocalized failures stay explicitly unresolved. Bounded repair suggestions touch only HARN-09 mutable knobs and never change proposition/theory/campaign authority. Immutable HARN-09 before/after proof refs feed [`MechanismDispositionRecordV1`](../../src/slm_training/harnesses/experiments/mechanism_disposition_report.py) as `retain_diagnostic` / `inconclusive` / `blocked` — never adopt/promote. A diagnostic witness cannot turn failed/unknown into proof success.

## Migration rules

1. **Read adapters, write canonical owners.** Historical artifacts remain readable through explicit adapters (EVID-06); no silent reinterpretation of persisted schema versions.
2. **Extend schemas in place** when the concern is already owned (`FormalPreflightV1`, `FormalObjectV1`, `ExperimentCampaignV1`).
3. **New submodule only** when dataclasses cannot live in the owner module without violating the sibling-file rule documented in `ownership_map.json` — and then register in `revmath_owner_map.json` + global map in a parent PR.
4. **Profile registration** (`reasoning/revmath`) may add discovery hooks under `harnesses/reasoning/` (HARN-01) but must not fork campaign, evidence, or proof execution.
5. **Version stamps:** doc-only owner freeze uses `governance.revmath_owners` with `no-bump:` history; code changes to watched paths require a component bump.

## Revmath profile (`reasoning/revmath`)

The profile id is **`reasoning/revmath`**. It parameterizes the G4 reasoning harness ([`harnesses/reasoning/`](../../src/slm_training/harnesses/reasoning/)) to:

- declare reverse-mathematics / computability analysis goals;
- attach formal preflight obligations and four-axis ledger fields (EVID-03+);
- route all measured results through `ExperimentCampaignV1` and honest ship gates;
- consume witnesses, activations, replay bundles, and dispositions from the owners above.

**Registration seam (HARN-01):** [`harnesses/reasoning/profiles.py`](../../src/slm_training/harnesses/reasoning/profiles.py) registers `reasoning/g4` (default) and `reasoning/revmath` (opt-in, `task_semantics_ready=True` after HARN-03). `resolve_profile(None)` keeps the historical default — discovering/configuring revmath does not alter G4 bench or warmup behavior.

Task schemas (HARN-02), runner/replay/report (HARN-03), assumption-ablation (HARN-04), reversal (HARN-05), constructivization/counterexample (HARN-06), quantitative-bound extraction (HARN-07), conservative labeling / anti-overclaim (HARN-08), proposition-preserving self-healing (HARN-09), campaign-locked profile CLI binding (HARN-10), and fixture-corpus packaging (HARN-11) are present. This document owns **authority boundaries and parity obligations**; missing parity rows are explicit blockers in the matrix, never silent fallbacks.

## Harness / self-healing parity matrix (HARN-01)

Machine-checkable obligations live in [`revmath_harness_parity.json`](../../src/slm_training/resources/revmath_harness_parity.json) and are certified by `python -m scripts.verify_revmath_harness_parity`.

| Category | Role |
| --- | --- |
| commands / schemas / resources | CLI + typed contracts + committed inputs |
| campaign_binding / evidence / replay | No bypass of `ExperimentCampaignV1` or canonical envelopes |
| bounded_execution | `run_bounded_process` / formal preflight only |
| repair | Extend `SemanticRepairRecordV1`; honor freeze surface |
| versions / docs / agent_surfaces | Single version registry; design/ADR; INTEG-09 for skill surfaces |

Every row is either `status=present` with a live `required_owner` path, or `status=blocker` with a non-empty `blocker_issue`. Unknown statuses fail the verifier.

### Self-healing freeze surface

Declared in the parity resource (`self_healing`); implemented by HARN-09
(`harnesses/reasoning/revmath/self_healing.py`).

| May modify (within locked budget) | Must freeze |
| --- | --- |
| tactic sequence, proof-search strategy, retrieval choices, type-equivalent theorem selection, serialization/bridge generation, temporary decomposition, resource allocation inside the locked budget | proposition, base theory / allowed assumptions, theorem direction, corpus membership, experiment arms, budgets / stopping rules, judge definitions, promotion gates, authority policy, expected counterexample semantics |

`may_modify` and `must_freeze` must stay non-empty and disjoint — the verifier fails closed on overlap.

## Absorbed predecessor: HARN-12

**HARN-12** (*harness / self-healing parity audit for `reasoning/revmath`*) is **absorbed into HARN-01** and is no longer a separate Linear task.

| HARN-12 planned deliverable | Where it lives now |
| --- | --- |
| Obligation matrix across commands/schemas/resources/campaign/evidence/replay/bounded-exec/repair/versions/docs/agent surfaces | `revmath_harness_parity.json` + `verify_revmath_harness_parity` |
| Explicit blockers for missing seams (no silent fallback) | `status=blocker` + `blocker_issue` rows |
| Self-healing may-modify / must-freeze freeze | `self_healing` object in the parity resource |
| Discoverable profile registration without default drift | `harnesses/reasoning/profiles.py` |

Reason: a standalone HARN-12 would duplicate the registration seam certificate and force a second verifier stack; folding the audit into HARN-01 keeps one owner for profile discovery + parity.

## Absorbed predecessor: EVID-02

**EVID-02** (*machine-readable formal/evidence owner registry + duplicate-owner CI guard*) is **absorbed into EVID-01** and is no longer a separate Linear task.

| EVID-02 planned deliverable | Where it lives now |
| --- | --- |
| Canonical owner registry for formal/evidence surfaces | `revmath_owner_map.json` + table above |
| Duplicate-owner / shadow-module guard | `verify_revmath_owners.py` prohibited-pattern scan |
| Agent discoverability | design doc + ADR + CI certificate |
| Global `ownership_map.json` churn | deferred to parent integration agent per swarm contract |

Reason: a standalone EVID-02 would duplicate EVID-01's ADR and force premature global map edits; the focused map + verifier satisfies fail-closed acceptance without a parallel orchestration layer.


## HARN-02 schemas (SLM-527)

Typed contracts live under [`harnesses/reasoning/revmath/schemas.py`](../../src/slm_training/harnesses/reasoning/revmath/schemas.py):

| Type | Role |
| --- | --- |
| `RevmathTaskV1` | Task kinds + frozen proposition / base-theory / direction / corpus / budget / verifier-judge / campaign binding |
| `RevmathCorpusEntryV1` | Corpus entry wrapping a task with matching corpus identity |
| `RevmathResultV1` | Embeds `SolverJudgmentV1` (KERN-01) + proof/checker refs + `RevmathFourAxisAnalysisV1` |
| `RevmathRepairRecordV1` | Immutable before/after proof digests; only `self_healing.may_modify` knobs |
| `RevmathReportV1` | Campaign-bound report with closed judgment counts |

Malformed tasks fail closed. Repairs cannot alter proposition / assumption / campaign identity (`assert_repair_preserves_identity`). Unknown judgment payloads cannot be normalized into witnessed/refuted. Four-axis ledger fields on `FormalPreflightV1` are owned in place as `four_axis_ledger: FormalPreflightFourAxisLedgerV1` (EVID-03 / SLM-519), reusing HARN-02 `RevmathFourAxisAnalysisV1` vocabulary; results carry the analysis attachment.

### EVID-03 formal-preflight ledger (SLM-519)

| Field / type | Role |
| --- | --- |
| `FormalPreflightV1.four_axis_ledger` | Optional in-place ledger; historical v1 JSON without it stays readable |
| `FormalPreflightFourAxisLedgerV1.analysis` | `RevmathFourAxisAnalysisV1` (assumption / computability / resource_bounds / implementation_refinement) |
| `computability_classification` | Practical labels: `finite_decidable`, `bounded_search`, `total_recursive`, `semidecidable`, `oracle_relative`, `classical_noncomputable_existence`, `unclassified` |
| `rm_subsystem_id` + `rm_interpretation_status` | Genuine `RCA0`/`WKL0`/… labels require `explicit_reversal` or `explicit_interpretation` (never inferred from `#print axioms`) |
| `resource_bounds.bound_ast_id` | Stable id into EVID-04 registry (`formal/bound_ast.py` / `bound_ast_registry.v1.json`): `bound.finite_search.prefix_tree.v1`, `bound.finite_search.coarse.v1`, `bound.closure.live_upper.v1`, `bound.placeholder.pending_evid04.v1`; exact `Fraction` eval; no raw `eval` |
| `empirical_remainder_claim_ids` | Claims that remain empirical (e.g. wall-clock, neural quality) |
| `theorem_claim_kinds` | Fail closed if `wall_clock_latency` or `neural_quality` are asserted as theorem consequences |

Migration: `migrate_formal_preflight_v1` / `formal_preflight_payload` / `formal_preflight_sha256` in `autoresearch/formal.py`. Digests omit an unset ledger so historical artifacts keep their hashes.

### EVID-04 safe bound AST (SLM-525)

| Module / artifact | Role |
| --- | --- |
| [`formal/bound_ast.py`](../../src/slm_training/formal/bound_ast.py) | Typed AST (`ConstInt`/`ConstRat`/`Var`/`Add`/`Mul`/`Max`/`Min`/`Div`/`Floor`/`Ceil`/`SumOver`/`ProdOver`/`PrefixCumprodSum`/`Len` + `Le`/`Lt`/`Eq`), exact `Fraction` evaluator, pretty printer, canonical digest |
| `resources/formal/bound_ast_registry.v1.json` | Registered `bound.*` documents with digests; KERN-03 prefix-tree and KERN-04 black-box lower-bound ids resolve here |
| `resources/formal/bound_ast_parity_fixtures.v1.json` | Lean↔Python parity cases for search/closure/cost bounds |

Fail closed: no `eval`/`exec`, unknown variables, division by zero, code-like payloads, or unregistered `bound_ast_id` on proved/refuted resource-bounds axes.

## HARN-03 runner / replay / report (SLM-536)

| Module | Role |
| --- | --- |
| [`runner.py`](../../src/slm_training/harnesses/reasoning/revmath/runner.py) | Bounded `run_revmath_task`; maps process + plugin evidence through `SolverJudgmentV1` |
| [`plugins.py`](../../src/slm_training/harnesses/reasoning/revmath/plugins.py) | Task-kind plan/interpret seam — no per-kind orchestrator fork |
| [`replay.py`](../../src/slm_training/harnesses/reasoning/revmath/replay.py) | Deterministic `ReplayBundleV1` composition with identity + capture digests |
| [`report.py`](../../src/slm_training/harnesses/reasoning/revmath/report.py) | `RevmathReportV1` aggregation + version stamp |
| [`scripts/run_revmath_task.py`](../../scripts/run_revmath_task.py) | Hermetic/Lean CLI |

Timeout, missing tool, incomplete check, unsupported kind, and malformed proof stay `unknown`/`invalid` — never witnessed/refuted. Runner does not mutate proposition or campaign state. Hermetic fixtures live under `resources/revmath/fixtures/`.


## HARN-04 assumption ablation (SLM-544)

| Module | Role |
| --- | --- |
| [`assumption_ablation.py`](../../src/slm_training/harnesses/reasoning/revmath/assumption_ablation.py) | Deterministic candidate generation + dependency audit + finite-lattice search report |
| [`AssumptionAblationPlugin`](../../src/slm_training/harnesses/reasoning/revmath/plugins.py) | Plan/interpret seam for `task_kind=assumption_ablation` (shared runner owns orchestration) |
| Hermetic fixtures | `resources/revmath/fixtures/ablation_{positive,redundant,necessary,timeout,hidden_import}.{task,meta}.json` + `hermetic_ablation_checker.py` |

A successful ablation proves the **same proposition** (`statement_sha256`) under a recorded weaker `allowed_assumption_ids` set. Hidden reintroduction is detected when proof dependencies intersect `import_edges` for removed assumptions (strength-bearing lemmas). **Minimality is never claimed beyond** `MINIMALITY_CLAIM_SCOPE=explored_finite_candidate_lattice`. Timeout stays `unknown` (KERN-01). Replay remains deterministic via the HARN-03 runner.


## HARN-06 constructivization + counterexample (SLM-546)

| Module | Role |
| --- | --- |
| [`constructivization.py`](../../src/slm_training/harnesses/reasoning/revmath/constructivization.py) | Modes + masquerade guard + constructivization report |
| [`counterexample.py`](../../src/slm_training/harnesses/reasoning/revmath/counterexample.py) | Exact weakened-theorem check + replayable model/trace |
| Plugins | `ConstructivizationPlugin`, `CounterexamplePlugin` in `plugins.py` |
| Hermetic fixtures | `constructivization_{bounded,witness,oracle,remainder,timeout}.*` + `counterexample_{checked,search_failed,no_counterexample,mismatch_prop}.*` |

Constructivized tasks bind `proposition` to the **constructivized** statement only. Counterexample refutation requires `independently_checked`; search failure stays `unknown`.


## HARN-11 frozen fixture corpus (SLM-554)

| Module / artifact | Role |
| --- | --- |
| [`corpus.py`](../../src/slm_training/harnesses/reasoning/revmath/corpus.py) | Assemble hermetic fixtures into a versioned content-addressed catalog; portable replay digests; train/eval root-family leakage detection; mutation-gate probes |
| [`hermetic_v1.manifest.json`](../../src/slm_training/resources/revmath/corpus/hermetic_v1.manifest.json) | Frozen manifest (`revmath_fixture_corpus/v1`) binding proposition/assumptions, toolchain, four-outcome expectations, labels, budgets, checker refs, replay digests, mutation expectations |
| Tests | `tests/test_harnesses/reasoning/test_revmath_corpus.py` — exact replay, distinct unknown/invalid vs refutation, declared mutation failures |

Coverage classes: ablation success/necessary; reversal equivalence/one-way; constructivization success/unknown; checked counterexample; quantitative-bound success/nonextractable; practical computability; proposition/toolchain/source/certificate mutations; timeout/missing-tool/unsupported/malformed/incomplete-domain. Does not invent task plugins — reuses HARN-03..09 hermetic checkers under `resources/revmath/fixtures/`.


## Honesty

This document freezes **ownership and extension seams** — not ship readiness, model quality, or empirical reverse-mathematics conclusions. Fixture and formal-preflight evidence remain wiring/diagnostic until promoted under normal campaign and `--ship-gates` policy.

## Commands

```bash
python -m scripts.run_revmath_task --task src/slm_training/resources/revmath/fixtures/hermetic_forward_theorem.task.json --hermetic
python -m scripts.verify_revmath_owners
python -m scripts.verify_revmath_harness_parity
python -m scripts.verify_agent_surfaces
python -m scripts.verify_ownership_map
python -m scripts.repo_policy
```
