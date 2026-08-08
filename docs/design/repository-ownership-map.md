# Repository ownership map

> **Goal law:** this document is bound by [decode-invariants.md](decode-invariants.md) — a rejected approach never closes a goal, and every agent surface carries the law (I7/I14/I15).

SGS-001 (SLM-435, milestone "Compatibility and ownership gate"). Commits a machine-readable owner/authority map for the live OpenUI synthesis stack (`SemanticPlanV1`, the G0-G12 verifier gates, semantic-failure taxonomy, repair, counterfactual replay, the experiment registry, telemetry, certified completion artifacts, DSL packs, search/lattice state, versioning, and generated docs) so the downstream SGS/VCE/PCT/SRP/SIE/RSP backlog extends an existing owner instead of duplicating one.

**Machine-readable source of truth:** [`src/slm_training/resources/ownership_map.json`](../../src/slm_training/resources/ownership_map.json) (`schema: ownership_map/v1`). **Verified by:** `python -m scripts.verify_ownership_map` (static AST/text check, no imports — every `owner_module` must exist and every `owner_symbols` entry must actually be defined there; every downstream row must cite a real extension point or justify a new owner; `OUTPUT_CONTRACT_VERSION` in code and every version mention in the output-contract doc must agree). Regression coverage: `tests/test_scripts/test_verify_ownership_map.py`.

*This page is mechanically generated from the JSON above — do not hand-edit the tables below; edit the JSON and regenerate.*

## Authority tiers

Every subsystem below is tagged with one of these tiers (acceptance criterion: "Authority table distinguishes compiler-hard, verifier-hard, advisory learned, oracle diagnostic, and evaluation-only evidence").

| Tier | Meaning |
| --- | --- |
| `compiler-hard` | Deterministic, grammar/contract-derived authority. Wrong output is a bug, never a soft preference; verified by static/AST checks, not sampling. |
| `verifier-hard` | Authority derived from running the G0-G12 verifier stack or an equivalent oracle against a candidate; pass/fail is binding, not advisory. |
| `advisory-learned` | Model- or heuristic-derived signal used to rank or prioritize among options that are already legal under a compiler-hard or verifier-hard gate. Never authorizes an illegal output and never substitutes for a missing verifier-hard check. |
| `oracle-diagnostic` | Ground-truth substitution or intervention used to diagnose or attribute causes (e.g. plan-oracle substitution, counterfactual probes). Used for measurement and repair-target construction, not served to users. |
| `evaluation-only` | Scoring, telemetry, or reporting surface. Observes and records; does not gate production decode and is not cited as decode-time authority. |

## Subsystem owners

| Subsystem | Owner module | Key symbols | Tier | Design doc |
| --- | --- | --- | --- | --- |
| `semantic_plan` — SemanticPlanV1 program-side plan contract | `src/slm_training/data/progspec/semantic_plan.py` | `SemanticPlanV1`, `PlanIdentity`, `PlanArchetype`, `RoleSlot`, `PlanTopology` | `compiler-hard` | [semantic-planning-valid-state.md](semantic-planning-valid-state.md) |
| `semantic_failure_taxonomy` — Semantic-failure taxonomy / evidence labeling | `src/slm_training/evals/semantic_failure.py` | `SemanticFailureFamily`, `SemanticFailureTaxonomyV1` | `evaluation-only` | [semantic-failure-census.md](semantic-failure-census.md) |
| `verifier_gate_stack` — G0-G12 deterministic verifier gate sequence | `src/slm_training/data/verify/stack.py` | `Gate`, `GATE_NAMES`, `GateResult`, `verify_record` | `verifier-hard` | [verifier-stack.md](verifier-stack.md) |
| `verifier_cascade_eval` — Eval-time verifier cascade wiring | `src/slm_training/evals/verifier_cascade.py` | `VerifierCascade`, `VerifierCascadeResult`, `VerifierStage`, `make_gate_stage`, `default_openui_cascade` | `evaluation-only` | [verifier-stack.md](verifier-stack.md) |
| `semantic_repair` — Verifier-guided minimal semantic repair targets | `src/slm_training/harnesses/distill/semantic_repair.py` | `SemanticRepairRecordV1`, `LegalEdit`, `RepairEvidence`, `RepairPolicyName`, `SemanticRepairScorer` | `advisory-learned` | [iter-spv2-05-semantic-repair-20260720.md](iter-spv2-05-semantic-repair-20260720.md) |
| `plan_oracle_substitutor` — Plan-level oracle substitution | `src/slm_training/data/semantic_plan/oracle.py` | `PlanOracleSubstitutor` | `oracle-diagnostic` | [semantic-planning-valid-state.md](semantic-planning-valid-state.md) |
| `counterfactual_probe` — Verifier-backed counterfactual action-value probe | `src/slm_training/harnesses/preference/counterfactual_probe.py` | `RolloutBackend` | `advisory-learned` | [ldi3-03-counterfactual-action-values-20260718.md](ldi3-03-counterfactual-action-values-20260718.md) |
| `counterfactual_replay` — Judge-based grammar-legal continuation replay | `src/slm_training/harnesses/preference/counterfactuals.py` | `SEMANTIC_VERIFIER_V1` | `verifier-hard` | [ldi3-04-remine-intervene-campaign-20260718.md](ldi3-04-remine-intervene-campaign-20260718.md) |
| `semantic_counterfactual_synthesis` — Training-pair counterfactual mutation generator | `src/slm_training/harnesses/train_data/semantic_counterfactuals.py` | `CounterfactualPairV1` | `evaluation-only` | [iter-slm366-counterfactuals-20260725.md](iter-slm366-counterfactuals-20260725.md) |
| `solver_replay` — Decode trace recording and replay validation | `src/slm_training/dsl/solver/replay.py` | `solver_replay_violations`, `solver_events_from_closure`, `solver_events_from_search`, `solver_trace_counters` | `verifier-hard` | [verified-scope-solver.md](verified-scope-solver.md) |
| `valid_state_search_generic` — Bounded proof-carrying search controller (generic) | `src/slm_training/dsl/solver/controller.py` | `SearchStatus`, `SearchDecision`, `SearchResult`, `search` | `compiler-hard` | [verified-scope-solver.md](verified-scope-solver.md) |
| `lattice_search_forest_adapter` — Compiler-forest-specific lattice search | `src/slm_training/dsl/grammar/fastpath/lattice_search.py` | `LatticeSearchState` | `compiler-hard` | [lattice-recursive-search.md](lattice-recursive-search.md) |
| `experiment_campaign` — Pre-registered experiment campaign contract | `src/slm_training/autoresearch/experiment_campaign.py` | `ExperimentCampaignV1`, `CampaignEndpointV1`, `CampaignArmV1`, `CampaignGateV1`, `campaign_manifest_sha256`, `validate_result_claim` | `compiler-hard` | [experiment-campaign-governance.md](experiment-campaign-governance.md) |
| `decode_telemetry` — Per-decode-call immutable telemetry | `src/slm_training/models/decode_stats.py` | `DecodeStats`, `collect_decode_stats`, `aggregate_stats` | `evaluation-only` | [decode-invariants.md](decode-invariants.md) |
| `runtime_telemetry` — Runtime-level telemetry sink | `src/slm_training/runtime/telemetry/__init__.py` | `SpanStats`, `CycleTelemetry`, `get_telemetry`, `timed` | `evaluation-only` | [decode-invariants.md](decode-invariants.md) |
| `certified_completion_artifact` — Certified static-LALR completion artifact | `src/slm_training/dsl/grammar/fastpath/completion_artifact.py` | `CompletionArtifact`, `StaticLalrAdapter`, `build_completion_artifact`, `load_checked_completion_artifact` | `compiler-hard` | [certified-completion-artifact-and-tps-target.md](certified-completion-artifact-and-tps-target.md) |
| `dsl_pack_registry_canonical` — DSL pack registry (canonical, singular) | `src/slm_training/dsl/pack.py` | `DslPack`, `register_pack`, `get_pack`, `list_packs` | `compiler-hard` | [dsl-pack-contract.md](dsl-pack-contract.md) |
| `dsl_pack_registry_shadow` — DSL pack registry (shadow, plural) -- DUPLICATE RISK | `src/slm_training/dsl/packs/types.py` | `DSLPack`, `PlaceholderPolicy` | `compiler-hard` | [dsl-pack-contract.md](dsl-pack-contract.md) |
| `output_contract` — Symbol-only output contract / OUTPUT_CONTRACT_VERSION | `src/slm_training/dsl/language_contract.py` | `OUTPUT_CONTRACT_VERSION` | `compiler-hard` | [symbol-only-output-contract.md](symbol-only-output-contract.md) |
| `completion_domain` — Grammar-capability-derived completion domain | `src/slm_training/dsl/grammar_capabilities.py` | `CompletionDomainV1`, `GrammarCapabilityAdapterV1`, `GrammarCapabilityAuthorityV1` | `compiler-hard` | [decode-invariants.md](decode-invariants.md) |
| `choice_codec` — Deterministic choice-sequence codec | `src/slm_training/models/choice_tokenizer.py` | `ChoiceTokenizer`, `ChoiceDecodeState` | `compiler-hard` | [decode-invariants.md](decode-invariants.md) |
| `production_codec` — Grammar production codec | `src/slm_training/dsl/production_codec.py` | `ProductionProgram`, `ProductionVocab`, `decode_productions` | `compiler-hard` | [decode-invariants.md](decode-invariants.md) |
| `version_registry` — Component-version registry and enforcement | `src/slm_training/resources/versions.json` | — | `compiler-hard` | [version-stamp-contract.md](version-stamp-contract.md) |
| `data_corpus_audit` — Cross-snapshot data-corpus leakage/dedup audit | `scripts/audit_data_corpora.py` | `main`, `_overlap_matrix`, `_near_dup_clusters` | `evaluation-only` | [data-corpus-audit.md](data-corpus-audit.md) |
| `openwiki_generation` — Generated agent-navigation documentation | `scripts/update_openwiki.py` | `run_openwiki`, `main` | `evaluation-only` | [quickstart.md](../openwiki/quickstart.md) |
| `agent_surface_parity` — Obligation x agent-harness-surface parity | `scripts/verify_agent_surfaces.py` | `AgentSurfaceError`, `Obligation`, `check`, `main` | `compiler-hard` | [agent-harness-parity-audit.md](agent-harness-parity-audit.md) |
| `repository_organization_policy` — Tracked-file placement policy | `scripts/repo_policy.py` | `ALLOWED_ROOTS`, `validate_skill_mirrors`, `main` | `compiler-hard` | [repository-organization.md](../repository-organization.md) |

Notes, one per subsystem:

- **`semantic_plan`**: Versioned, provenance-aware, pack-neutral plan IR. Single definition site; all consumers (models/semantic_plan_factors.py, data/semantic_contrast/transforms.py, data/semantic_plan/*) import from here.
- **`semantic_failure_taxonomy`**: Deliberately an adapter over the G0-G12 verifier stack; it versions/labels verifier evidence, it does not re-score it.
- **`verifier_gate_stack`**: Canonical G0(Lexical)..G12(HumanAudit) gate numbering and pass/fail. Any module that reimplements gate numbering instead of importing this Gate enum is a duplicate risk.
- **`verifier_cascade_eval`**: Re-imports Gate/GATE_NAMES from data/verify/stack.py directly; correct precedent for reuse, not a second gate authority.
- **`semantic_repair`**: Fixture-only interim baseline (SPV2-05). Its own docstring defers real verifier-backed counterfactual action values to SLM-131/VSS (counterfactual_probe). Not dead code; a documented pending supersession.
- **`plan_oracle_substitutor`**: Distinct from semantic_repair's corruption-repair targets: this substitutes ground-truth oracle values for factor-wise diagnosis, not corruption recovery.
- **`counterfactual_probe`**: LDI3-03/SLM-131. Named by semantic_repair.py as the eventual authority for verifier-backed repair action values -- currently a parallel, not yet merged, track.
- **`counterfactual_replay`**: Actively imported by decision_events_v2.py and local_decisions.py. Does not cross-reference counterfactual_probe.py; audit should confirm the split is intentional layering, not redundancy.
- **`semantic_counterfactual_synthesis`**: DSH2-06/SLM-366. Data-synthesis purpose (one-fact training-pair mutation), distinct from decode-time replay owners above.
- **`solver_replay`**: VSS1-04/SLM-64. Validates the event stream produced by DecodeTraceRecorder (owned separately by src/slm_training/harnesses/distill/trace_store.py) against the solver search trace; does not own DecodeTraceRecorder itself.
- **`valid_state_search_generic`**: VSS1-02. Docstring states explicitly: this is the new generic controller; the compiler-forest LatticeSearchState is retained unchanged as the forest-specific adapter. This split is a documented, intentional reconciliation, not an accidental duplicate.
- **`lattice_search_forest_adapter`**: Retained forest adapter beneath valid_state_search_generic; not superseded, not dead code.
- **`experiment_campaign`**: AP-007+ runners and every promotion candidate use this contract. No competing implementation found.
- **`decode_telemetry`**: Timing/step-count/fastpath hit-miss telemetry collected during inference/eval.
- **`runtime_telemetry`**: Lower-level runtime namespace used by runtime/decode_schedule.py. Consumes/produces DecodeStats; not a re-implementation of it.
- **`certified_completion_artifact`**: Checkpoint-bound, safetensors-encoded control-table artifact. Generated via scripts/build_completion_artifact.py; certified by dsl/grammar/fastpath/static_control_domain.py.
- **`dsl_pack_registry_canonical`**: F1/SLM-34. 114 importers across dsl/operators/*, dsl/harness_dsl.py, dsl/grammar/fastpath/*, and web/routes.py (dashboard pack API). This is the live, wired-in pack registry.
- **`dsl_pack_registry_shadow`**: Also cites F1/SLM-34 but is a second, independently-registered pack system (note capitalization DSLPack vs DslPack). Only 7 importers, all under dsl/packs/*, harnesses/reasoning/bench.py, and their tests. See duplicate_subsystem_risks below; flagged for consolidation, not rewritten by this audit (non-goal: no new search/semantic behavior). Registry functions (register_pack/get_pack/list_packs) live in dsl/packs/__init__.py; the DSLPack dataclass itself is defined in dsl/packs/types.py.
- **`output_contract`**: OUTPUT_CONTRACT_VERSION=2 (checkpoint-incompatible with v1). See known_drift for the doc mismatch this audit reconciles.
- **`completion_domain`**: Single definition site; used by the fastpath engine to compute legal-symbol candidates.
- **`choice_codec`**: Fastpath alternative to open-vocab tokenization. Coexists deliberately with the DSL-native tokenizer (is_choice_tokenizer discriminator); not a duplicate.
- **`production_codec`**: Related production-level codec consumed alongside choice_codec.
- **`version_registry`**: Enforced by scripts/verify_version_stamps.py in CI, pre-commit, and agent hooks. Governs every other component id in this map.
- **`data_corpus_audit`**: AGENTS.md data-quality law: cross-snapshot overlap/leakage/near-dup auditing owner. New corpora (e.g. SRP-008 symbolic-regression corpus, VCE-008 split audits) run through this existing tool rather than a new leakage checker.
- **`openwiki_generation`**: Regenerates docs/openwiki/*; do not hand-edit generated pages.
- **`agent_surface_parity`**: Certifies every repository law appears on every configured coding-harness instruction surface (AGENTS.md, CLAUDE.md, GEMINI.md, .github/copilot-instructions.md, .cursor/rules/*.mdc, .codex/, .grok/).
- **`repository_organization_policy`**: Enforces ALLOWED_ROOTS and validate_skill_mirrors; executable counterpart to docs/repository-organization.md and .agents/skills/organize-repository/SKILL.md.

## Duplicate-subsystem risks

Concerns with two owners registered against the same concept. Acceptance criterion: "No new registry/search/verifier/telemetry/pack/version owner duplicates an existing canonical owner" — these are the two the audit found on the *existing* stack; new work must not add a third.

### DSL pack registry

- **Canonical owner:** `src/slm_training/dsl/pack.py`
- **Shadow owner:** `src/slm_training/dsl/packs/types.py`
- **Status:** `unresolved-duplicate`
- **Evidence:** Both cite ticket F1/SLM-34 as origin. Canonical has 114 importers incl. the dashboard pack API (web/routes.py); shadow has 7 importers, all confined to dsl/packs/* and harnesses/reasoning/bench.py.
- **Recommendation:** New DSL packs (including SRP-001's symbolic-regression pack) register with the canonical dsl.pack.DslPack registry. Consolidating or retiring dsl/packs/ is out of scope for this audit (non-goal: no new semantic/search behavior) and should be filed as its own follow-up.

### Valid-state / lattice search

- **Canonical owner:** `src/slm_training/dsl/solver/controller.py`
- **Shadow owner:** `src/slm_training/dsl/grammar/fastpath/lattice_search.py`
- **Status:** `reconciled-by-docstring`
- **Evidence:** controller.py (generic search controller, VSS1-02) docstring states the forest-specific LatticeSearchState in lattice_search.py is retained unchanged as an adapter beneath it.
- **Recommendation:** Not a defect; record as an intentional layered design so future audits do not re-flag it.

## Known drift (reconciled)

### `output_contract_version_doc_mismatch`

docs/design/symbol-only-output-contract.md stated 'The canonical contract is OUTPUT_CONTRACT_VERSION = 4' while src/slm_training/dsl/language_contract.py defines OUTPUT_CONTRACT_VERSION = 2. AGENTS.md, .cursor/rules/decode-invariants.mdc, docs/design/decode-invariants.md, and the current test suite (tests/test_dsl/test_language_contract.py) all agree with the code at 2. A prior investigation (docs/design/autotrain-cycle-a6r56e-c2-allowed-id-set-fingerprint-fix.md) found and fixed the same drift in the test suite but never fixed the design doc.

**Status:** `fixed-in-SGS-001`. **Fix:** docs/design/symbol-only-output-contract.md corrected to cite OUTPUT_CONTRACT_VERSION = 2, matching src/slm_training/dsl/language_contract.py. scripts/verify_ownership_map.py now checks this pair on every run so the drift cannot silently return.

## Related overlaps

Existing Linear issues/docs that downstream SGS-001 work must coordinate with rather than re-derive (per SGS-001's scope: "Record overlaps with VSS, LDI, Semantic Planning, SLM-106, SLM-131, SLM-159, and SLM-160").

| Issue | Title | Doc | Relation |
| --- | --- | --- | --- |
| SLM-106 | EFS0-04: Audit judge independence with cross-family scoring and a blinded human pair study | [judge-independence-audit.md](judge-independence-audit.md) | VCE-010 (evaluator calibration/blinded adjudication) and SIE-005/SIE-006 must coordinate with this existing judge-independence track rather than re-run it. |
| SLM-131 | LDI3-03: Generate verifier-backed counterfactual action values for delayed failures | [ldi3-03-counterfactual-action-values-20260718.md](ldi3-03-counterfactual-action-values-20260718.md) | Owner of counterfactual_probe; semantic_repair.py names it as the eventual authority for verifier-backed repair action values. |
| SLM-159 | SPV4-01: Replicate the winning semantic-plan and valid-state stack on GraphQL plus a second DSL pack | [iter-slm159-cross-dsl-replication-20260720.md](iter-slm159-cross-dsl-replication-20260720.md) | Precedent for SRP-001/SRP-010 second-pack portability work; do not re-derive the replication protocol. |
| SLM-160 | SPV4-02: Publish the causal architecture disposition and integrate only qualified winners into model lineage | [semantic-planning-valid-state-disposition.md](semantic-planning-valid-state-disposition.md) | Canonical disposition-report pattern that SGS-009 (mechanism disposition reports) and RSP-009 (cross-experiment disposition) should follow, not reinvent. |

## Downstream extension map

Every SGS/VCE/PCT/SRP/SIE/RSP backlog item mapped to the existing subsystem(s) it extends, or an explicit justification for a new owner (acceptance criterion: "Map every downstream SGS/VCE/PCT/SRP/SIE/RSP item to an existing extension point or explicitly justify a new owner").

| Issue | Linear | Extension point(s) | New owner? | Why |
| --- | --- | --- | --- | --- |
| **SGS-002** — Enforce machine-checkable contract/documentation version consistency | SLM-436 | `output_contract`, `version_registry` | no | Extends the code/docs consistency check this audit establishes (see known_drift) into a standing CI guard. |
| **SGS-003** — Add prompt-side partial semantic requirements without duplicating SemanticPlanV1 | SLM-437 | `semantic_plan` | yes | SemanticPlanV1 is the program-side output contract, not a prompt-side input type; a new adjacent 'prompt_requirements' owner is justified because merging into SemanticPlanV1 would violate its single-schema contract. |
| **SGS-004** — Implement conservative deterministic request-to-requirement extraction | SLM-442 | `semantic_plan` | no | Extends the SGS-003 prompt_requirements owner; no new owner beyond it. |
| **SGS-005** — Add production/oracle/hard-authority projections with invariance tests | SLM-443 | `semantic_plan` | no | Extends SGS-003/004's prompt_requirements owner using the authority_tiers vocabulary this map publishes. |
| **SGS-006** — Integrate prompt requirements with existing semantic-plan/evaluation/decode owners | SLM-454 | `semantic_plan`, `verifier_gate_stack`, `decode_telemetry` | no | Pure integration issue across three existing owners; no new owner. |
| **SGS-007** — Implement VerifiedSynthesisProblemV1 and generated JSON Schema | SLM-444 | `semantic_plan`, `verifier_gate_stack`, `dsl_pack_registry_canonical` | yes | No existing contract shapes a pack-supplied, verifier-checked synthesis problem; a new VerifiedSynthesisProblemV1 owner is justified, layered on the three existing owners it must not duplicate. |
| **SGS-008** — Add fail-closed SyGuS/SemGuS capability reports and conformance fixtures | SLM-455 | `dsl_pack_registry_canonical`, `completion_domain` | no | Extends the canonical pack registry's capability surface; no new owner. |
| **SGS-009** — Generate mechanism disposition and stale-evidence supersession reports | SLM-456 | — | yes | No existing subsystem owns disposition/supersession reporting as reusable code; it follows the documented narrative precedent in docs/design/semantic-planning-valid-state-disposition.md (SLM-160) rather than a registered code owner, and its output feeds version_registry once mechanisms are actually retired/frozen/promoted. |
| **SGS-010** — Close schema versioning, migrations, and compatibility tests for synthesis contracts | SLM-445 | `version_registry` | no | Extends version_registry to cover SGS-007's VerifiedSynthesisProblemV1 once it exists. |
| **VCE-001** — Extend semantic-failure traces into lossless typed verifier witnesses | SLM-438 | `semantic_failure_taxonomy`, `verifier_gate_stack` | no | Extends the existing taxonomy adapter and gate stack; witnesses are a lossless superset of current traces. |
| **VCE-002** — Unify repair residuals with existing SemanticRepairRecordV1 and conflict slices | SLM-446 | `semantic_repair` | no | Title names the existing owner directly. |
| **VCE-003** — Add verifier-witness determinism, replay, tamper, and redaction security tests | SLM-457 | `solver_replay`, `verifier_gate_stack` | no | Tests the VCE-001 witness type through the existing replay owner. |
| **VCE-004** — Promote existing PlanOracleSubstitutor into a full factor-wise intervention harness | SLM-447 | `plan_oracle_substitutor` | no | Title names the existing owner directly. |
| **VCE-005** — Add no-op, destructive, shuffled, one-factor, and all-oracle intervention controls | SLM-458 | `plan_oracle_substitutor` | no | Extends the VCE-004 intervention harness with control arms. |
| **VCE-006** — Extend the existing hard-valid semantic contrast corpus for missing factor families | SLM-448 | `semantic_plan` | no | Extends data/semantic_contrast/transforms.py, an existing SemanticPlanV1 consumer. |
| **VCE-007** — Add metamorphic prompt/program, alpha-renaming, and positive-equivalence generators | SLM-459 | `semantic_plan`, `counterfactual_replay` | no | Extends the VCE-006 contrast corpus and the existing judge-based replay verifier. |
| **VCE-008** — Add leakage, deduplication, topology, and OOD split audits | SLM-463 | `data_corpus_audit` | no | Extends the existing data_corpus_audit tool (scripts/audit_data_corpora.py) named in AGENTS.md data-quality law, not a new owner. |
| **VCE-009** — Record oracle/contrast fixture campaigns in governed evidence envelopes | SLM-468 | `experiment_campaign` | no | ExperimentCampaignV1 is the existing governed-evidence-envelope owner. |
| **VCE-010** — Define evaluator calibration, blinded adjudication, and risk-coverage protocol | SLM-469 | `verifier_gate_stack` | no | Must coordinate with the existing judge-independence audit (SLM-106, see related_overlaps) rather than duplicate it. |
| **PCT-001** — Extend immutable telemetry with authority, exact-work, witness, and cold/warm fields | SLM-439 | `decode_telemetry` | no | Title names the existing owner directly. |
| **PCT-002** — Add hermetic replay traces and an optional observability sink | SLM-449 | `runtime_telemetry`, `solver_replay` | no | Extends the existing runtime telemetry sink and solver replay recorder. |
| **PCT-003** — Implement an honest end-to-end cold and warm benchmark harness | SLM-450 | `decode_telemetry` | no | Extends the existing bench_* harness family named in AGENTS.md's experiment triggers list; no new owner. |
| **PCT-004** — Generalize the certified OpenUI completion artifact behind a pack-owned provider seam | SLM-440 | `certified_completion_artifact`, `dsl_pack_registry_canonical` | no | Title names both existing owners directly. |
| **PCT-005** — Prove OpenUI artifact byte, digest, decision, and tamper parity through the provider seam | SLM-451 | `certified_completion_artifact` | no | Verifies the PCT-004 provider seam. |
| **PCT-006** — Add static/request-boundary differential tests and singleton zero-work guards | SLM-460 | `certified_completion_artifact`, `completion_domain` | no | Extends the existing artifact and completion-domain owners; guards goal-invariant I2 (forced bypass on singletons). |
| **PCT-007** — Add mechanism activation, choice-change, and disposition telemetry | SLM-452 | `decode_telemetry` | no | Extends decode_telemetry with the disposition concept SGS-009 formalizes. |
| **PCT-008** — Run fixture cold/warm and artifact-parity evidence campaign | SLM-464 | `experiment_campaign`, `certified_completion_artifact` | no | Runs PCT-003/004/005/006 work through the existing campaign contract. |
| **SRP-001** — Register an isolated symbolic-regression DslPack and problem contract | SLM-441 | `dsl_pack_registry_canonical` | yes | New pack instance is expected and correct; MUST register with the canonical dsl.pack.DslPack registry, not the shadow dsl/packs/ registry (see duplicate_subsystem_risks). |
| **SRP-002** — Implement typed symbolic-expression IR, grammar, codec, and symbol-only contract | SLM-453 | `output_contract`, `production_codec` | yes | New pack-local IR is justified (no existing symbolic-expression grammar) but must layer under the shared output_contract, not fork it. |
| **SRP-003** — Implement a safe vectorized NumPy evaluator with explicit numerical-domain policies | SLM-461 | `dsl_pack_registry_canonical` | yes | No existing numeric evaluator for this pack; new owner justified, scoped inside SRP-001's pack. |
| **SRP-004** — Implement affine coefficient analysis and deterministic least-squares fitting | SLM-465 | `dsl_pack_registry_canonical` | no | Extends the SRP-003 evaluator. |
| **SRP-005** — Implement canonicalization and a proof-scoped rewrite-certificate schema | SLM-462 | `verifier_gate_stack` | yes | New pack-local canonicalizer justified; its rewrite-certificate must follow the VCE-001 witness pattern rather than invent a parallel proof format. |
| **SRP-006** — Add evaluator, fitting, canonicalization, and numerical differential tests | SLM-470 | `dsl_pack_registry_canonical` | no | Tests SRP-003/004/005; no new owner. |
| **SRP-007** — Implement validity, fit, extrapolation, complexity, and Pareto reporting | SLM-466 | `dsl_pack_registry_canonical` | no | Reporting layer over SRP-003/004/005. |
| **SRP-008** — Implement deterministic symbolic-regression corpus generation with OOD and leakage controls | SLM-471 | `data_corpus_audit` | no | Reuses data_corpus_audit (scripts/audit_data_corpora.py) leakage-control machinery for a new corpus, not a new owner. |
| **SRP-009** — Implement a bounded canonical enumerative symbolic-regression baseline | SLM-472 | `valid_state_search_generic` | no | Reuses the generic bounded search controller as the search backbone rather than inventing a new search owner. |
| **SRP-010** — Add end-to-end symbolic-pack portability, default-isolation, and standards fixtures | SLM-474 | `dsl_pack_registry_canonical` | no | Extends the canonical pack registry's portability guarantees to the SRP-001 pack. |
| **SRP-011** — Define an optional PySR/SRBench adapter with isolated environment manifest | SLM-475 | — | yes | External, optional dependency by design; isolated deliberately and does not extend any internal owner. |
| **SIE-001** — Register EXP-SR campaign catalogue and promotion guardrails | SLM-467 | `experiment_campaign` | no | Registers the EXP-SR catalogue in the existing ExperimentCampaignV1 contract. |
| **SIE-002** — Execute held-out factor-wise semantic oracle localization (EXP-SR-1) | SLM-476 | `plan_oracle_substitutor`, `experiment_campaign` | no | Runs the VCE-004 intervention harness through the campaign contract. |
| **SIE-003** — Implement prompt-to-partial-requirements prediction behind the advisory-only seam (EXP-SR-2) | SLM-477 | `semantic_plan` | no | Extends SGS-003/004's prompt_requirements owner; must stay advisory-learned, never promoted to compiler-hard. |
| **SIE-004** — Run matched prompt-requirement predictor campaign (EXP-SR-2) | SLM-482 | `experiment_campaign` | no | Runs SIE-003 through the campaign contract. |
| **SIE-005** — Prepare blinded human/independent-judge data, adjudication packet, and credential runbook | SLM-478 | `verifier_gate_stack` | no | Coordinates with the existing SLM-106 judge-independence track (see related_overlaps) and VCE-010's protocol. |
| **SIE-006** — Execute evaluator construct-validity and human calibration campaign (EXP-SR-3) | SLM-483 | `experiment_campaign`, `verifier_gate_stack` | no | Runs VCE-010 + SIE-005 through the campaign contract. |
| **SIE-007** — Implement ComputeStateFeaturesV1 and a safe symbolic-controller replay substrate | SLM-473 | `decode_telemetry`, `solver_replay` | yes | New typed feature record justified (no existing compute-state feature schema), layered on the existing telemetry and replay owners it must not duplicate. |
| **SIE-008** — Execute symbolic value-of-compute controller campaign (EXP-SR-5) | SLM-484 | `experiment_campaign` | no | Runs SIE-007 through the campaign contract; capacity/compute claims must stay advisory-learned and size-matched per goal-invariant VI. |
| **RSP-001** — Execute witness-guided CEGIS repair experiment (EXP-SR-4) | SLM-488 | `semantic_repair`, `verifier_gate_stack`, `experiment_campaign` | no | Runs semantic_repair + VCE-001 witnesses through the campaign contract. |
| **RSP-002** — Execute neural-guided bounded search and infilling-order experiment (EXP-SR-6) | SLM-489 | `valid_state_search_generic`, `experiment_campaign` | no | Runs the generic search controller through the campaign contract; arms must be size-matched per goal-invariant VI. |
| **RSP-003** — Execute extended static-artifact and packed semantic-summary experiment (EXP-SR-7) | SLM-485 | `certified_completion_artifact`, `experiment_campaign` | no | Runs the PCT-004/005 provider seam through the campaign contract. |
| **RSP-004** — Execute a proof-scoped e-graph/equivalence-region experiment in the symbolic pack (EXP-SR-8) | SLM-479 | `dsl_pack_registry_canonical`, `experiment_campaign` | no | Runs SRP-005's canonicalization/rewrite-certificate work through the campaign contract. |
| **RSP-005** — Execute prospective certified macro/library-learning experiment (EXP-SR-9) | SLM-480 | `dsl_pack_registry_canonical`, `certified_completion_artifact`, `experiment_campaign` | no | Runs pack + artifact owners through the campaign contract. |
| **RSP-006** — Execute quality-diversity corpus generation experiment (EXP-SR-10) | SLM-481 | `experiment_campaign` | no | Runs SRP-008 corpus generation through the campaign contract. |
| **RSP-007** — Execute matched PySR and SRBench-compatible symbolic benchmark (EXP-SR-11) | SLM-486 | `experiment_campaign` | no | Runs the SRP-011 optional external adapter through the campaign contract. |
| **RSP-008** — Run real OpenUI + symbolic-regression second-pack portability certification (EXP-SR-12) | SLM-487 | `dsl_pack_registry_canonical`, `experiment_campaign` | no | Runs SRP-010 portability work through the campaign contract, following the SLM-159 replication precedent. |
| **RSP-009** — Publish cross-experiment disposition and retire, freeze, or promote mechanisms honestly | SLM-490 | `version_registry`, `experiment_campaign` | no | Follows the SLM-160 disposition-report precedent and this map's known_drift/duplicate_subsystem_risks pattern; no new owner. |

## Validation

```bash
python -m scripts.verify_ownership_map
python -m pytest tests/test_scripts/test_verify_ownership_map.py -q
python -m scripts.verify_agent_surfaces
python -m scripts.verify_version_stamps --check
PYTHONPATH=src python -m scripts.verify_decode_invariants
```

`verify_ownership_map` checks every owner claim by parsing the owner module's AST and confirming each claimed symbol is defined there as a class, function, or assignment target. It reads only the JSON, never the prose in this file — importer counts in the notes above come from a manual audit and are not themselves certified.

## Non-goals

No new semantic/search behavior and no historical-result rewriting. The `dsl/packs/` shadow pack registry and the `semantic_repair.py` → `counterfactual_probe.py` supersession are recorded here as known state, not resolved by this audit — resolving them is downstream work (see the DSL pack registry risk above and VCE-002/VCE-004).

