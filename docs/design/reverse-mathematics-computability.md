# Reverse mathematics, computability, and canonical evidence ownership

**Status:** HARN-04 assumption ablation landed (SLM-544); EVID-03 four-axis ledger (SLM-519); HARN-03 runner/replay/report (SLM-536); HARN-02 schemas (SLM-527); profile + parity (HARN-01 / SLM-520); owner map (EVID-01 / SLM-515)  
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
| `revmath_replay` | replay | `harnesses/reasoning/revmath/replay.py` | `build_revmath_replay_bundle` | HARN-03 ReplayBundleV1 composition |
| `revmath_report` | report | `harnesses/reasoning/revmath/report.py` | `build_revmath_report` | HARN-03 typed reports |
| `continuous_formal_promote` | command | `scripts/run_autotrain_continuous.py` | `ensure_promote_formal_preflight` | promotion gate |

Design references: [formal-autoresearch.md](formal-autoresearch.md), [formal-objects-multi-prover.md](formal-objects-multi-prover.md), [experiment-campaign-governance.md](experiment-campaign-governance.md), [repository-ownership-map.md](repository-ownership-map.md).

Cross-links to the global ownership map (`ownership_map.json` subsystems): where a row names `ownership_map_subsystem`, downstream work must extend that subsystem rather than introduce a shadow module.

## Extension seams (downstream work)

| Issue | Extends | New owner? | Rule |
| --- | --- | --- | --- |
| EVID-03 / SLM-519 | `formal_preflight_schema` | no | **done** — `FormalPreflightV1.four_axis_ledger` + `FormalPreflightFourAxisLedgerV1` |
| EVID-04 / SLM-525 | `formal_preflight_schema` | yes (eval submodule) | safe symbolic bound AST consumed by ledger |
| EVID-06 / SLM-526 | `formal_object_schema` | no | v2 envelope adapts v1 objects |
| HARN-01 / SLM-520 | `reasoning_harness_parent` | no | register `reasoning/revmath` profile + parity matrix |

## Prohibited duplicate owners

Agents **must not** add modules that claim semantic authority for:

| Concern | Canonical owner | Blocked shadow patterns |
| --- | --- | --- |
| Campaign manifest / arms / gates | `experiment_campaign.py` | `revmath.*campaign`, `RevMathCampaign` |
| Evidence persistence | `decode_stats.py` (`DecodeStatsRecordV1`) | `revmath.*evidence`, `RevMathEvidenceStore` |
| Proof preflight execution | `autoresearch/formal.py` | `revmath.*proof`, `RevMathProofStack` |
| Orchestration / trainer | `harnesses/reasoning/` + campaign owners | `revmath_trainer`, `revmath_orchestrator` |

`verify_revmath_owners` scans `src/` and `scripts/` for these shadow patterns and fails closed.

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

Task schemas (HARN-02), deterministic runner/replay/report (HARN-03), and assumption-ablation validation (HARN-04) are present; other task-kind validators + repair controller (HARN-09) remain downstream. This document owns **authority boundaries and parity obligations**; missing parity rows are explicit blockers in the matrix, never silent fallbacks.

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

Declared in the parity resource (`self_healing`); implemented by HARN-09.

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
| `resource_bounds.bound_ast_id` | Stable id into EVID-04 safe bound AST (`bound.finite_search.prefix_tree.v1`, `bound.placeholder.pending_evid04.v1`, …); no raw `eval` |
| `empirical_remainder_claim_ids` | Claims that remain empirical (e.g. wall-clock, neural quality) |
| `theorem_claim_kinds` | Fail closed if `wall_clock_latency` or `neural_quality` are asserted as theorem consequences |

Migration: `migrate_formal_preflight_v1` / `formal_preflight_payload` / `formal_preflight_sha256` in `autoresearch/formal.py`. Digests omit an unset ledger so historical artifacts keep their hashes.

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
