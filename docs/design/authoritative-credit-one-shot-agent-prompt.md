# One-shot coding-agent instruction: authoritative credit + sealed evaluation for slm-training

**Audience:** A parent LLM coding agent that may spawn specialized subagents.  
**Repository:** [Tyler-R-Kendrick/slm-training](https://github.com/Tyler-R-Kendrick/slm-training)  
**Base for this prompt:** verify `git rev-parse HEAD` at start; work from current `main` tip (audit snapshot included `3345ff71` / PR #1233 hill-climb governance; later main may include climb-policy and Phase A quality fixes—rebase, do not assume stale paths).  
**Delivery:** Full repo SDLC (implement → tests → docs → version stamps → PR → adversarial closeout → squash-merge) unless an explicit external block is documented. Do **not** end with “ship later.”

## Shipped TCB status (implementation land)

Authoritative credit TCB (criteria from the session goal plan) is implemented in-tree:

| Piece | Location |
| --- | --- |
| Defaults (volatile thresholds) | `resources/experiments/authoritative_credit/defaults.v1.json` |
| Pure recompute | `src/slm_training/autoresearch/credit_engine.py` |
| Promotion/ship validation | `validate_result_claim` requires `observation_table` + `analysis_plan` + `credit_report`; recomputes and rejects summary-only / mismatched caller Holm |
| Governance cannot clear evidence | `promotion.py` + `web/routes.py` keep `sufficient_evidence`; no promotable=True from governance alone |
| Sealed score-once | `SealedScoreLedger` + `assert_score_once` |
| Tests | `tests/test_autoresearch/test_credit_engine.py` |

Remaining §17 experiment families (Monarch, BOHB, pathology archive production, muP, …) stay **non-goals** unless filed separately.

---

## 0. Verified problem statement (do not re-litigate)

### 0.1 What is already true in this repo

Non-negotiable laws live in `AGENTS.md` and `docs/design/decode-invariants.md` (I1/I2/I6 constrained decode, singleton zero-forward, fail-closed grammar, EG_params / size-match, ship vs fixture honesty, ExperimentCampaignV1 preregistration, version stamps, docs-after-runs).

Campaign governance (`src/slm_training/autoresearch/experiment_campaign.py`, `hillclimb.py`, climb policy resources when present) already enforces:

- claim-class climb vs fixture separation;
- locked eval digests (and optional on-disk digest verification);
- multi-seed climb fields and causal campaign shape;
- artifact kind counts + content hashes of declared files;
- EG_params fail-closed on capacity growth;
- exhausted-knob / synthesis-feedback gates on some train/autoresearch paths.

**Proven (code-level):** `CampaignResultV1` carries caller-supplied `endpoint_values` and `holm_results`. `validate_result_claim` checks presence, finiteness, gate thresholds against those values, and artifact file digests—it does **not** recompute endpoints, paired effects, CIs, p-values, or Holm decisions from a task-level observation table.

**Proven (tests):** `tests/test_autoresearch/test_experiment_campaign.py` `_complete_result` invents `endpoint_values` / Holm rows and writes placeholder JSON `{"kind": ...}` whose hash matches the result; `test_complete_promotion_candidate_passes` expects success. That proves **structural lineage**, not empirical truth.

**Proven (promotion wrapper):** `src/slm_training/harnesses/experiments/promotion.py` (and `web/routes.py`) replace a sole failure of `sufficient_evidence` with `governed_campaign_evidence` and set `promotable=True` when campaign governance passes. Checkpoint registration adds more gates, but **governance of self-reported numbers must not substitute for evidence of an effect**.

### 0.2 What the conversation got right vs wrong

| Source | Keep | Reject / demote |
| --- | --- | --- |
| Kimi harness SOTA review | Proposal≠credit; sealed eval; pathology archives; budget-matched baselines; Monarch≠grammar projection; stronger evolver confounds P3 | Treat any of this as already fully implemented in this repo |
| Grok follow-ups | Structural separation of artifacts exists; efficiency claims need local wall-clock | Labeling “proven” without full ledger audit; Monarch projection complexity as \(O(n^2)\) (paper: analytical projection is \(O(n^{5/2})\)); assuming 2022 V100 speedups transfer to this grammar/orchestration-heavy stack |
| ChatGPT audit | **Authoritative credit gap** is the top trust hole; fourteen experiment families; default-off structured matrices; grammar-alignment only on ambiguous legal decisions | Requiring live Terminal-Bench agent-harness runs as the primary deliverable (wrong product domain) |

### 0.3 Highest-priority product change (this prompt’s north star)

```text
Immutable raw paired observations (bytes) + locked analysis plan
        → deterministic credit engine
        → every decision-bearing statistic + promotion decision
```

Caller-supplied endpoint/Holm/gate summaries may be **echoed for display only** after recomputation, or **rejected** if they disagree. They must never be the sole authority.

---

## 1. Goal for the coding agent

Implement **authoritative statistical credit** and the supporting experiment harnesses so that:

1. A promotion / ship result **cannot** be created from summaries alone.
2. Every decision-bearing statistic is **deterministically reproducible** from immutable raw observations and a locked analysis plan.
3. Adversarial tests prove caller-supplied stats are ignored or rejected when they disagree with recomputation.
4. Fixture/smoke paths remain honest (claim class + locked eval + climb policy) and never mint capability claims.
5. Decode invariants (I1/I2/I6) are untouched: no unconstrained production decode; singleton bypass remains zero-forward; no legality weakening for “alignment” experiments.

**Non-goals:** paid GPU trains; unlocking RL readiness; weakening ship gates; inventing a parallel autotrain path; claiming Monarch wall-clock wins without local benchmarks; full multi-rung NL ladder automation.

---

## 2. Parent agent role and subagent roster

The **parent** plans layers, owns the trusted computing base (TCB), integrates PRs, runs adversarial closeout, and enforces repo laws. Parent must **not** implement every layer alone when subagents can.

Spawn these subagents (names are roles; use `general-purpose` / `plan` / `explore` as available). Each receives: this document’s relevant sections, allowed paths, forbidden paths, local test commands, and “do not touch lower-layer ownership without parent.”

| ID | Subagent | Owns | Must not |
| --- | --- | --- | --- |
| S1 | **Schema architect** | New schemas under `src/slm_training/resources/` + Pydantic models; version stamps | Rewrite decode / grammar cores |
| S2 | **Credit engine** | Deterministic reduction: observations → endpoints, paired Δ, CIs, p-values, Holm, gate outcomes | Call LLMs; trust caller stats |
| S3 | **Campaign integration** | Wire `ExperimentCampaignV1` / `validate_result_claim` / `CampaignResultV1` to credit engine; reject V1 summary-only promotion | Leave dual authority paths |
| S4 | **Promotion / registration** | Fix `promotion.py` + `routes.py` so governance cannot replace missing comparative evidence | Ship checkpoints without certificates |
| S5 | **Sealed evaluation** | Score-once final eval, sealed suite identity, no proposal-time score reuse | Leak sealed items into agent-visible logs |
| S6 | **Budget & null arms** | Matched-budget control requirements; standing null/randomization/neural-necessity fixtures | Default-on capacity growth |
| S7 | **Pathology archive** | WHERE×WHY credited archive + admission gates (validity, activation, authoritative credit) | Task-id-only archive keys as sole index |
| S8 | **Metric / OOD / construct** | Semantic-metric construct validity, representation ceiling, cluster-aware OOD harnesses (default-off) | Change ship thresholds without meta-campaign |
| S9 | **Decode / cache / matrix probes** | Grammar-alignment on ambiguous legal ranks only; structured-matrix property + wall-clock probes default-off | Touch singleton bypass or force-emit authority |
| S10 | **Formal / adversarial tests / SDLC** | Lean digest binding non-vacuity; adversarial tests for credit; docs; `versions.json`; PR stack closeout | Rewrite goldens without explanation |

Parent integrates S1→S2→S3→S4 first (hard dependency). S5–S9 may parallelize after S1 schemas exist. S10 runs continuously (tests/docs) and owns merge.

---

## 3. Trusted computing base (TCB)

Only these may **mint** decision-bearing numbers:

1. Deterministic eval runners writing **observation records** (JSONL/JSON with schema + content hash).
2. The **credit engine** (pure functions + locked analysis plan digest).
3. Optional Lean certificates for metric bands (already exist)—they certify ranges, **not** empirical primary endpoints unless bound to observation digests.

Explicitly **out of TCB:** LLM researcher memos, agent chat, markdown narrative, caller-filled `endpoint_values` / `holm_results` without matching recomputation, dashboard edits, unsealed “scratch” scores.

---

## 4. Schema and artifact design (S1)

Add versioned resources (same spirit as `resources/experiments/semantic_factor_frontier/*.v1.json` and `resources/experiments/autotrain_climb/policy.v1.json`):

### 4.1 `observation_table/v1`

Immutable task-level rows. Minimum fields (extend carefully, forbid free-form authority):

- `schema`, `version`
- `observation_id` (content-addressed)
- `campaign_id`, `experiment_id`, `manifest_sha256`, `locked_eval_manifest_sha256`
- `arm_id`, `seed`, `example_id` (stable pairing key)
- `metric_id`, `value` (finite float), `higher_is_better` or direction enum
- `claim_class`, `suite_id`, `split` (`search` | `seal` | `ship`)
- `code_sha` / `version_stamp` reference
- optional `wall_ms`, `trainable_params`, `budget_tokens`

Content-address the full table; store under campaign artifacts with kind `observation_table`.

### 4.2 `analysis_plan/v1`

Locked before outcomes (preregistered with campaign or bound in lock):

- primary metric + direction + minimum effect
- paired vs unpaired
- seed list
- multiplicity family + Holm alpha
- CI method (e.g. stratified bootstrap / normal LCB) + parameters
- power inputs (prospective only)
- seal suite id + score-once policy
- budget match requirements
- standing null arm ids

Digest must be in `CampaignLockV1` or a sibling lock artifact.

### 4.3 `credit_report/v1`

Output of the credit engine only:

- digests of observation table + analysis plan
- per-endpoint recomputed values
- paired effect, CI, p-value, Holm rows (recomputed)
- gate outcomes
- `underpowered: true` → disposition `inconclusive` (not credited negative)
- `promotable_empirical: bool` (separate from structural governance)

### 4.4 Migration

- Keep reading historical `CampaignResultV1` summaries.
- New promotion_candidate / ship_gate **require** observation_table + credit_report digests.
- Mark summary-only historical results `claim_class` / honesty label as **non-authoritative** in docs/MODEL_CARD language when re-surfaced.

Bump `src/slm_training/resources/versions.json` for every watched path (no dual ownership).

---

## 5. Credit engine (S2)

Implement pure module e.g. `src/slm_training/autoresearch/credit_engine.py` (or under `harness_core/` if truly DSL-agnostic):

```text
credit_report = credit_from_observations(observation_table, analysis_plan) → CreditReportV1
```

Requirements:

- Deterministic given bytes (same inputs → same report digest).
- Recompute: endpoint aggregates, paired Δ (candidate−control per example/seed), direction-signed improvement (reuse `hillclimb.improvement_signed` / climb policy direction).
- Holm over preregistered family only; ranks/thresholds recomputed; **reject** caller Holm if mismatch.
- If power inputs show underpower and non-significant: `inconclusive`, never “negative proven.”
- No network, no LLM, no reading agent markdown.

Unit tests: golden vectors with hand-computed small tables; adversarial permutation of input rows must not change report if sort-normalized; tampering one observation must change digest and fail match.

---

## 6. Campaign / promotion integration (S3–S4)

### 6.1 `validate_result_claim` changes

For `claim_class in {promotion_candidate, ship_gate}`:

1. Require artifact kinds: `observation_table`, `credit_report` (and existing kinds).
2. Load table + plan + report from `artifact_root`; verify sha256.
3. Re-run credit engine; compare to report (fail closed on mismatch).
4. Use **recomputed** endpoints for gate matching—not `result.endpoint_values` unless equal to recomputed.
5. Optionally still require `endpoint_values` to **match** recomputation (dual-write for UI) or deprecate them.

### 6.2 Adversarial tests (mandatory)

- Invented Holm / endpoint values with empty or inconsistent observations → **fail**.
- Placeholder `{"kind":...}` observation_table → **fail** semantic validation.
- Governance-only path: campaign structure green, no comparative observations → **not** promotable; `governed_campaign_evidence` must **not** clear `sufficient_evidence` without credit_report pass.
- Fixture claim_class still cannot label climb.

### 6.3 Fix promotion wrapper

In `promotion.py` and `routes.py`:

- Remove or narrow the branch that sets `promotable=True` solely from governance when engine failed `sufficient_evidence`.
- Require `credit_report.promotable_empirical` (or equivalent check) for promotion_candidate/ship_gate.
- Keep checkpoint registration certificates; do not weaken them.

---

## 7. Sealed evaluation (S5)

- Preregister `seal` split identity (manifest sha) distinct from search/screening.
- Score-once: credit engine or eval runner records `scores_emitted=true`; re-score of same seal bytes for proposal loops is denied.
- Agent-visible logs may show pass/fail aggregates only for seal when policy says so; never ship-gate item text into researcher evidence snapshots.

Hermetic tests: mock observation tables labeled `split=seal`; attempt score reuse → fail.

---

## 8. Budget-matched baselines & standing nulls (S6)

Externalize thresholds in climb policy or a sibling `credit_policy.v1.json` (high-volatility numbers out of code):

- Required arms: matched-budget control; optional parallel-sample / sequential-refine **null** arms under same token/GPU budget accounting.
- Neural-necessity fixture: weight-zero / frozen-random head / shuffle labels—learned mechanism must beat null or claim is not load-bearing.
- Size-match or EG_params LCB (already partially implemented)—wire into credit positive classification, not only promotion evaluate.

Default-off for expensive arms in CI; fixtures prove wiring.

---

## 9. Pathology-keyed archive (S7)

GSME-inspired, **repo-native** (not Terminal-Bench):

- Index elites by `(WHERE × WHY)` where WHERE ∈ {data, model, decode, harness, eval, config}; WHY from typed diagnosis / gate failure codes already emitted by the stack.
- Admit only with: validity (schema + compile), activation (ran on frozen inputs), **authoritative credit** (this prompt’s credit engine) on sealed split.
- Immutable lineage: parent elite digests, observation table digests, analysis plan digests.

Do not key solely on `campaign_id` / task id.

---

## 10. Default-off experiment harnesses (S8–S9)

Implement as preregistered experiment modules + docs under `docs/design/`, claim_class `diagnostic` or `wiring` until credited:

| Experiment | Purpose | Constraint |
| --- | --- | --- |
| Semantic construct validity | Meaning metrics vs structure-only | No gate weakening |
| Representation ceiling | Capacity vs data vs decode | EG_params |
| Cluster-aware OOD | Lineage/cluster holdouts | Decontamination intact |
| Grammar-alignment audit | Rank among **legal** tokens only | **Never** override I2 singleton; never unmask illegal |
| Cache authority | Prefill/cache must not change legal domain | Fail closed |
| Structured matrix (Monarch/BTT) property + wall-clock | Property tests + optional bench | Default-off; **no speedup claim** without local measure; citation: Dao et al. ICML 2022 arXiv:2204.00595; complexity: do **not** claim \(O(n^2)\) for optimal projection |

---

## 11. Formal obligations (S10 partial)

- Bind formal preflight digests to analysis_plan / observation digests where claims are empirical.
- Non-vacuous Lean (existing SFF pattern): no hollow tautologies.
- Formal status is **not** empirical promotion by itself.

---

## 12. Citations to preserve in committed docs

Carry these into `docs/design/` measured-results / design notes (not only the agent prompt):

- Dao et al., Monarch, ICML 2022, arXiv:2204.00595  
- Wang et al., Rethinking Evaluation of Harness Evolution, arXiv:2607.12227  
- Luo et al., GSME, arXiv:2607.13683  
- Constrained diffusion / legality: arXiv:2508.10111; speculative verify arXiv:2602.00612  
- Repo laws: `AGENTS.md`, `docs/design/decode-invariants.md`, `docs/design/experiment-campaign-governance.md`, `docs/design/autotrain-climb-policy.md` (if present), `docs/design/adversarial-review.md`, `docs/design/honest-ship-eval` skill  

---

## 13. Implementation constraints (repo laws)

- Obey `MAX_RUN_MINUTES` / levers; timed-out runs are not evidence.
- No production `allow_unconstrained_fallback=True`.
- No weakening ship gates to green CI; fixture demos labeled fixture.
- Prefer extending `experiment_campaign.py` / `hillclimb.py` / climb policy over parallel registries.
- Pure predicates testable without GPU.
- `python -m scripts.verify_version_stamps --check` green; no dual path ownership.
- Use `organize-repository` / `git mv` for moves; no sprawl of duplicate harnesses.
- Skills when applicable: `honest-ship-eval`, `documenting-experiment-results`, `ponytail`, `sdlc`.

---

## 14. Validation commands (parent + S10)

Minimum focused suite (must be green):

```bash
python -m pytest -q tests/test_autoresearch/test_experiment_campaign.py
python -m pytest -q tests/test_autoresearch/test_hillclimb.py
python -m pytest -q tests/test_autoresearch/   # includes climb_policy if present
python -m pytest -q tests/test_harnesses/experiments/test_ladder_promotion.py
python -m pytest -q <new credit / sealed / archive tests>
ruff check src scripts tests
python -m scripts.verify_version_stamps --check
python -m scripts.verify_decode_invariants
python -m scripts.verify_agent_surfaces
python -m scripts.repo_policy
```

Hermetic: no network, no paid HF write, no real tokens in tests. Benchmarks dry-run / fixture-labeled non-evidentiary.

Pre-commit / changed-file hooks must pass on commit.

---

## 15. Definition of done

Complete only when **all** hold:

- [ ] Promotion/ship cannot be created from summaries alone  
- [ ] Decision-bearing stats recomputed from immutable observations + locked analysis plan  
- [ ] Caller-supplied endpoint/Holm/gate values ignored or rejected; adversarial tests prove it  
- [ ] Candidate–control pairing and baseline uncertainty are authoritative  
- [ ] Underpowered outcomes are inconclusive, not credited negatives  
- [ ] V1 summary-only evidence cannot promote  
- [ ] Governance cannot substitute for missing empirical comparison (`governed_campaign_evidence` fixed)  
- [ ] Decision-bearing artifacts have semantic validators (not kind-only placeholders)  
- [ ] Final evaluation mechanically separated from proposal/search score reuse  
- [ ] Budget comparability and standing nulls are first-class requirements (config-externalized)  
- [ ] Neural-necessity fixtures reject non-load-bearing mechanisms when claimed  
- [ ] Pathology archive admits only authoritatively credited elites with lineage  
- [ ] Metric/OOD/representation/actionability/alignment harnesses exist default-off where scoped  
- [ ] Decode/cache/matrix work preserves legality + zero-forward singleton  
- [ ] Monarch/BTT default-off, property-tested, no unmeasured speedup claim  
- [ ] Formal artifacts digest-bound and non-vacuous where touched  
- [ ] Docs + version registries + agent surfaces updated  
- [ ] Focused hermetic validation green; full suite failures classified pre-existing vs new  
- [ ] Parent completed SDLC closeout (PR squash-merged or explicit external block documented)

---

## 16. Required final report

Return:

1. Base SHA and merge SHA(s)  
2. PR/stack URLs and disposition  
3. Architecture + TCB summary  
4. Files added/changed  
5. Schema/version migration  
6. Exact tests/commands + results  
7. Adversarial attacks proven blocked  
8. Fixture experiments run and honesty class  
9. Real experiments deliberately left unrun  
10. Deviations from this prompt and why  
11. Residual risks (sealing OS assumptions, statistical power, external evaluator integrity)  
12. Citations preserved in committed docs  

---

## 17. Additional experimental hypotheses (file as issues after credit lands)

These are **not** required for the credit TCB MVP, but should be filed with enough context for later one-shots (bind each to credit_report when they claim promotion):

1. **Authoritative credit + tamper resistance** (this prompt’s core).  
2. **Score-once sealed final evaluation.**  
3. **Matched-budget simple baselines** (parallel sample / sequential refine under same budget).  
4. **Neural-necessity + randomization controls.**  
5. **Pathology-keyed credited archive (WHERE×WHY).**  
6. **Semantic-metric construct validity** vs structure-only.  
7. **Representation-sufficiency ceiling** (size vs data vs decode).  
8. **Neural actionability / inference scheduling** under symbol tables (I4).  
9. **Training–decode causal alignment** (train objective vs certified decode path).  
10. **Lineage/cluster-aware OOD** evaluation.  
11. **Proposer vs evolver strength factorial** (same model self-improve vs stronger teacher harness edits—**do not confound**).  
12. **Grammar-alignment among legal candidates only** + cache authority.  
13. **Structured-matrix property + local wall-clock** (default-off).  
14. **Formal-proof digest binding + non-vacuity.**  
15. **Search policy class**: PriorBand/BOHB/ASHA with LLM as prior only—not greedy climb (literature: HPO reviews; PriorBand NeurIPS 2023).  
16. **Data-pipeline knobs as primary search** (quality filters, mix, synthetic yield)—fixture config micro-search is second-order.  
17. **muP / width-transfer** freeze LR×width knobs.  
18. **Always-valid / group-sequential** sequential testing for continuous monitoring (avoid naive p-hacking).  
19. **Agent cannot write results store** (orchestrator-only scores; no-computation baseline for fabrication).  
20. **Historical ledger replay audit** of autotrain-wf-smoke “wins” for unreproducible claims.

---

## 18. Adversarial self-check for the implementing agent

Before claiming done, attack your own PR:

- Can I invent Holm p-values and promote? **Must be no.**  
- Can governance alone mark promotable without comparative observations? **Must be no.**  
- Can sealed scores leak into proposal matrices? **Must be no.**  
- Did any decode path lose singleton zero-forward? **Must be no.**  
- Did any “speedup” claim lack local measurement? **Must remove claim.**  
- Are high-volatility thresholds in versioned JSON rather than magic numbers in classifiers? **Must be yes.**

---

*End of coding-agent instruction. Parent owns integration and SDLC closeout.*
