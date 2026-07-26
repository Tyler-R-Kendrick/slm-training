# DSH3-18 CAP2 operator-policy rebase addendum (SLM-393)

Date: 2026-07-25
Status: design-only addendum; no runtime behavior change; no E-number allocated
Scope: inventory, issue-disposition, and machine-readable dependency map over the frozen DSH3-13 CAP2 suite (SLM-381) and the terminal DSH3-17 disposition (SLM-385)

## Decision

Kimi found the prior DSH3-14/15/17 plan stale by roughly 190 SLM-numbered experiments: much of the proposed scorer/eval surface already exists. This addendum inventories exactly what exists today (state-local heads, candidate representations, singleton bypass, the SLM-127 selector, collapse hard negatives, coverage semantics, telemetry), records update proposals for DSH3-14/15/17, and publishes a dependency map for every new DSH3 M5/M6/M7 rebase issue (DSH3-19 .. DSH3-33) so none of them start ahead of an unmet blocker or duplicate an existing callable.

## Stop-rule check

HEAD does not contain a trained typed operator policy with powered CAP2 evidence (DSH3-28/SLM-403 is unstarted); the portfolio is not obsolete and this addendum proceeds as an inventory/dependency contract, not a closure.

## Frozen suite identity (read-only; not regenerated or mutated)

```json
{
  "code_commit": "8a29de4b81da07393ec3acb3b906376baa593145",
  "evidence_id": "SLM-381.cap2_operator_v1",
  "operator_corpus_fingerprint": "5ee0d27141a3fa72be35bedbdec347f97f513c0e7af672ca4be580e5b982682e",
  "source_records_fingerprint": "f18b2fa1d9e271fcb8789c766cbb3717262353d1bbddfd37c5a1b85bca16a00e",
  "suite_hash": "16f210786bac7fd5f5edb64d13888c3cc7d634330a81b5065150e7a41fcb1d4d",
  "suite_n": 20,
  "suite_version": "cap2_operator_v1"
}
```

## Reproducibility check (DSH3-18 "Required tests")

This session ran `pytest -q tests/test_evals/test_cap2_operator.py` against the frozen DSH3-13 suite. Two environment-only blockers were found and fixed first:

- src/apps/openui_bridge/node_modules were not installed
- a sandbox-global NODE_OPTIONS=--import tsx breaks the bridge subprocess invocation in src/slm_training/dsl/operators/registry.py

After fixing both, the suite still does not reproduce: `ValueError: CAP2 generated operator corpus drifted (src/slm_training/evals/cap2_operator.py:365) -- the regenerated content_fingerprint does not match the frozen manifest fingerprint even after both environment blockers are fixed`

Treated as an open reproducibility gap of the DSH3-13 authority itself, not a DSH3-18 finding to fix. The frozen docs/design/dsh3-13-cap2-operator-eval-20260723/report.json remains the evidence of record (adversarial control: prove with behavior, never mutate the frozen suite). A dedicated follow-up should bisect why build_symbolic_operator_corpus no longer reproduces its frozen fingerprint in a clean environment.

## Inventory

| Symbol | Status | Anchor |
| --- | --- | --- |
| `models.local_action_head.LocalFlatHead` | `fixture_only` | `src/slm_training/models/local_action_head.py:180-234` |
| `models.local_action_head (GlobalMaskedHead|TernaryDigitHead|TernaryECOCHead|GrammarFactorizedHead|ResidualTritPlaneHead)` | `fixture_only` | `src/slm_training/models/local_action_head.py:122-645` |
| `models.legal_edit_scorer.DirectLegalEditPolicy.decide` | `production_ready` | `src/slm_training/models/legal_edit_scorer.py:194-216` |
| `models.legal_edit_batch (FEATURE_NAMES, _stable_scalar)` | `production_ready` | `src/slm_training/models/legal_edit_batch.py:16-38` |
| `models.legal_edit_scorer.LegalEditScorer` | `fixture_only` | `src/slm_training/models/legal_edit_scorer.py:74-146` |
| `harnesses.experiments.candidate_selector (SLM-127/EFS3-04)` | `fixture_only` | `src/slm_training/harnesses/experiments/candidate_selector.py:1-6` |
| `dsl.operators.collapse (HardNegativeOutcome, replay_collapsed_instruction, collapse_conversation_trace)` | `untested` | `src/slm_training/dsl/operators/collapse.py:37-41,261-329` |
| `dsl.operators.legal_set (OperatorSupportVerdict.UNKNOWN, LegalSetCoverage.PARTIAL)` | `production_ready` | `src/slm_training/dsl/operators/legal_set.py:43,49,226,244,429,456-457,510,518,537` |
| `dsl.operators.merge._conflict_kind (DELETE_MODIFY heuristic)` | `production_ready` | `src/slm_training/dsl/operators/merge.py:368-398` |
| `models.decode_stats.DecodeStats` | `production_ready` | `src/slm_training/models/decode_stats.py:12-175` |
| `evals.cap2_disposition.Cap2CapabilityDispositionV1 (DSH3-17/SLM-385)` | `production_ready` | `src/slm_training/evals/cap2_disposition.py:1-304` |

Notes:

- **`models.local_action_head.LocalFlatHead`**: Per-action embeddings are held in a plain python dict (`self.action_embeddings`), not an `nn.ParameterDict`/`nn.Embedding`. Verified empirically: `head.parameters()` and `head.state_dict()` omit these tensors even though autograd tracks their gradients -- not optimizer-visible or checkpoint-stable. DSH3-21 targets exactly this gap and is already In Review (PR #897) as of this addendum.
- **`models.local_action_head (GlobalMaskedHead|TernaryDigitHead|TernaryECOCHead|GrammarFactorizedHead|ResidualTritPlaneHead)`**: Real `nn.Linear`/`nn.Embedding` parameters (optimizer- and checkpoint-visible), but exercised only by unit fixtures; never wired into a training loop or the CAP2 harness. `_check_forced` gives every family the single-legal-action singleton bypass.
- **`models.legal_edit_scorer.DirectLegalEditPolicy.decide`**: `count == 1` returns a forced decision with zero model calls before scoring -- the same singleton-bypass invariant as `local_action_head._check_forced`, already present on this adjacent legal-edit path.
- **`models.legal_edit_batch (FEATURE_NAMES, _stable_scalar)`**: Candidate features -- including `successor_fingerprint` -- are SHA-derived one-dimensional scalars, not typed embeddings. DSH3-23 targets replacing this; note the current feature vector already includes `successor_fingerprint` as a plain feature, which DSH3-23 must isolate rather than assume absent.
- **`models.legal_edit_scorer.LegalEditScorer`**: Trainable `nn.Module` scorer consuming `legal_edit_batch` features; exercised in unit tests and small harness scripts (SLM-197/198/199/200), not the CAP2 M5-M7 pipeline.
- **`harnesses.experiments.candidate_selector (SLM-127/EFS3-04)`**: Module docstring self-declares 'Wiring/fixture harness only'; built for general candidate selection with calibrated abstention, never applied to operator legal-action candidates specifically.
- **`dsl.operators.collapse (HardNegativeOutcome, replay_collapsed_instruction, collapse_conversation_trace)`**: `CONFLICT`/`DIFFERENT_RESULT` hard negatives and replay-verified collapse are implemented, but no test file in `tests/` imports `slm_training.dsl.operators.collapse` -- zero direct test coverage today. DSH3-24/25/30 must add coverage when building policy rows from this module.
- **`dsl.operators.legal_set (OperatorSupportVerdict.UNKNOWN, LegalSetCoverage.PARTIAL)`**: `UNKNOWN`/`PARTIAL` are already typed and used at the legal-set/data layer. DSH3-25's real gap is the *learning* side (loss denominator, DEFER semantics) -- not the coverage taxonomy, which already exists.
- **`dsl.operators.merge._conflict_kind (DELETE_MODIFY heuristic)`**: Line 385 classifies `DELETE_MODIFY` by substring test (`"remove" in value or "delete" in value`) over operator IDs -- exactly the name-dependent heuristic DSH3-19 must replace with an effect-derived classifier. `STALE_REF` conflicts are raised earlier and independently (lines 543/550/588), so DSH3-19's priority requirement is already structurally satisfied.
- **`models.decode_stats.DecodeStats`**: `compiler_lattice_false_hard_eliminations`, `compiler_lattice_selector_regret`, `compiler_lattice_invalid_selected_over_valid`, and `constrained_dead_ends` (with related counters) already exist and are reusable as DSH3-31 features; no operator-specific replay-outcome label (illegal/executor-rejection/replay-failure/semantic-miss/DEFER) exists yet on top of them.
- **`evals.cap2_disposition.Cap2CapabilityDispositionV1 (DSH3-17/SLM-385)`**: Terminal disposition already published (CERT_CAP2 not issued, DSH4 closed). DSH3-33 must publish a *new* disposition instance over the rebased M5-M7 evidence rather than mutate this one; historical evidence at docs/design/dsh3-17-cap2-disposition-20260723/ stays immutable.

## Update proposals for DSH3-14/15/17

| Alias | Issue | Action | Superseded by |
| --- | --- | --- | --- |
| `DSH3-14` | SLM-382 | `unchanged` | - |
| `DSH3-15` | SLM-383 | `rebased` | - |
| `DSH3-17` | SLM-385 | `superseded` | DSH3-33 |

Notes:

- **`DSH3-14`**: Serialized-action control (E803) already ran and is Done; stands unchanged as the reserved-token discrete-action baseline the new M5-M7 work is measured against.
- **`DSH3-15`**: Rebase onto local_action_head.py: build the compiler-masked hierarchical head as a new `LocalActionHead` family in that module (reusing `StateContext`/`LocalActionOutput`/`ActionDecision` and the `_check_forced` singleton bypass) instead of a parallel scorer stack. Effectively superseded by the DSH3-21/22/28 track.
- **`DSH3-17`**: Terminal CERT_CAP2-rejected disposition already published from the pre-rebase E803 evidence (docs/design/dsh3-17-cap2-disposition-20260723/); DSH3-33 supersedes it once M5-M7 evidence exists. Prior historical evidence is unchanged.

## Dependency map (DSH3-19 .. DSH3-33)

`blocked_by` lists only directly-observed Linear `blockedBy` edges (never a guessed transitive closure). `ready` is true only when every direct blocker is Done, In Review, or one of the external DSH3-13/DSH3-18 prerequisites.

| Alias | Issue | Status | Blocked by | Ready |
| --- | --- | --- | --- | --- |
| `DSH3-19` | SLM-394 | `backlog` | DSH3-18 | yes |
| `DSH3-20` | SLM-395 | `backlog` | DSH3-19 | no |
| `DSH3-21` | SLM-396 | `in_review` | DSH3-18 | yes |
| `DSH3-22` | SLM-397 | `backlog` | DSH3-18, DSH3-20 | no |
| `DSH3-23` | SLM-398 | `backlog` | DSH3-21, DSH3-22 | no |
| `DSH3-24` | SLM-399 | `backlog` | DSH3-18, DSH3-22 | no |
| `DSH3-25` | SLM-400 | `backlog` | DSH3-22, DSH3-24 | no |
| `DSH3-26` | SLM-401 | `backlog` | DSH3-24, DSH3-25 | no |
| `DSH3-27` | SLM-402 | `backlog` | DSH3-22, DSH3-23, DSH3-25 | no |
| `DSH3-28` | SLM-403 | `backlog` | DSH3-21, DSH3-22, DSH3-23, DSH3-24, DSH3-25, DSH3-26, DSH3-27 | no |
| `DSH3-29` | SLM-404 | `backlog` | DSH3-25, DSH3-28 | no |
| `DSH3-30` | SLM-405 | `backlog` | DSH3-24, DSH3-28 | no |
| `DSH3-31` | SLM-406 | `backlog` | DSH3-28 | no |
| `DSH3-32` | SLM-407 | `backlog` | DSH3-28 | no |
| `DSH3-33` | SLM-408 | `backlog` | DSH3-18, DSH3-19, DSH3-21, DSH3-22, DSH3-25, DSH3-26, DSH3-27, DSH3-28, DSH3-32 | no |

As of this addendum, only `DSH3-19` and `DSH3-21` are ready; `DSH3-21` (SLM-396) had already started and reached In Review (PR #897) before `DSH3-18` was claimed in this session -- recorded here rather than silently corrected, since Linear state is the source of truth, not this doc.

## Non-goals confirmed

No runtime behavior changed. No frozen-suite edit. No quality/efficiency claim. No E-number allocated.
