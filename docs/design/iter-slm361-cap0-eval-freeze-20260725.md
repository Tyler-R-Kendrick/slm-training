# SLM-361 (DSH1-09): CAP0 eval-suite freeze + anti-collapse gates

Date: 2026-07-25 · Run class: contract_fixture · Decision: supported
Machine JSON: `iter-slm361-cap0-eval-freeze-20260725.json`

Freezes the CAP0 eval suite as six immutable subsuites built deterministically
from the DSH1 artifact pipeline, with zero-tolerance data invariants and
versioned anti-collapse gates
(`src/slm_training/harnesses/train_data/cap0_eval_freeze.py`, schema
`cap0_eval_freeze/v1`).

## Fixture scoreboard (5 eval-split answers, seed 7)

| Subsuite | Rows | Source |
| --- | ---: | --- |
| atomic_coverage | 5 | DSH1-01/02 authority + production traces |
| expanded_canonicalization | 17 | DSH1-05 proven equivalence transforms |
| compositional_held_out | 3 | DSH1-06 `COMPOSE_DOCUMENT` (held out) |
| prefix_completion | 6 | DSH1-07 certificates + verified accepted sets |
| marker_permutation | 1 | DSH1-04 authority derangement (must reject) |
| anti_identity | 1 | DSH1-08 same-symbols/different-structure (must reject) |

- Rows are sha256-addressed under a tamper-evident manifest
  (`rows_sha256` in the JSON); `verify_suite_integrity` is fail-closed and any
  payload or manifest tamper raises `SuiteIntegrityError` (tested).
- Train/eval root families are disjoint by construction: only families the
  `RootFamilySplitPolicyV1` assigns to an eval split are admitted; a leaked
  train-family row is a `split_leakage` rejection artifact (tested).

## Zero-tolerance invariants

`audit_suite` returns a `RejectionArtifactV1` for **every** finding — the
suite is invalid, nothing is silently dropped. Each kind is tested to fire:
`free_form_target`, `undeclared_marker`, `parse_static_mismatch`,
`canonical_mismatch`, `unresolved_scope`, `missing_provenance`,
`split_leakage`. Clean fixture: 0 findings, 0 stop-rule violations.

Stop rules (`stop_rule_violations`): any split leakage, any prefix row
without a verified compiler certificate (`unproven_prefix`), or any
accepted-set target rewarding an incomplete shortest output
(`incomplete_shortest_reward`) invalidates the suite (both tested).

## Anti-collapse gates (`cap0_gate/v1`, retention `cap0_gate_retention/v1`)

Thresholds: canonical AST equivalence ≥ 0.95, accepted-set mass ≥ 0.90,
production coverage ≥ 1.0, structure ≥ 0.90, binding-aware ≥ 0.90,
empty/trivial rate ≤ 0.05, diversity ≥ 0.50, identity-baseline score ≤ 0.50.
Retention semantics: a later gate version may only tighten thresholds or raise
preregistered n; loosening requires a new retention policy version.

- Constructed identity-only, unchanged-input, and empty/trivial baselines all
  FAIL the gate (tested).
- Preregistered n per subsuite (100–400; see JSON) with Wilson 95% confidence
  bounds. Perfect metrics at fixture n report `insufficient_n` and can never
  issue `CERT_CAP0`; preregistered n certifies (both tested).

## Claim limits

Fixture-scale contract evidence only. Fixture n cannot issue `CERT_CAP0`; no
corpus publication, no CAP0 training run, no model evaluation, no ship-gate
claim.

## Commands

```bash
python -m pytest -q tests/test_harnesses/train_data/test_cap0_eval_freeze.py
python -m pytest -q tests/test_harnesses/train_data tests/test_dsl
python -m scripts.verify_version_stamps --check
python -m scripts.repo_policy
python -m ruff check src/slm_training/harnesses/train_data/cap0_eval_freeze.py tests/test_harnesses/train_data/test_cap0_eval_freeze.py
git diff --check
```
