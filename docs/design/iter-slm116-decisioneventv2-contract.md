# SLM-116 / LDI0-02: DecisionEventV2 contract memo

**Status:** Specification implemented. No training or quality claim is made here;
this memo documents the V2 schema, identity rules, and a small fixture corpus.

## What changed

`src/slm_training/harnesses/preference/local_decisions.py` now carries the
`DecisionEventV2` state + action-evidence contract alongside the existing V1
implementation. V1 corpora remain readable and produce the same content
fingerprints. A one-way V1→V2 migration utility preserves existing evidence but
marks it as incomplete when rollout/verifier evidence is missing.

## Fixture corpus

The following JSONL rows are a minimal valid V2 corpus. It contains one exact
decision state and two independently replayed actions. Rewards are deliberately
small and synthetic — they prove schema round-tripping and materializer
behavior, not model quality.

```json
{"evidence_kind": "counterfactual", "outcomes": [{"action_id": 4, "confidence_interval": [0.9, 1.0], "continuation_seeds": [7], "evidence_confidence": 1.0, "evidence_ids": ["ev-good-1"], "legal": true, "mean_value": 0.9, "outcome_hashes": ["hash-good"], "reward_vectors": [{"reward": 0.9, "fidelity": 0.85}], "rollout_policy_sha": "policy-sha", "state_id": "state-abc", "verifier_vectors": [{"G0": "pass"}]}, {"action_id": 5, "confidence_interval": [0.1, 0.3], "continuation_seeds": [8], "evidence_confidence": 1.0, "evidence_ids": ["ev-bad-1"], "legal": true, "mean_value": 0.2, "outcome_hashes": ["hash-bad"], "reward_vectors": [{"reward": 0.2, "fidelity": 0.2}], "rollout_policy_sha": "policy-sha", "state_id": "state-abc", "verifier_vectors": [{"G0": "fail"}]}], "state": {"abstract_state_role": "root_child", "architecture": "twotower", "canvas_ids": [1, 2, 2], "context_text": "Generate a card", "decode_config_hash": "decode-sha", "decision_kind": "component", "decision_position": 1, "generation_step": 3, "grammar_state_hash": "grammar-sha", "group_id": "record-1", "legal_action_ids": [3, 4, 5], "policy_checkpoint_sha": "policy-sha", "split": "train", "state_id": "state-abc", "tokenizer_sha": "tokenizer-sha", "verifier_bundle_hash": "verifier-sha"}, "version": 2}
```

The `state_id` above is abbreviated; the canonical value is the SHA-256 over the
immutable state/runtime fields only (no good/bad labels, no outcome order, no
ordinal file position).

## Materializer divergence

The same evidence produces different objective views depending on the
materializer config:

| Materializer | good | bad | ambiguous | unobserved |
| --- | --- | --- | --- | --- |
| Pareto `reward>=0.5, fidelity>=0.5` | 4 | 5 | 3 | — |
| Threshold `mean>=0.5, lower>=0.5` | 4 | 5 | 3 | — |
| Single best/worst | 4 | 5 | 3 | — |
| Set partition Pareto | 4 | 5 | 3 | — |
| Constraint-shadow diagnostic | diagnostic-only | diagnostic-only | — | 3 |

For this tiny corpus the numeric materializers agree; on richer corpora they
diverge by design. The manifest records separate `state_fingerprint`,
`evidence_fingerprint`, and `objective_fingerprint` values so that a change in
row order changes none of them, a change in evidence metadata changes the
evidence fingerprint, and a change in reward/verdict values changes the
objective fingerprint.

## Identity invariants verified by tests

- Reordering action outcomes leaves `state_id` unchanged.
- Two samples for the same state merge into one state table.
- Conflicting `evidence_kind` under one `state_id` is rejected.
- Unknown schema fields are rejected on load.
- Counterfactual outcomes must be legal actions; constraint-shadow outcomes may
  be illegal and are marked non-semantic.
- V1 migration is deterministic, idempotent, and reports incomplete evidence.

## Honest boundary

This change is schema and compatibility only:

- No new counterfactual rollout policy.
- No trainer objective change beyond a V1-view compatibility shim.
- No adapter/LoRA/PEFT implementation.
- No end-to-end train, eval, matrix, bench, or checkpoint claim.
- The unchanged ship gates remain the only promotion authority.

## Verification run

```bash
.venv/bin/python -m pytest tests/test_harnesses/preference/test_local_decisions.py -q
.venv/bin/python -m pytest tests/test_harnesses/preference -k "counterfactual or decision" -q
.venv/bin/python -m ruff check src/slm_training/harnesses/preference tests/test_harnesses/preference
.venv/bin/python -m scripts.repo_policy
.githooks/check-changed
```

All passed at implementation time.
