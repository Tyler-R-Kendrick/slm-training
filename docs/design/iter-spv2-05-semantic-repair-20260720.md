# SLM-153 (SPV2-05): Verifier-guided minimal semantic repair fixture

**Status:** fixture / wiring only.  
**Claim class:** `wiring`.  
**Honest verdict:** `fixture_wiring`.

This change implements a minimal, fixture-only verifier-guided semantic-repair baseline. It is **not** a ship-ready training pipeline and does not run an external verifier-backed counterfactual rollout. Real action-value scoring is deferred to SLM-131 / VSS finite replay.

## What this exercises

- `SemanticRepairRecordV1` schema with replayable failure evidence, conflict slice, legal edits, and accepted/oracle edit sets.
- `build_repair_records_from_corruption`: turn the existing hard-valid corruption taxonomy into repair records.
- `RepairFeatureExtractor`: torch-free feature extraction over the broken program, the candidate repair, and the conflict slice.
- `SemanticRepairScorer`: tiny MLP scorer that consumes the feature vector and ranks legal edits.
- Baseline policies: `oracle`, `edit_distance`, and `random`.
- `train_repair_policy_fixture`: a minimal Adam loop that ranks accepted edits above synthetic non-accepted edits.

## Repair record contract

Each record carries:

- `source_fingerprint` of the original hard-valid program.
- `failure_evidence`: reason-coded verifier/contract observations with analyzer provenance.
- `conflict_slice`: stage, failing nodes, dependency frontier, protected nodes, and completeness class.
- `legal_edits`: the complete live acceptable repair set from the corruption taxonomy.
- `accepted_edit_ids`: all known acceptable repairs (empty means `UNKNOWN`).
- `oracle_edit_id`: the lowest-cost accepted repair.
- `lineage`: operator, family, AST path, and edit distance.

## Fixture recipe

| Key | Value |
| --- | --- |
| `source_program` | `root = Stack([cta], "column")\ncta = Button(":cta.label")` |
| `fixture_steps` | 40 |
| `fixture_lr` | 0.05 |
| `backend` | cpu |
| `scorer_id` | semantic-repair-scorer-v1 |

## Fixture result table

| Metric | Value |
| --- | --- |
| `n_records` | 39 |
| `n_families` | 6 |
| `oracle_success_rate` | 1.000 |
| `oracle_mean_cost` | 3.95 |
| `oracle_unknown_rate` | 0.000 |
| `edit_distance_success_rate` | 1.000 |
| `edit_distance_mean_cost` | 3.95 |
| `edit_distance_unknown_rate` | 0.000 |
| `random_success_rate` | 1.000 |
| `random_mean_cost` | 3.95 |
| `random_unknown_rate` | 0.000 |
| `learned_initial_loss` | 0.608378 |
| `learned_final_loss` | 0.001466 |
| `learned_n_decisions` | 24 |
| `learned_accepted_rank_one` | 8 |
| `learned_accepted_outrank_rate` | 1.000 |

## Caveats

- No real verifier-backed counterfactual rollouts are run.
- No TwoTower checkpoint is trained or promoted.
- No ship gate is evaluated or weakened.
- The SLM-131 / VSS finite-replay integration is not wired in this baseline.

## Verification commands

```bash
python -m pytest tests/test_harnesses/distill/test_semantic_repair.py -q
python -m scripts.verify_version_stamps --check
```

Both commands passed on this branch at the time of writing.