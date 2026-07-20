# SLM-150 (SPV2-02): Global semantic energy/value critic baseline

Matrix set: `slm150-spv2-02-global-semantic-critic`
Version: `spv2-02-v1`
Status: **wiring_only / fixture evidence only**

## What this exercises

A tiny global semantic energy/value critic that is ready to consume the SPV2-01
hard-valid contrast corpus when it exists. The current evidence uses a
synthetic fixture corpus and does **not** claim ship readiness.

## Model contract

- **Sign convention**: lower energy is better; ``value = -energy`` is exposed
  explicitly so signs are never mixed.
- **Shared MLP**: context encoder + scalar energy head + confidence head
  (sigmoid) + per-factor heads.
- **Factor heads**: ``coverage``, ``roles``, ``topology``, ``bindings``,
  ``contract``.
- **Inference-safe features only**: ``featurize`` ignores any key that looks
  like gold/evaluator leakage (``gold_*``, ``target_*``, ``judge_*``,
  ``accepted_output``, ``verdict``, ``semantic_score``, ``label``,
  ``factor_targets``).
- **Abstention**: the critic abstains when the pack is unsupported or when
  confidence is below ``confidence_threshold``.
- **Fail-closed checkpoints**: ``save`` / ``from_checkpoint`` preserve the
  schema, config, and weights; schema/config mismatches raise.

## Losses

- ``global_critic_pairwise_loss``: within each group, positive labels should
  have lower energy than negatives; ``UNKNOWN`` (-1) and energy ties are
  skipped.
- ``global_critic_listwise_loss``: per-group softmax over ``-energies`` with a
  uniform target over positive labels; groups with no positive are skipped;
  ``UNKNOWN`` rows are excluded from the softmax.
- ``global_critic_factor_loss``: masked MSE over the factor head predictions.

## Fixture results

| Check | Result |
| --- | --- |
| Energy/value sign convention | OK |
| Factor head count | 5 |
| Pairwise/listwise/factor losses finite | OK |
| Checkpoint round-trip | OK |
| Unsupported-pack abstention | OK |
| All-candidate abstention | OK |
| Deterministic reranking | OK |
| ``lambda_global=0`` uses local scores only | OK |
| Gold/evaluator keys ignored by featurize | OK |
| Batch/scalar score equivalence | OK |
| Factor heads do not change energy/value | OK |

## Fixture corpus

- ``make_fixture_examples(n_groups=16, candidates_per_group=4)`` produces 64
  deterministic contrast examples.
- Multiple positives and ``UNKNOWN`` labels are present per group.
- Families and severities are varied across groups.

## Recipe

- Device: CPU
- ``d_model``: 64
- ``hidden_dim``: 64
- ``num_factors``: 5
- ``scorer_id``: ``global-semantic-critic-v1``
- Fixture groups: 16
- Candidates per group: 4

## Honest verdict

``fixture_wiring`` — no ship claim. The SPV2-01 hard-valid contrast corpus is
absent from the repo, so the fixture corpus is synthetic; the implementation is
ready to consume the real corpus when it exists.

## Verification commands

```bash
.venv/bin/python -m pytest tests/test_models/test_global_semantic_critic.py tests/test_models/test_global_semantic_critic_selector.py -q
.venv/bin/python -m scripts.verify_version_stamps --check
```
