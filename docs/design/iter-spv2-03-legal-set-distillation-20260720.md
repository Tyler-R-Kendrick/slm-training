# SLM-151 SPV2-03: Dense legal-set distillation fixture

**Status:** fixture / wiring only.  
**Claim class:** `wiring`.  
**Honest verdict:** `fixture_wiring`.

This change implements a minimal, fixture-only dense legal-set knowledge-distillation baseline. It is **not** a ship-ready training pipeline and does not run an external teacher model. Real teacher scoring is deferred to the SLM-108 external scorer.

## What this exercises

- `LegalSetKLConfig` and `LegalSetDistillExample` dataclasses.
- `legal_set_teacher_distribution`: build a teacher probability vector restricted to the compiler-owned legal action set, with epsilon smoothing for zero-probability stability.
- `legal_set_kl_loss`: slice both student and teacher to the legal set and compute `F.kl_div(log_softmax(student/t), teacher_probs)`.
- `legal_set_kl_loss_from_examples`: batch over examples that may have different legal sets.
- `train_legal_set_kl_fixture`: tiny Adam loop that updates a student network to match a teacher network over synthetic legal sets.
- `TeacherTraceManifest` and `LegalSetTeacherTrace`: deterministic synthetic teacher-trace schema plus JSON/JSONL round-trip helpers.

## Legal-set KL objective

For one decision state with legal actions `L(a)`:

```
student_log_probs = log_softmax(student_logits[L(a)] / T)
teacher_probs     = softmax(teacher_logits[L(a)] / T)   (or renormalized prob input)
loss              = kl_div(student_log_probs, teacher_probs, reduction="batchmean")
```

Logits outside `L(a)` do not participate in the loss or its gradient.

## Teacher trace schema

- `TeacherTraceManifest`: provenance envelope (model id, revision, prompt-template hash, pack id, compiler version, schema version, timestamp, provenance).
- `LegalSetTeacherTrace`: one synthetic decision with `state_id`, `prefix_ids`, `legal_action_ids`, optional `teacher_logits` / `teacher_probs`, `accepted_action_ids`, `source`, `coverage`, and an `approximate` flag.

The fixture builder emits deterministic traces; some rows use `teacher_probs` instead of logits and some rows are flagged `approximate=True` so consumers can filter them if desired.

## Fixture recipe

| Key | Value |
| --- | --- |
| `n_states` | 16 |
| `vocab_size` | 32 |
| `fixture_steps` | 20 |
| `fixture_lr` | 0.05 |
| `teacher_source` | synthetic fixture |
| `optimizer` | Adam |
| `temperature` | 1.0 |
| `reduction` | `batchmean` |

## Fixture result table

| Metric | Value |
| --- | --- |
| `n` | 16 |
| `initial_kl` | 0.090226 |
| `final_kl` | 0.001196 |
| `mean_legal_set_size` | 4.438 |
| `zero_kl_case` | 0.0 |
| `illegal_ignore_delta` | 0.0 |
| `teacher_prob_equiv_delta` | 0.0 |
| `empty_set_kl` | 0.0 |
| `approximate_traces` | 4 |
| `prob_trace_count` | 3 |

The fixture trainer reduces KL from the random-initialization mismatch toward the synthetic teacher, which is the only wiring claim made here.

## Caveats

- No real teacher model is downloaded or scored.
- No TwoTower checkpoint is trained or promoted.
- No ship gate is evaluated or weakened.
- The external scorer integration from SLM-108 is not wired in this baseline.

## Verification commands

```bash
python -m pytest tests/test_harnesses/distill/test_legal_set_kl.py -q
python -m scripts.verify_version_stamps --check
```

Both commands passed on this branch at the time of writing.
