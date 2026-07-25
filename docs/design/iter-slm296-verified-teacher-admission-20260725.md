# SLM-296 (AP-013): verifier-filtered teacher-admission wiring evidence

**Claim class:** wiring / fixture only

**Status:** `yield_limitation_no_real_teacher_available`

## Recipe

```json
{
  "seeds": [
    0,
    1,
    2
  ],
  "components": [
    "TextContent",
    "Button",
    "Separator"
  ],
  "teacher_model_family": "deterministic_programspec_generator_v1",
  "locked_holdout_seed": 999
}
```

## Admission counts

| metric | value |
| --- | --- |
| proposed | 27 |
| locked_set_overlap_rejected | 23 |
| accepted | 4 |
| accepted_gold_or_silver | 0 |
| unique_accepted_canonical_roots | 4 |
| tier_counts | {'BRONZE': 4} |
| gate_failure_counts | {} |

## Family-independence control

- same-family judge independent: False
- different-family judge independent: True
- shadow-judge sample fully quarantined: True (10/10)

## Limitations

- No real separate-family teacher is available (no network/GPU in this sandbox); candidates come from the repository's own deterministic ProgramSpec generator, explicitly not a teacher. SLM-266 already measured that a local CPU teacher run is infeasible here (>= 11 days extrapolated for 10k requests).
- The >= 500 accepted-record target is not attempted at this scale; this is a bounded wiring campaign only.
- No fine-tuning run occurs and no grammar-off/meaning-v2 delta is measured or claimed.
- Gold/Silver promotion is structurally blocked here because 'teacher' is a weak source with no human_audit_passed evidence -- every accepted row is Bronze at best, by construction of the real verifier stack, not by a claim made in this report.
