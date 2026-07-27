# Verifier-filtered teacher-admission wiring evidence

SLM-296 (AP-013) asks for >= 500 accepted records from a capable
separate-family teacher, filtered through G0-G10 plus independent judgment
and human audit, then a matched-budget fine-tune with a grammar-on/off
evaluation. This sandbox has no network access to a real external teacher
and no GPU. SLM-266 (`docs/design/teacher-program-data.md`) already measured
that even a local CPU teacher run is infeasible here: an 8-token probe took
95.65s, and a 10k-request campaign would take at least 11 continuous days.

This issue is scoped down to a bounded wiring slice instead: a deterministic
offline candidate proposer (explicitly *not* a real teacher) feeding the
real G0-G12 verifier stack, exercising the two safety mechanisms the
acceptance criteria actually test -- that teacher self-judgment cannot
promote a record, and that decontamination against a locked holdout works.

The committed [disposition JSON](iter-slm296-verified-teacher-admission-20260725.json)
records the exact recipe and counts. Rerun the exact command to regenerate it:

```bash
python -m scripts.run_verified_teacher_admission --mode fixture
```

## Measured fixture-scale campaign -- 2026-07-25

27 candidates were proposed across 3 seeds from a 3-component ProgramSpec
grid (the same bounded generator used for SLM-267). A disjoint-seed "locked
holdout" was built from the same grid; because the generator's candidate
grid is finite and shared across seeds (a finding already established in
SLM-267), most proposed candidates also appear in the holdout by canonical
root, so 23/27 were correctly excluded as locked-set overlap before any
tier was assigned. Of the 4 remaining, all 4 passed every deterministic gate
(G0-G10) and were admitted at **Bronze** tier -- never Gold or Silver, since
`"teacher"` is a weak source in the verifier stack's tier rule
(`_WEAK_SOURCES`) and no `human_audit_passed` evidence was supplied.

The family-independence control re-verified a 10-candidate sample with
`independent_judge_passed` wired from `check_family_independence` when the
(hypothetical) judge shares the teacher's model family: **all 10** were
quarantined by the G11 gate. The same check against a genuinely different
family returns `True` (would pass, never exercised against real candidates
here since no real judge exists).

## Honest scope

This is wiring evidence only:

- The candidate proposer is the repository's own deterministic ProgramSpec
  generator, not a separate-family teacher; `TEACHER_MODEL_FAMILY` is
  labeled `deterministic_programspec_generator_v1` throughout.
- The >= 500 accepted-record target is not attempted at this scale.
- No fine-tuning occurs and no grammar-off/meaning-v2 delta is measured.
- Gold/Silver promotion is structurally blocked by the real verifier stack's
  tier rule for weak sources, not by an accounting claim made in this report.
- Family independence and decontamination are exercised with real pass/fail
  logic (`check_family_independence`, canonical-root locked-set overlap),
  not asserted narratively.

Disposition: **`yield_limitation_no_real_teacher_available`** -- the
mechanism (verifier-filtered admission, tier-capping for weak sources,
same-family self-judgment quarantine, locked-set decontamination) is real,
tested, and reproducible; the accepted-record scale and grammar-on/off
fine-tune comparison SLM-296 asks for remain unaddressed in this sandbox.
