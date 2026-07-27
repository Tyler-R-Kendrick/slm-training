# SLM-379 (DSH3-11): Operator frames — schema-grounded NL edit turns

- **Issue:** SLM-379 / DSH3-11 — schema-grounded NL descriptions of operator
  applications (paraphrase EXACT operator effects; never let an LLM define
  transformation semantics).
- **Date:** 2026-07-25. **Run class:** contract fixture (offline, no network).
- **Decision:** supported at contract/fixture scale, with the stop rule
  exercised and recorded (`symbolic_cap2_only`).
- **Result JSON:** `iter-slm379-operator-frames-20260725.json` (carries
  `version_stamp`, schema `version_stamp/v1`).

## What was built

1. **`src/slm_training/dsl/operator_frame.py` — `OperatorFrameV1`.**
   Deterministic, fail-closed derivation from the DSH3 compiler evidence
   (`AstOperatorV1` declaration, typed bound arguments, before-state identity,
   `ActionEffectV1`, `ReferenceTableV1`):
   - **required edit facts** — every effect delta (target, delta kind,
     before/after) plus consumed/produced roles and binders;
   - **forbidden edit facts** — every reference-table entry the effect does
     NOT touch, claimed only under `compiler_coverage=exact`; empty under
     bounded/approximate coverage (no overclaim);
   - **state-reference facts** — bound arguments + effect targets resolved
     through the typed descriptor tables (inference-visible compiler facts
     only; undescribed refs raise `UnsupportedOperatorFrameError`).
   Every fact carries exact provenance (source, slot id, delta kind,
   descriptor fingerprint). Frame equivalence + `frame_conflicts` included.
2. **`src/slm_training/harnesses/train_data/operator_nl_turns.py` — NL
   generation + admission.** Prompt candidates are rendered from frame facts
   only through the **reused CAP1 contract**: DSH2-03 provider protocol,
   `ParaphraseRequestV1`/`ParaphraseResponseV1` provenance,
   `validate_raw_response`/`redact_text`, and the SLM-365 offline fixture
   provider pattern (`OperatorNlFixtureProviderV1`, fully offline,
   credential-free). Admission under the current state + exact
   `OperatorLegalSetV1`:
   - target application must be in the legal set (`missing_legal_membership`)
     and in the declared accepted set (`not_in_accepted_set`);
   - every state reference must be unambiguous or accepted-set-scored —
     a descriptor shared with non-equivalent legal actions rejects
     `ambiguous_state_reference`;
   - the prompt must not be compatible with any non-equivalent application
     outside the accepted set (`compatible_with_non_equivalent`); several
     accepted matches require the prompt to name the accepted SET
     (`accepted_set_not_named`);
   - declared `OperatorLeakDetectorV1` rejects serialized actions, operator
     ids, opaque/request ids, digest surfaces, marker surfaces, and
     provenance fields (`leak`).
   Targets stay **symbol-only**: the reserved `OPERATOR <id> <typed args>`
   serialization, the resulting canonical AST, or a declared dual view.
   Natural language never enters targets (asserted in tests).

## Disambiguation report (fixture operator families)

| Family | Candidates | Admitted | Rate | Disposition |
| --- | --- | --- | --- | --- |
| fixture_unambiguous_single_accept | 2 | 2 | 1.0 | nl_prompts_admitted |
| fixture_equivalent_accepted_set | 2 | 2 | 1.0 | nl_prompts_admitted (set named) |
| fixture_non_equivalent_single_accept | 2 | 0 | 0.0 | symbolic_cap2_only |
| fixture_ambiguous_shared_descriptor | 2 | 0 | 0.0 | symbolic_cap2_only |

Overall disambiguation rate: **0.5** (4/8 candidates admitted).

Reading: prompts that uniquely identify one accepted application — or that
explicitly name the accepted set of equivalent applications — admit at rate
1.0. Families whose prompts are mechanically compatible with non-equivalent
applications, or whose state references cannot be disambiguated through the
typed descriptor tables, admit nothing and record the stop-rule disposition
`symbolic_cap2_only`: symbolic CAP2 is retained for those families, honestly,
exactly as the issue's stop rule requires. This is consistent with SLM-368's
fixture-scale CERT_CAP1 rejection — NL CAP2 remains closed at fixture scale;
this issue lands the contract-level machinery and measures where
disambiguation holds.

## Tests

- `tests/test_dsl/test_operator_frame.py` (6): derivation determinism +
  provenance, required/forbidden facts from `ActionEffectV1`, bounded
  coverage no-overclaim, conflict detection, stale table / undescribed ref
  fail-closed.
- `tests/test_harnesses/train_data/test_operator_nl_turns.py` (12):
  admission recovers the accepted application, symbol-only targets across all
  three views, ambiguous reference rejected, non-equivalent-compatible
  rejected, accepted-set naming admitted, equivalent-set single-accept
  admitted without naming, missing legal membership, not-in-accepted-set,
  leak (operator id + marker surface), classifier scoring, per-family
  disambiguation report with stop-rule disposition, determinism.

## Version stamp

Component `harness.experiments.slm379_operator_frames` **v1** registered in
`src/slm_training/resources/versions.json`; `harness.train_data` carries a
same-version `no-bump:` history note (additive module only, no existing
train_data path modified). DSH3 operator components
(`dsl.operators.contracts`/`legal_set`/…) are untouched — no bump forced.
