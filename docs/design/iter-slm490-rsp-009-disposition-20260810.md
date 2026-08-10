# SLM-490 (RSP-009): Cross-experiment EXP-SR disposition (slm490_rsp009_fixture)

**Schema:** `rsp009_cross_experiment_disposition/v1` / `mechanism_disposition_report/v1`

**Matrix set:** `slm490_rsp009_disposition`

**Status:** fixture — disposition audit only; no ship-gate bypass.

**Evidence cutoff commit:** `874b95d95764727d14420080d50141dc602b95f8`

**Generated at:** 2026-08-10T23:28:47.243601Z

## Executive finding

RSP-009 audited all twelve EXP-SR catalogue families (exp-sr-1..12) using committed sibling evidence. OpenUI-scoped rows (localization, predictor, calibration, static summary, QD) are separated from symbolic-pack rows (e-graph, macro library, PySR) and from the dual-scope EXP-SR-12 portability certification. No row satisfies adopt_primary or adopt_optional under SGS-009 fail-closed rules on fixture/scratch/blocked evidence — champion pointers stay empty.

## Disposition table (EXP-SR-1..12)

| Family | Pack scope | Disposition | Default | Evidence class | Rationale |
| --- | --- | --- | --- | --- | --- |
| exp-sr-1 | openui | retain_diagnostic | off | fixture | Fixture oracle localization authorized roles/bindings for later predictor work; claim_class=fixture — retain as OpenUI diagnostic, never default-on. |
| exp-sr-2 | openui | reject | off | fixture | Catalogue falsifier holds: predictor F1 does not beat the frequency/no-prediction baseline on the held-out slice (fixture evidence). |
| exp-sr-3 | openui | revise_and_retest | off | fixture | SLM-491 external-blocked prepared package filed (SIE-005 pin + SIE-006 replay verified); real blinded human/provider calibration remains unavailable — kappa/calibration metrics stay null. |
| exp-sr-4 | openui, symbolic_regression | retain_diagnostic | off | fixture | Witness CEGIS cleared matched controls on the fixture corpus (recommendation='adopt-optional'), but claim_class=fixture caps disposition at retain_diagnostic per SGS-009 — no adopt_primary/adopt_optional. |
| exp-sr-5 | openui, symbolic_regression | reject | off | fixture | Symbolic VoC controller failed the catalogue falsifier versus the hand-threshold control on fixture decode traces. |
| exp-sr-6 | openui, symbolic_regression | retain_diagnostic | off | fixture | Neural-guided ordering beat left-to-right on the bounded fixture (recommendation='adopt-optional'), but claim_class=fixture — retain default-off. |
| exp-sr-7 | openui | retain_diagnostic | off | fixture | Packed semantic summary achieved exact domain parity on the differential fixture (certified=true) but cold/warm cost evidence is fixture-scale only — retain OpenUI diagnostic, not production default. |
| exp-sr-8 | symbolic_regression | retain_diagnostic | off | fixture | E-graph regions match canonical-hash dedup on the fixture with certified rewrites (not_adopted_stays_experimental); no Pareto win — isolated symbolic-pack experiment only. |
| exp-sr-9 | symbolic_regression | retain_diagnostic | off | fixture | Certified macro library fixture shows size reduction with zero semantics regressions; claim_class=fixture — diagnostic for symbolic-pack search only. |
| exp-sr-10 | openui | reject | off | fixture | Quality-diversity coverage did not beat matched-compute rejection sampling on the OpenUI ProgramSpec fixture grid. |
| exp-sr-11 | symbolic_regression | blocked | blocked | blocked | SLM-492 external-blocked prepared package filed (RSP-007 + SRP-011 isolation replayable); PySR/Julia remain unavailable — gap not scored as a benchmark loss. |
| exp-sr-12:openui | openui | retain_diagnostic | off | scratch | OpenUI pack passes shared-seam readiness checks in the RSP-008 diagnostic; reference control only — not a promotion of second-pack portability. |
| exp-sr-12:symbolic_regression | symbolic_regression | revise_and_retest | off | scratch | Symbolic-regression pack still requires pack-specific seam forks (['shadow_dsl_packs_registry', 'openui_pinned_language_contract', 'sygus_inspired_not_conformance']); zero-fork certification falsified. |
| exp-sr-3 | openui | revise_and_retest | off | blocked | SLM-491 filed an honest external-blocked prepared package (docs/design/iter-slm491-exp-sr-3-external-blocked-closeout-20260810.json): SIE-005 pin + SIE-006 replay verified, but real blinded human/provider calibration remains unavailable — kappa/calibration metrics stay null, not mocked-as-real. |
| exp-sr-11 | symbolic_regression | blocked | blocked | blocked | SLM-492 filed an honest external-blocked prepared package (docs/design/iter-slm492-exp-sr-11-external-blocked-closeout-20260810.json): RSP-007 harness + SRP-011 adapter isolation replayable, but PySR/Julia remain unavailable — gap is not scored as a benchmark loss; no SOTA claim. |

## Rejected or blocked

- `exp-sr-2`
- `exp-sr-5`
- `exp-sr-10`
- `exp-sr-11`

## Supersessions

- `exp-sr-3`@`874b95d95764727d14420080d50141dc602b95f8` → `exp-sr-3`@`874b95d95764727d14420080d50141dc602b95f8`: SLM-491 external-blocked prepared package closeout — evaluator_calibration_protocol follow-up evidence filed
- `exp-sr-11`@`874b95d95764727d14420080d50141dc602b95f8` → `exp-sr-11`@`874b95d95764727d14420080d50141dc602b95f8`: SLM-492 external-blocked prepared package closeout — pysr_srbench_adapter follow-up evidence filed

## Follow-up gaps

- Eliminate shared-seam pack forks for symbolic_regression before EXP-SR-12 can certify second-pack portability.
- Measured (non-fixture) witness CEGIS and bounded-search confirmation on live decode/search loops (EXP-SR-4/6).
- Production cold/warm confirmation for packed semantic summary beyond the fixture differential corpus (EXP-SR-7).
- Re-open EXP-SR-2 only if a learned predictor beats the frequency baseline on a real held-out prompt slice.

## Champion / default pointers

_Empty — no adopt_primary or adopt_optional rows; rejected/blocked cannot enter champion/default pointers._

## Reproducibility

```bash
python -m scripts.run_rsp009_disposition --mode fixture
```
