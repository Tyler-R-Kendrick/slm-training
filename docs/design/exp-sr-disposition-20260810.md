# EXP-SR cross-experiment disposition (RSP-009 / SLM-490)

**Schema:** `mechanism_disposition_report/v1` (SGS-009, `harnesses/experiments/mechanism_disposition_report.py` -- reused, not forked)
**Matrix set:** exp-sr-1 through exp-sr-12 (`harnesses/experiments/exp_sr_catalogue.py`)
**Evidence cutoff:** `2026-08-10`
**Claim class:** every disposed mechanism stayed fixture-scale (mock external judges, synthetic human raters, small in-process corpora); none is `adopt_primary`/`adopt_optional` -- that combination with `evidence_class=fixture` is structurally rejected by `build_mechanism_disposition_record`'s own validator, not just this report's judgment.

## Canonical artifact pair

- `docs/design/exp-sr-disposition-20260810.json` -- the structured `MechanismDispositionReportV1` (hash-verified).
- `docs/design/exp-sr-disposition-20260810.md` -- this document.

## Executive finding

Of 12 registered EXP-SR families, 7 are retained as honest positive-or-mixed diagnostic signals (oracle_localization, evaluator_calibration_protocol, semantic_repair_witness_cegis, bounded_search_neural_prior, packed_semantic_summary, symbolic_egraph, symbolic_macro_library), 4 are rejected on measured evidence (prompt_factor_predictor, compute_state_controller, quality_diversity_corpus, second_pack_portability), and 1 never executed and is disposed blocked (pysr_srbench_adapter). No mechanism reached adopt_primary/adopt_optional: every campaign this initiative ran stayed fixture-scale, and fixture-only evidence cannot adopt (this closeout's own preregistered rule, enforced structurally by the disposition schema's validator). Production changes to any default/champion pointer occur only through the normal model-lineage process, never from this report directly.

## Evidence chronology

- `exp-sr-1` (SLM-467, SLM-476): `oracle_localization`
- `exp-sr-2` (SLM-477, SLM-482): `prompt_factor_predictor`
- `exp-sr-3` (SLM-478, SLM-483): `evaluator_calibration_protocol`
- `exp-sr-4` (SLM-488): `semantic_repair_witness_cegis`
- `exp-sr-5` (SLM-473, SLM-484): `compute_state_controller`
- `exp-sr-6` (SLM-489): `bounded_search_neural_prior`
- `exp-sr-7` (SLM-485): `packed_semantic_summary`
- `exp-sr-8` (SLM-479): `symbolic_egraph`
- `exp-sr-9` (SLM-480): `symbolic_macro_library`
- `exp-sr-10` (SLM-481): `quality_diversity_corpus`
- `exp-sr-11` (SLM-486): `pysr_srbench_adapter`
- `exp-sr-12` (SLM-474, SLM-487): `second_pack_portability`

## Mechanism disposition table

# Mechanism disposition report

| Mechanism | Disposition | Default state | Evidence class | Rationale |
| --- | --- | --- | --- | --- |
| oracle_localization | retain_diagnostic | n/a | fixture | SIE-002/EXP-SR-1 localized which semantic-plan factors have a material causal ceiling on a VCE-008-cleaned fixture pool (no-op/shuffled/destructive matched controls). Fixture-scale (claim_class=fixture): cannot promote a default. Authorizes only roles/bindings for later learned-predictor work -- archetype/style_layout stays an explicit placeholder, never a claimed authorization. |
| prompt_factor_predictor | reject | off | rejected | SIE-004/EXP-SR-2 evaluated the SIE-003 predictor seam end-to-end against no-predictor/deterministic/frequency/retrieval controls: the learned predictor is dominated by the simpler frequency baseline. Per this closeout's own rule ('when mechanisms are equivalent within uncertainty, prefer the simpler/lower-cost owner'), the predictor is rejected -- stays advisory-learned, off, never entered any hard-prune path. |
| evaluator_calibration_protocol | retain_diagnostic | n/a | fixture | SIE-006/EXP-SR-3 ran VCE-010's calibration protocol through the locked exp-sr-3 identity, but this environment has no real external-judge credentials or human raters (SLM-483's own external-blocked escape hatch): both the external-judge and human-rater arms are labeled mocks. The kappa=0.0 result is honest evidence about the MOCK fixture, not the real evaluator's true calibration -- retained as diagnostic-only pending a genuine external-judge/blinded-human run, never rejected on synthetic data alone. |
| semantic_repair_witness_cegis | retain_diagnostic | off | fixture | RSP-001/EXP-SR-4 shows a clear positive fixture-scale signal for witness-guided CEGIS repair over regenerate/edit-distance/ learned-scorer controls. claim_class=fixture structurally blocks adopt_optional under this schema's own validator (fixture-only evidence cannot adopt) regardless of the source doc's own 'adopt-optional' recommendation language -- retained as diagnostic, default lever stays explicitly off. |
| compute_state_controller | reject | off | rejected | SIE-008/EXP-SR-5 replayed the symbolic value-of-compute controller against hand_threshold/linear_scorer/gold_oracle arms on the SIE-007 substrate: the symbolic rule loses to both simpler controls. Rejected; capacity/compute claims stay advisory-learned and size-matched per goal-invariant VI. |
| bounded_search_neural_prior | retain_diagnostic | off | fixture | RSP-002/EXP-SR-6 shows a clear positive fixture-scale signal for neural-prior search ordering over the existing decode baseline and a model-free enumerative control. Fixture-only evidence blocks adoption under this schema; retained as diagnostic, default stays off pending real reachability-matched, multi-seed evidence. |
| packed_semantic_summary | retain_diagnostic | off | fixture | RSP-003/EXP-SR-7 measured exact domain parity against the live oracle on this fixture, with a real cold-path cost win. The source evidence doc's own recommendation label was 'inconclusive_fixture' (cost/promotion posture, not measurement doubt) -- the underlying result is an unambiguous fixture-scale positive, so retain_diagnostic reflects it more accurately than inconclusive; excluded from production default/checkpoint regardless, per fixture-only evidence. |
| symbolic_egraph | retain_diagnostic | off | fixture | RSP-004/EXP-SR-8: e-graph saturation/extraction is fully certified but produces no frontier improvement over plain canonicalize_expr -- mathematically expected, since SRP-005's current REWRITE_RULES are each complexity-non-increasing and confluent, so both mechanisms provably converge to the same fixed point. Per SLM-479's own acceptance criterion, stays default-off regardless of fixture win; no production canonicalization path imports this module. |
| symbolic_macro_library | retain_diagnostic | off | fixture | RSP-005/EXP-SR-9: the MDL net-gain objective clears the no-macros baseline but shows no measured advantage over the simpler frequency-only control on this small fixture vocabulary -- an honest, unresolved comparison, not a clean win for the MDL-specific claim. All 16 admitted macros passed independent expansion-equivalence re-verification; retained diagnostic, evaluation-only, imported by no production canonicalization path. |
| quality_diversity_corpus | reject | off | rejected | RSP-006/EXP-SR-10: the quality-diversity generation strategy does not beat matched-compute random rejection sampling -- an exact tie, not a directional loss, but the falsifier ('doesn't exceed matched-compute rejection sampling') fires on a tie by its own preregistered wording. Rejected; the simpler random_matched control is preferred per this closeout's own equivalence rule. |
| pysr_srbench_adapter | blocked | n/a | scratch | RSP-007/EXP-SR-11 was never executed: no locked exp-sr-11 campaign or docs/design evidence artifact exists in this repository as of this closeout's evidence cutoff. SRP-011's isolated adapter (symbolic_expr_pysr_adapter.py) is tested in isolation only (mocked/golden output; never imports pysr/Julia), which is not the same as a matched external-benchmark comparison. Disposed blocked rather than inferred from isolated unit coverage or silently omitted; a follow-up issue to actually run (or formally external-block) EXP-SR-11 is filed alongside this report. |
| second_pack_portability | reject | n/a | rejected | RSP-008/EXP-SR-12: zero-fork cross-pack certification is falsified -- the fork inventory IS the evidence the falsifier fired correctly, satisfying this closeout's own rule that cross-pack claims require actual replication, not an inference from single-pack success. Rejected as a cross-pack certification claim; OpenUI and symbolic-regression pack claims remain scoped separately. |

## Cross-pack summary

- OpenUI-pack mechanisms (exp-sr-1, exp-sr-2, exp-sr-4, exp-sr-6, exp-sr-7) and symbolic-regression-pack mechanisms (exp-sr-8, exp-sr-9) are each scoped and disposed within their own pack; none of their results are extrapolated across packs.
- The one genuine cross-pack claim (exp-sr-12, second-pack portability) was rejected on real replication evidence (3 required shared-seam forks) rather than assumed from single-pack success -- satisfying this closeout's own 'cross-pack claims require actual replication' rule.

## Rejected mechanisms

- `prompt_factor_predictor`: SIE-004/EXP-SR-2 evaluated the SIE-003 predictor seam end-to-end against no-predictor/deterministic/frequency/retrieval controls: the learned predictor is dominated by the simpler frequency baseline. Per this closeout's own rule ('when mechanisms are equivalent within uncertainty, prefer the simpler/lower-cost owner'), the predictor is rejected -- stays advisory-learned, off, never entered any hard-prune path.
- `compute_state_controller`: SIE-008/EXP-SR-5 replayed the symbolic value-of-compute controller against hand_threshold/linear_scorer/gold_oracle arms on the SIE-007 substrate: the symbolic rule loses to both simpler controls. Rejected; capacity/compute claims stay advisory-learned and size-matched per goal-invariant VI.
- `quality_diversity_corpus`: RSP-006/EXP-SR-10: the quality-diversity generation strategy does not beat matched-compute random rejection sampling -- an exact tie, not a directional loss, but the falsifier ('doesn't exceed matched-compute rejection sampling') fires on a tie by its own preregistered wording. Rejected; the simpler random_matched control is preferred per this closeout's own equivalence rule.
- `second_pack_portability`: RSP-008/EXP-SR-12: zero-fork cross-pack certification is falsified -- the fork inventory IS the evidence the falsifier fired correctly, satisfying this closeout's own rule that cross-pack claims require actual replication, not an inference from single-pack success. Rejected as a cross-pack certification claim; OpenUI and symbolic-regression pack claims remain scoped separately.

## Blocked mechanisms

- `pysr_srbench_adapter`: RSP-007/EXP-SR-11 was never executed: no locked exp-sr-11 campaign or docs/design evidence artifact exists in this repository as of this closeout's evidence cutoff. SRP-011's isolated adapter (symbolic_expr_pysr_adapter.py) is tested in isolation only (mocked/golden output; never imports pysr/Julia), which is not the same as a matched external-benchmark comparison. Disposed blocked rather than inferred from isolated unit coverage or silently omitted; a follow-up issue to actually run (or formally external-block) EXP-SR-11 is filed alongside this report.

## Reproducibility

```
python -m scripts.publish_exp_sr_disposition --check
```

## Limitations

- Every measured result here is fixture-scale: small in-process corpora, mocked external judges, synthetic human raters, seeds 0-2 at most. None of these dispositions is a statistically-powered, production-scale claim.
- exp-sr-3 (evaluator calibration) and exp-sr-11 (PySR/SRBench benchmark) both remain genuinely open: neither has been measured against real external evidence (real judge/human raters; a real external tool run). Follow-up issues are filed for both rather than treating the mock/blocked results as final answers.
- This report does not itself change any production default, champion pointer, or checkpoint -- every disposed mechanism's `default_state` stays `off`/`n/a`, matching each source experiment's own explicit lever state.

