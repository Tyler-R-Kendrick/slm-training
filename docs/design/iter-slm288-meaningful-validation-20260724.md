# SLM-288 / LAR0-04: Meaningful-program metric validation

Matrix set: `slm288-meaningful-validation` · Version: `slm288-v1` · Status: **complete** · Claim class: diagnostic

## What was built

- **Typed ordered reason codes for v1** — `meaningful_program_v1_report` in
  `eval_runner.py` (schema `meaningful_program_v1_reasons/v1`). Every clause
  is evaluated without short-circuit; `verdict` and the first failure's
  legacy reason string are byte-identical to `meaningful_program_v1`
  (verified on all 33 retained rows). Codes: `parse_failed`,
  `free_form_output_string`, `empty_root_stack`, `empty_card`,
  `empty_children`, `no_content_components`, `no_placeholders`,
  `low_component_recall`, plus `component_recall_unobservable` — the
  UNKNOWN/unobservable case (no gold, or gold with only Stack components),
  which is reported explicitly and **never counted as a semantic failure**.
- **Reason-code histograms** over the committed frontier replay bundle
  (33 rows, 11 checkpoint families, smoke suite).
- **Frozen blinded annotation packet** —
  `src/slm_training/resources/evals/slm288_annotation_packet_v1.json`
  (sha256 `0980b71d…c43d6161a`) + rubric (`ba09ece7…64aba4ce`). Blinded
  rater export carries only `{packet_id, suite, prompt, prediction}`.
- **Two independent rater passes** (agent raters, separate blind contexts,
  different item order) — labels + agreement in
  `docs/design/slm288-annotation-labels-20260724.json`.
- **Harness** `scripts/run_slm288_meaningful_validation.py`:
  `histograms` / `freeze-packet` / `export-blind` / `analyze`.

## Results (diagnostic)

| Measure | Value |
| --- | --- |
| Packet n (target ~100) | **33** — shortfall declared; only 33 prompt-available retained rows exist |
| v1 verdicts on retained rows | 7 pass / 26 fail |
| Top fail reasons | `low_component_recall` 26, `no_content_components` 13, `empty_root_stack` 9, `no_placeholders` 9 |
| Rater agreement (Cohen κ) | 1.0 (n=33, adjudication rate 0.0) |
| Metric vs labels | precision 0.43, recall 0.75 (tp 3, fp 4, fn 1, tn 25) |
| Recall-floor sweep (predeclared 0.3/0.5/0.7) | 0.3/0.5 identical (P 0.43 / R 0.75); 0.7 → R 0.0, 4 FN |

## Disposition: `demote_to_diagnostic_for_lar_decisions`

- The primary validity threshold requires **human** agreement κ ≥ 0.6. The
  two raters are independent agent passes from the same model family —
  wiring/diagnostic evidence, not human validation — so the threshold is
  **not certifiably met** regardless of the observed κ = 1.0.
- The packet is underpowered by the issue's own bar (33 vs ~100).
- Diagnostic signal is unfavorable anyway: precision 0.43 with 4
  metric-pass false positives, and recall collapsing to 0.0 at the 0.7
  floor.

Consequences:

- `meaningful_program_rate` (v1) **remains the ship-gate primary** — this
  issue makes no retroactive metric, threshold, or gate change.
- For LAR1+ program decisions, meaningful-program rate is diagnostic-only
  until a powered, human-labeled packet certifies κ ≥ 0.6 with documented
  precision/recall.
- `binding_aware_meaningful_v2` stays `candidate_pending_calibration`.

## Honest caveats

- Agent raters share a model family; κ = 1.0 likely overstates true
  independent agreement. No human labels exist in-repo.
- The packet covers the smoke suite only (that is all the retained replay
  bundle contains); docs/design iter JSONs carry predictions without prompts
  and cannot be labeled honestly.
- `test_v7_speculative`-style long suites were not re-run here; verification
  used the targeted test files below.

## Verification

- `tests/test_scripts/test_run_slm288_meaningful_validation.py` — 8 tests:
  v1/report verdict+reason parity over all 33 rows, typed ordered codes,
  UNKNOWN-not-failure, empty-children detail, histogram shape, packet
  determinism + blind export purity, κ/PR/sweep math + determinism, label
  schema fail-closed.
