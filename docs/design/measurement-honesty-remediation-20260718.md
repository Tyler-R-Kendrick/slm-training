# Measurement-honesty remediation — 2026-07-18

**Prompt:** "a lot of metrics stuck at 0 for all experiment runs — suggesting
we're not actually measuring what we're optimizing for, or that experiment
quality is poor." A three-way audit (all 378 committed `docs/design/*.json`
records; the metric-computation code; every consumer) confirmed the
observation and decomposed it into five defect classes. The prose ledger
(MODEL_CARD, README, per-run verdicts) was honest throughout — the defects
lived in measurement semantics and evidence→consumer plumbing. This note
records the audit numbers and the code remediation. Policy deltas:
`adversarial-review.md` § 2026-07-18.

## Audit findings (measured on the committed corpus)

| Finding | Measurement |
| --- | --- |
| Fields exactly 0.0 in 100% of occurrences | 596 distinct names — mostly decode-telemetry counters for features that never fired |
| `contract_precision` exactly 1.0 | 56% of occurrences (empty-set 0/0 branch, not skill) |
| `contract_recall` exactly 1.0 | 52.6% |
| `parse_rate` = 1.0 while `meaningful_program_rate` = 0.0 | 73 suite-blocks (decoder-guaranteed syntax post-E226) |
| Suite `n` distribution | n=3 most common (169 files); n=1 in 55; n=0 in 10; >5 is a rounding error |
| `meaningful_program_rate` (primary gate lever) | 63% of measured suite-blocks exactly 0.0; max ever 0.6667 (= 2/3, an n=3 quantization ceiling); 0.0 on held_out/ood/adversarial in every run |
| Dashboard research scoreboard | rendered 12 of 378 committed records (glob + one accepted shape); E292–E295 boards invisible |
| Promotion gate `request_coverage` | required by lineage + model_cycle, emitted by nothing (renamed long ago); test fixture hand-injected it |
| Dangling artifact references | 330 `outputs/…` paths (gitignored) incl. foreign absolute paths (`/home/codex/…`) |
| Loss vs measured quality | `last_loss` varies 0.016→92.7 across runs while meaningful/agentv stay pinned ≈0 — objective/measurement decoupling |

## Remediation (this change)

1. **eval_runner measurement semantics** — empty-set metrics return null (not
   vacuous 1.0); unmeasured aggregates null (not fabricated 0.0);
   `metric_defined_n` denominators; harness exceptions become
   `match_error_count`/`reward_error_count` instead of silent 0.0 scores;
   `empty_prediction_count`; real `fallback_count` from decode telemetry;
   `decoder_guaranteed` labeling; single-suite AgentV publishes one real case
   (`include_missing_suites=False`) instead of 4 missing-suite pseudo-failures.
2. **Gates** — `certified_fallback` fails closed when unmeasured; per-suite
   evidence floor `min_n` (default 20) with `insufficient_n` failures; Wilson
   `{metric}_ci95` on count-based rates (n=3 → [0, 0.56] makes quantization
   visible); ceiling + length-budget diagnostics recorded in every
   `evaluate_suites` board so all-zero scoreboards are attributable.
3. **Promotion** — `request_coverage` → `contract_recall` in lineage
   HARD_METRICS and model_cycle; `resume_climb` evaluates the full gate policy
   (its promote branch was structurally unreachable); key-contract test pins
   consumer-required metrics to real evaluator output.
4. **Evidence plumbing** — `evals/record_schema.py` normalizes every committed
   dialect (long keys, syntax variant, short keys, `honest_evaluation` nesting,
   single-suite blocks) onto one canonical vocabulary: research scoreboard now
   renders **257 of 378** records with the remaining 121 typed
   (`no_metric_blocks` — design notes/benchmarks, not experiment records) and
   exposed via `unparsed` instead of silently dropped. Corpus test locks this
   in (`tests/test_web/test_committed_corpus_normalizes.py`).
5. **Dashboard honesty** — no unguarded `parse_rate`→meaningful substitution in
   either renderer; legacy-tagged values render `*` with a legend; unmeasured
   renders `—`; parity kept (`.openui` programs + manifest + e2e in both modes).
6. **Writers going forward** — every eval payload stamps `schema_version=1` +
   `run_class` (`fixture_demo | scratch_matrix | ship_eval`; `--run-class`,
   ship gates imply `ship_eval`). History immutable; readers normalize.
7. **Noise + provenance** — `aggregate_stats` omits never-fired counters
   (named in `counters_omitted_zero`; timings always report), removing the
   ~596 always-zero fields from new records; `repo_policy` rejects
   machine-absolute artifact paths in newly added design records.

## Recipe / verification

Code-change remediation (no train run; no checkpoint created or promoted — no
MODEL_CARD delta). Verified by:

```bash
uv run pytest tests/test_harnesses/model_build tests/test_lineage tests/test_web \
  tests/test_evals/test_agentv.py tests/test_autoresearch tests/test_scripts/test_repo_policy.py -q
python scripts/validate_page_dsl.py          # 6 page programs valid
npx playwright test --project=desktop-chrome tests/e2e/dashboard.spec.ts  # 8/8 both renderer modes
python -m scripts.repo_policy                # ok
```

Known pre-existing failures in this environment (also fail at the base commit,
unrelated to this change): DSL tokenizer vocab-size lock (500 > 480, 2 tests),
`test_twotower_quantized_export_loads_in_browser_adapter`, and the
`binding_aware_meaningful_v2` family (10 tests, official-bridge dependent).

## Honest status

No model became better: ship gates still fail everywhere, and that is the
point — they now fail for measured reasons (`insufficient_n`, real zeros,
unmeasured-fails-closed) instead of drowning in vacuous 1.0s, fabricated 0.0s,
and invisible evidence. Fixture-demo vs ship is now machine-labeled
(`run_class`), and promotion is reachable by an honest run that actually
clears the bars.
