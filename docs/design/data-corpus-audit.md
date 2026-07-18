# Data corpus audit

Generated 2026-07-18T19:04:36.221667+00:00 by `scripts/audit_data_corpora.py` (semantic engine: lexical-tfidf).

## Snapshots

| snapshot | records | hard-fail rate | below-0.55 rate | mean score |
| --- | ---: | ---: | ---: | ---: |
| remediated_roots | 108 | 0.6019 | 0.3241 | 0.7046 |
| remediated_unique | 198 | 0.1667 | 0.0 | 0.9308 |
| remediated | 585 | 0.1385 | 0.0 | 0.9428 |
| remediated_roots_judged | 255 | 0.0314 | 0.0235 | 0.9631 |
| remediated_cap3 | 324 | 0.0278 | 0.0 | 0.9816 |
| e177_semantic_judge_v2 | 496 | 0.0121 | 0.0121 | 0.9487 |
| e214_schema_role_judge_v3 | 447 | 0.0 | 0.0 | 0.9648 |
| e218_schema_normalized_judge_v5 | 480 | 0.0 | 0.0 | 0.9654 |
| e230_diverse_judged_roots_v2 | 126 | 0.0 | 0.0 | 0.9937 |
| e283_signature_support_synth_v1 | 2 | 0.0 | 0.0 | 0.925 |
| e297_semantic_contract_judge_v1 | 480 | 0.0 | 0.0 | 0.9654 |
| scope_graded_v1 | 1223 | 0.0 | 0.0 | 0.9862 |

## Cross-snapshot exact overlap (top rows)

| a | b | shared pairs | shared structures | containment |
| --- | --- | ---: | ---: | ---: |
| e177_semantic_judge_v2 | e218_schema_normalized_judge_v5 | 474 | 88 | 0.9875 |
| e177_semantic_judge_v2 | e214_schema_role_judge_v3 | 447 | 83 | 1.0 |
| e214_schema_role_judge_v3 | e218_schema_normalized_judge_v5 | 447 | 83 | 1.0 |
| e218_schema_normalized_judge_v5 | e297_semantic_contract_judge_v1 | 390 | 89 | 0.8125 |
| e177_semantic_judge_v2 | e297_semantic_contract_judge_v1 | 385 | 88 | 0.8021 |
| e214_schema_role_judge_v3 | e297_semantic_contract_judge_v1 | 363 | 83 | 0.8121 |
| remediated | remediated_cap3 | 242 | 92 | 0.7469 |
| remediated | remediated_unique | 198 | 95 | 1.0 |
| e177_semantic_judge_v2 | remediated_roots_judged | 191 | 32 | 0.749 |
| e218_schema_normalized_judge_v5 | remediated_roots_judged | 179 | 30 | 0.702 |
| remediated_cap3 | remediated_unique | 170 | 92 | 0.8586 |
| e214_schema_role_judge_v3 | remediated_roots_judged | 167 | 28 | 0.6549 |
| e297_semantic_contract_judge_v1 | remediated_roots_judged | 149 | 30 | 0.5843 |
| remediated_roots_judged | scope_graded_v1 | 141 | 88 | 0.5529 |
| e230_diverse_judged_roots_v2 | scope_graded_v1 | 126 | 108 | 1.0 |

## Redundancy

- MinHash near-dup clusters (≥2 members): 828 (691 span multiple snapshots).
- Semantic dedup on the union would drop 3108 of 4724 records (2384 cross-snapshot).

Full detail: `docs/design/data-corpus-audit.json`.

## Queued experiment: strict-profile corpus vs permissive baseline

Evidence above (66% union redundancy; `remediated_roots` 60% hard-fail) plus
the strict build measurement (`--version qa-strict`: 1073 candidates → 451
admitted, 559 redundancy drops, 6 eval n-gram overlaps rejected) motivates a
matrix rerun on curated data.

- **Hypothesis**: a strict-profile corpus matches or beats the permissive
  build of the same sources on `meaningful_program_rate`,
  `component_type_recall`, and `placeholder_fidelity` at equal token budget,
  despite ~58% fewer records (quality-over-quantity, phi/SmolLM2).
- **Design**: build the same `--source all` corpus twice (`--profile
  permissive` vs default strict), same seed/recipe/steps via
  `run_quality_matrix` rows; evaluate on the shared frozen suites with
  `--ship-gates`; compare per-suite metrics and `docs/design/`-document both.
- **Success**: strict ≥ permissive on the gated metrics for smoke + held_out
  with `n` reported; **falsification**: strict loses >2 points on any gated
  metric at equal budget.
- **Knobs**: `profile`, `semantic_dedup_threshold`, `ngram_overlap_threshold`,
  `max_records_per_parent`, `--dedup-against`.
- Per-build feedback (`synthesis_feedback.json`) decides the follow-up
  synthesis fixes (`synthesis-feedback` skill).
