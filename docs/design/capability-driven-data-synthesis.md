# Capability-driven synthetic data

Status: **infrastructure landed; no model or ship-quality claim**.
Companion JSON: [capability-driven-data-synthesis.json](capability-driven-data-synthesis.json).

This document describes the typed corpus-generation contract that makes
training data a first-class research object. It does **not** certify CAP1,
does **not** promote a checkpoint, and does **not** change production decode
or ship gates.

## Why this exists

The continuous path was evolving model, loss, and decoder levers against a
fixed fixture corpus while prompt wrappers (`OpenUI layout request`,
`Generate the <family> OpenUI program`) taught product/DSL triggers instead of
user intent. Row count is not the scale variable. Unique canonical roots,
semantic frames, prompt surfaces, and shortcut-resistant pair checks are.

## Contract

- `synthesis_plan/v1` files still load with exact-key validation. Historical
  v1 payloads keep their SHA if their bytes are untouched. The checked-in
  CAP0 fixture SHA **did change** in this change: only component-version
  aliases were refreshed so `require_executable()` matches the live
  `harness.train_data` / surface / ship-gate registry. That is an alias
  refresh, not a silent v1 semantic change.
- `synthesis_plan/v2` adds a required `corpus_generation` policy
  (`corpus_generation/v1`) covering unique-root targets, generator bounds,
  prompt surface, cue policy, derivatives, trusted anchors, pair-quality
  thresholds, and nested-ladder prefix rules.
- Shared loader: `load_synthesis_plan()`. Migration is explicit via
  `migrate_plan_v1_to_v2()`.
- Capability and prompt surface stay separate axes:
  - CAP0 → `grammar_schema`
  - CAP1 → `simplified_nl` (no product/DSL vocabulary required)
  - CAP2 → `transformation_nl`
  - `complex_nl` remains unavailable until the existing capability gate
    authorizes it.

## Owners

| Concern | Owner |
| --- | --- |
| Plan + policy | `src/slm_training/harnesses/synthesis_plan.py` |
| Unique-root pools | `src/slm_training/data/progspec/generate.py` (`generate_program_pool`) |
| Simplified-NL prompts | `src/slm_training/harnesses/train_data/semantic_prompts.py` |
| One-fact derivatives | `src/slm_training/harnesses/train_data/semantic_counterfactuals.py` |
| Pair/shortcut audit | `src/slm_training/harnesses/train_data/pair_quality.py` |
| Build/report/feedback | `pipeline.py`, `report.py`, `feedback.py` |
| Split/anchors | `split_policy.py`, `catalog.py` |
| Data-only knobs | `autoresearch/schemas.py` `DataGenerationKnobs` |
| Learnability probe | `evals/learnability_diagnostics.py` |

## CLI

```bash
python -m scripts.build_train_data \
  --source programspec \
  --synthesis-plan src/slm_training/resources/synthesis_plans/corpus/cap0_tiny_v2.json \
  --version cap0-tiny-corpus \
  --profile strict \
  --output-root outputs/data/train
```

`--programspec-count` remains the historical count path. A v2 plan overrides
count/seed/repairs from `corpus_generation`. CI fixtures stay tiny. An 8,192
root request is a **legal policy value**. This change did **not** generate,
train on, or commit an 8,192-root corpus.

## Artifacts

A plan-driven build still writes the canonical set: `records.jsonl`,
`manifest.json`, `stats.json`, `quality_report.json` (now with `unique_roots`
and `pair_quality` when a v2 plan is present), `rejected.jsonl`,
`synthesis_feedback.json` (findings + executable `data_only` experiment
candidates). Blocking findings cannot close with a prose acknowledgement;
`action_receipt()` requires a new manifest hash or an explicit diagnostic
waiver that cannot authorize promotion.

## Honest limits

- **Claim class:** `fixture_wiring`. No capability certificate. No ship
  claim. No model-quality claim.
- Offline `offline_fixture_renderer` prompts are not production natural
  language and cannot authorize a CAP1 simplified-NL certificate.
- A cue-only classifier on tiny support returns `insufficient_support`.
- Synthetic-only corpora can be diagnostic; they are not
  promotion-authoritative without trusted anchors.
- Masked-reconstruction vs free-running / LTR mismatch remains an
  independent hypothesis. The learnability probe can *label* that
  disposition; this change does not close it.
- `wf_smoke_v2` is unchanged and remains fixture-only.
- Plan-driven `quality_report.json` / `synthesis_feedback.json` stamp
  `claim_class=fixture_wiring` and `capability_certificate=false`. A
  missing pair-quality auditor is a fail, not a pass.

## Follow-ups (this change)

Wiring only. `claim_class` remains `fixture_wiring`; `capability_certificate`
is false; no model, CAP, or ship claim.

- Coverage minima are enforced at pool generation. CI still uses 4 roots;
  8192 is a legal policy value, never a committed corpus.
- Pair-quality auditor reasons map onto `FINDING_CODES`. There is no silent
  relabel to leakage.
- Learnability diagnostics run on the default single-suite path and use the
  first suite `prompt` fields, not metric names.
- Data-only arms are allowlisted; `compile_commands` lowers the full
  `DataGenerationKnobs` set, including `--unique-root-target` overlays that
  a v2 plan previously ignored.
- A v2 root admits every rendered prompt candidate; repairs and
  counterfactuals still fire once per root.
- Unique-root overlays raise `max_attempts` so a larger target cannot
  silently exhaust the plan's tiny attempt budget. `rebuild_data` now
  carries compileable `data_generation` knobs.
- Provider paraphrase stays offline in ordinary tests.

## Papers used as design guidance

Self-Instruct, TinyStories, Textbooks Are All You Need, AlpaGasus,
Superfiltering, LESS, DoReMi, annotation artifacts (NLI), SWAG, instruction
phrasing robustness, structured counterfactual augmentation, and synthetic-data
collapse. None of them prove this repository’s model will succeed; they
justify making the hypothesis measurable.
