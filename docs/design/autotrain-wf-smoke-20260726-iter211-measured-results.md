# autotrain_wf_smoke_20260726_iter211

**Honesty:** fixture_or_scratch. **Not ship.**

train_version=wf_smoke_v3_iter211 last_loss=27.732410430908203 stopped_on=steps wall=3.1647911359999625 max_wall=2.5833333333333335 n=3

## Recipe

| Field | Value |
| --- | --- |
| run_id | `autotrain_wf_smoke_20260726_iter211` |
| model | twotower / choice / scratch / cpu |
| steps | 8 (`--fast-train`, `--no-sync-checkpoints`) |
| seed | 1 |
| train records | 103 (filtered from `e937_role_safe_all_targets_v2`: root-bound programs only) |
| eval suite | smoke, n=3 (from `e938_role_safe_all_targets_v2`) |
| ship_gates | false (`--eval-limit 3 --suites smoke --run-class fixture_demo`) |
| eval outcome | 0/1 AgentV cases passed (criteria fail expected at 8-step scratch scale) |

## Provenance caveat (blocks reusing the historical `--source fixture` recipe)

The `autotrain_wf_smoke_2026072{5,6}` recipe used through iter210 built train/eval
data with `slm data build-train --source fixture ...`. That source still resolves
to the legacy `openui_verified_v1`-style seed corpus, whose `placeholders` are
named (e.g. `:auth.title`) rather than opaque `:slot_<ordinal>` identities.
`TwoTowerModel.from_records` now hard-fails via
`assert_canonical_template_markers` (`src/slm_training/data/contract.py:184`,
part of the E810–E821 canonical-slot harness) on every one of those records —
CI's fixture-build check (`.github/workflows/ci.yml` "Verify documented
disjoint data build") never exercises `train_model`, so this incompatibility
between the legacy fixture seed corpus and the canonical-marker invariant is
invisible to CI.

This iteration instead trained on a filtered, already-canonical 103-record
subset of the committed `e937_role_safe_all_targets_v2` corpus (root-bound
programs only; that corpus also carries 173 non-root lexical-scope edit
fragments not meant to be trained as standalone full programs). This keeps
the honest `fixture_or_scratch` label and the historical step/seed/model
recipe, but the train-data *source* differs from iter2–iter210 and should be
treated as a new sub-lineage (`wf_smoke_v3_iter211`), not a direct
continuation of `wf_smoke_v1`/`wf_smoke_v2`.

Also required and previously undocumented for this recipe to run at all in a
fresh checkout: `npm ci` in both the repo root (AgentV SDK, used by
`write_loss_suite_report`) and `src/apps/openui_bridge` (DSL validate/lex
bridge used by `build_train_data`), plus a Python 3.12 venv
(`pyproject.toml` pins `requires-python = ">=3.12,<3.13"`).
