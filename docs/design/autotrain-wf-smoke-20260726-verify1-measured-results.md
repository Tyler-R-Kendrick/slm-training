# wf_smoke_verify_check1

**Honesty:** fixture_or_scratch. **Not ship.**

train_version=wf_smoke_verify last_loss=32.610084533691406 stopped_on=steps
wall=3.218829468000081 max_wall=2.5833333333333335 record_count=101

## Why this run exists

This scheduled run was asked to continue
`docs/design/autotrain-loop-ledger-20260725.md`'s fixture-smoke loop past
`iter400`. Before adding more rows, it re-ran the documented recipe end to
end and reproduced, 100%, the exact blocker independently reported (and never
merged) by two prior sessions:

```text
ValueError: persisted template markers must use opaque :slot_<ordinal> identities
```

See the integrity notice at the top of `autotrain-loop-ledger-20260725.md`
for the full account of why `iter251`-`iter400` cannot be trusted. This doc
records the one iteration that PR actually ran and can vouch for, after
landing the fix.

## Root cause + fix

`src/slm_training/resources/train_seeds.jsonl` (the `--source fixture` seed
corpus) uses named, dotted placeholder markers (`:hero.title`, ...).
`TwoTowerModel.from_records` (`src/slm_training/models/twotower.py`)
unconditionally rejects non-canonical markers via
`assert_canonical_template_markers`
(`src/slm_training/data/contract.py`), added by PR #952. `_normalize_record`
(`src/slm_training/harnesses/train_data/pipeline.py`) persisted the seed's
named markers as-is instead of canonicalizing them first, so every fresh
`--source fixture` build was silently building data that its own SFT
consumer would reject.

Fix (reapplying the approach from the orphaned, never-merged PR #1006):
`_normalize_record` now calls `canonicalize_example_template_markers` +
`assert_canonical_template_markers` on all three of its return paths before
persisting. `harness.train_data` bumped v21 -> v22
(`src/slm_training/resources/versions.json`).

Three tests whose fixtures baked in the leaky named-marker behavior were
updated (`tests/test_harnesses/train_data/test_pipeline.py`):
`test_prompt_contracts_expose_component_counts_and_slots`,
`test_semantic_role_contract_uses_only_visible_slots_and_types`,
`test_build_train_data_from_rico_fixtures` (RICO fixture slice widened from
`rico_limit=10` to `rico_limit=80`, since canonical markers correctly
collapse more near-duplicate structural templates — 80 seeds -> 4 survivors,
76 rejected, accounted for exactly).

`tests/test_harnesses/train_data/test_source_families.py::test_pipeline_manifest_source_families`
and cases in `test_staged_materialization.py` referencing
`src/slm_training/resources/synthesis_plans/dsh0_cap0_fixture.json`'s pinned
`pack.corpus_generator: v18` fail on this branch's HEAD independent of this
change (that plan already pins `v18` against an active `v20`/`v21` generator
before this patch) — pre-existing drift, out of scope here.

## Recipe

```bash
slm data build-train --source fixture --version wf_smoke_verify --synthesizer quality
slm sft train --train-dir outputs/data/train/wf_smoke_verify \
  --model twotower --context-backend scratch --steps 8 \
  --run-id wf_smoke_verify_check1 --no-sync-checkpoints --device cpu
```

Environment: `python3.12` venv (`.venv-smoke/`, untracked, gitignored),
`torch==2.5.1+cu124` (CPU execution), Node bridge deps installed via
`cd src/apps/openui_bridge && npm ci` (with `NODE_OPTIONS` cleared —
the ambient `--import tsx` flag isn't accepted by plain `node`/`npm`).

## Result

`outputs/runs/wf_smoke_verify_check1/train_summary.json`:

| field | value |
| --- | --- |
| steps | 8 |
| stopped_on | steps |
| last_loss | 32.610084533691406 |
| record_count | 101 |
| elapsed_wall_seconds | 3.218829468000081 |
| max_wall_minutes | 2.5833333333333335 |
| device | cpu |

No `--ship-gates` scoreboard was requested; this is wiring verification only,
not a ship claim.

## Artifacts (not committed; `outputs/` is gitignored)

- `outputs/data/train/wf_smoke_verify`
- `outputs/runs/wf_smoke_verify_check1/train_summary.json`
- `outputs/runs/wf_smoke_verify_check1/checkpoints/last.pt`
- `outputs/runs/wf_smoke_verify_check1/trace.json`
