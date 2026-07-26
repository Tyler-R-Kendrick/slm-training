# autotrain_wf_smoke_20260726_iter341

**Honesty:** fixture_or_scratch. **Not ship. FAILED — loop blocked.**

## Outcome

`ok=False`. The smoke loop's `slm sft train --model twotower` step fails
outright on a freshly built `wf_smoke_v2` fixture corpus. This is not a flaky
or partial result: **all 103 records** in the fixture-sourced train build are
rejected before a single training step runs.

## Repro

```bash
python -m scripts.slm data build-train --source fixture --version wf_smoke_v2 \
  --synthesizer quality
python -m scripts.slm sft train --train-dir outputs/data/train/wf_smoke_v2 \
  --model twotower --context-backend scratch --steps 8 \
  --run-id autotrain_wf_smoke_20260726_iter341 --fast-train \
  --no-sync-checkpoints --device cpu
```

```
File "src/slm_training/models/twotower.py", line 14216, in from_records
    assert_canonical_template_markers(record)
File "src/slm_training/data/contract.py", line 195, in assert_canonical_template_marker_inventory
    raise ValueError(
ValueError: persisted template markers must use opaque :slot_<ordinal> identities
```

## Root cause

`src/slm_training/resources/train_seeds.jsonl` (the seed corpus behind
`--source fixture`) uses human-readable, dotted placeholder names —
e.g. `:auth.title`, `:auth.email.placeholder`, `:auth.password.placeholder`,
`:auth.continue`, `:hero.title` — instead of opaque `:slot_<ordinal>`
markers. That was harmless until PR #952 (`7cddae64`, 2026-07-25 18:58:28,
"SLM-315 (AP-023): add a lightweight discrete abstract-plan head to
TwoTower") added `assert_canonical_template_markers(record)` to
`TwoTowerModel.from_records` (`src/slm_training/models/twotower.py:14216`).
Every record built from these seeds now fails that assertion —
`assert_canonical_template_marker_inventory` at
`src/slm_training/data/contract.py:184-195` rejects the named markers
unconditionally.

Verified directly against the loaded records:

```
n records: 103
total bad: 103 of 103
```

Confirmed with both the default `strict` profile and `--profile permissive`
— same 100% failure either way, so this is not a decontamination/profile
setting; it's the seed corpus itself.

## Scope

- `tests/test_harnesses/model_build` (`-k "twotower or from_records or
  canonical"`, 61 tests) all pass — those fixtures are hand-authored inline
  with canonical markers already, so they never exercised this path.
- No test in `tests/test_harnesses/train_data/*` or `tests/test_integration/`
  feeds a `source="fixture"` build into `TwoTowerModel.from_records`, so
  PR #952 shipped without CI catching this integration gap.
- Every other committed `src/slm_training/resources/data/train/*` corpus
  (`e530_*`, `e527_*`, `remediated_*`, `openui_verified_v1`, …) already
  contains canonical `:slot_N` markers — only the raw seed file
  (`train_seeds.jsonl`) and anything built straight from it are affected.

## Why this blocks the loop

This smoke loop (`docs/design/autotrain-loop-ledger-20260725.md`, 340 prior
iterations) exists specifically to exercise `data build-train --source
fixture` → `sft train --model twotower` end to end. With 100% of fixture
records rejected, no further genuine iteration can run. Per this repo's
honesty rules, the loop does not fabricate a passing iteration or bypass/
weaken `assert_canonical_template_markers` to force a green run.

## Next step (not taken here — scope is a harness fix, not this loop)

Canonicalize `train_seeds.jsonl`'s placeholders to `:slot_<ordinal>` (or add
a canonicalization pass ahead of `TwoTowerModel.from_records` for
fixture-sourced records), then add an integration test that builds
`source=fixture` train data and feeds it through `TwoTowerModel.from_records`
so this gap can't silently reopen. Tracked via `improve-openui-harnesses`
scope, not the autotrain operating loop.
