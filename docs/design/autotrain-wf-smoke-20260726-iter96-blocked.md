# autotrain_wf_smoke_20260726_iter96 — blocked

**Honesty:** `fixture_or_scratch` attempt. **Not ship. Run did not complete.**

## What happened

Continuing the `docs/design/autotrain-loop-ledger-20260725.md` smoke loop
(iter1–95, PRs #961–#981) from a fresh container, the fixture train build
succeeded but the SFT step failed before any training step ran:

```
$ python -m scripts.build_train_data --source fixture --version wf_smoke_v4 \
    --synthesizer quality --output-root outputs/data/train --no-publish \
    --no-register-lineage --sanitize-mode enforce
wrote outputs/data/train/wf_smoke_v4   # 103 admitted, quality_report.json clean

$ python -m scripts.train_model --train-dir outputs/data/train/wf_smoke_v4 \
    --model twotower --context-backend scratch --device cpu \
    --steps 8 --fast-train --no-sync-checkpoints \
    --run-id autotrain_wf_smoke_20260726_iter96 --seed 1 --eval-suite smoke
...
  File "src/slm_training/models/twotower.py", line 14089, in from_records
    assert_canonical_template_markers(record)
  File "src/slm_training/data/contract.py", line 184, in assert_canonical_template_markers
    assert_canonical_template_marker_inventory([*declared, *observed])
ValueError: persisted template markers must use opaque :slot_<ordinal> identities
```

## Root cause

`TwoTowerModel.from_records` (added in `995813d`, part of the decode-invariant
work) unconditionally requires every record's declared `placeholders` to be
contiguous opaque `:slot_<n>` markers (`src/slm_training/data/contract.py`
`assert_canonical_template_markers`). `--source fixture` only reaches the
`fixture` / `fixture+aug` / `fixture+template` / `fixture+frontier_described`
synthesizer families, none of which canonicalize placeholders — they persist
the seed's natural names verbatim (`:auth.title`, `:auth.email.placeholder`,
...). Repo-wide `grep` for a marker-canonicalization routine
(`canonicaliz*`, `to_slot_form`, `opaque_placeholders`) in
`src/slm_training/harnesses/train_data/` returns nothing, and
`--sanitize-mode enforce` does not touch `placeholders`.

This is not specific to today's build: the same named-placeholder shape is
what most of the repo's committed train fixtures already use (checked
`records.jsonl`/`all.jsonl` placeholders across every
`src/slm_training/resources/data/train/*` directory) —
`dq_strict_fixture_r4_20260718`, `openui_verified_v1`, `remediated*`,
`slm230_symbol_only_v1`, the `e177`/`e214`/`e230`/`e283`/`e500`–`e530` family,
and `scope_graded_v1` all declare dotted names, not `:slot_N`. Only the
newer `e826`/`e845`/`e851`/`e872`/`e897`/`e899`/`e937`/`e1278`–`e1291` slot-
contract datasets already emit canonical `:slot_N`. So `from_records` on any
of the former today would hit the same assertion — this looks like a
standing gap between the decode-invariant contract and the fixture/human-
curated synthesis path, not a one-off fluke in this loop.

## What this session did NOT do

- Did not weaken, skip, or monkeypatch `assert_canonical_template_markers`
  (non-negotiable per `AGENTS.md` decode invariants).
- Did not fabricate a `last_loss`/`wall_s` row to keep the ledger streak
  going — no training step executed, so there is nothing honest to report
  under `wf_smoke_v4`.
- Did not touch `src/slm_training/resources/data/train/` — the one-off
  `--publish`-default side effect from the first attempt (auto-publishing a
  `wf_smoke_v3` fixture snapshot, inconsistent with 95 prior docs-only PRs)
  was reverted before commit.

## Ledger entry

Recorded as a failed row in `docs/design/autotrain-loop-ledger-20260725.md`
(`ok=False`, `stopped_on=canonical_template_marker_gate`) rather than
continuing the loop past it.

## Suggested next step

Either (a) add a placeholder-canonicalization pass to the fixture/human-
curated/layout-augment synthesizer families so `--source fixture` output
satisfies `assert_canonical_template_markers` (harness change — use
`improve-openui-harnesses`), or (b) point the smoke recipe at one of the
already-canonical `:slot_N` fixtures (e.g. `e937_role_safe_all_targets_v2`)
instead of a fresh `--source fixture` build. This needs an explicit choice
before the loop resumes.
