# autotrain_wf_smoke_20260726_iter251 — blocked (fresh checkout)

**Honesty:** `fixture_or_scratch`. **Not ship. No training completed this iteration.**

## What happened

Continuing the autotrain smoke loop in a fresh container (no prior `.venv`,
no `outputs/`). Environment bootstrap:

- `uv sync --extra torch --extra dev` + `uv pip install -e . --no-deps` (project
  itself is not auto-installed by `uv sync` in this checkout).
- `cd src/apps/openui_bridge && npm ci` (DSL bridge; needed even for
  `--source fixture` builds, contrary to `references/train-data.md`
  "Prerequisites: None for fixture builds").
- **Container gotcha:** the shell's `NODE_OPTIONS` includes `--import tsx`,
  which newer Node refuses to accept via `NODE_OPTIONS` ("`--import tsx` is
  not allowed in NODE_OPTIONS"). Every command that touches the OpenUI DSL
  bridge (data builds, and therefore `pytest tests/test_harnesses/train_data`)
  must run with `NODE_OPTIONS=""` or the bridge subprocess dies with exit 9
  and every marker/validate call fails. This cost the most debugging time —
  worth fixing at the shell-init level for this container image.

With those unblocked: `slm data build-train --source fixture --version
wf_smoke_v1 ...` and `slm data build-test ...` both **succeeded** (101/103
records survived quality gates; 16 eval records across smoke/held_out/
adversarial/ood).

`slm sft train --steps 8 ...` then **failed immediately** at model construction:

```
ValueError: persisted template markers must use opaque :slot_<ordinal> identities
  at src/slm_training/data/contract.py:196 (assert_canonical_template_marker_inventory)
  via TwoTowerModel.from_records -> assert_canonical_template_markers
```

## Root cause (verified)

`src/slm_training/harnesses/train_data/pipeline.py::_normalize_record` never
calls `canonicalize_example_template_markers` / `assert_canonical_template_markers`
(`src/slm_training/data/contract.py`) before returning a persisted record —
so every record built by `slm data build-train` (fixture source, `quality`
synthesizer) keeps its producer's named markers (e.g. `:auth.title`,
`:actions.primary`) instead of opaque `:slot_N` identities. The sibling
`src/slm_training/harnesses/test_data/pipeline.py` **does** call both
(`canonicalize_example_template_markers(out)` then `assert_canonical_template_markers(out)`
right before returning — see its `_normalize_record`), which is why
`build-test` never trips this and why the gap is invisible to `slm data
build-train`'s own output — it only surfaces once a record reaches
`TwoTowerModel.from_records` at SFT time. Confirmed 103/103 fixture-built
train records fail the assertion; 0/103 fail after RICO/eval-side test data
is inspected the same way (test-data path is clean).

Repro (from a clean checkout, after `NODE_OPTIONS=""` + bridge `npm ci`):

```python
from slm_training.harnesses.model_build.train_loop import load_train_records
from slm_training.data.contract import assert_canonical_template_markers
records = list(load_train_records("outputs/data/train/wf_smoke_v1"))
bad = [r for r in records if _fails(r)]  # 103/103 fail on HEAD (10b93a8)
```

## Why the obvious fix is wrong

Patched `_normalize_record`'s two `ExampleRecord` return sites to canonicalize
+ assert (mirroring `test_data/pipeline.py`) — this **does** fix the SFT
assertion (0/101 bad after rebuild) but **regresses** `tests/test_harnesses/train_data`
from 8 pre-existing failures to a different 8 (net-neutral count, but one
new failure): `test_build_train_data_from_rico_fixtures` drops from ≥5 RICO
records to 1, because canonicalizing markers to `:slot_N` **before** the
pipeline's n-gram/exact-pair decontamination and dedup stages collapses
previously-distinct records (differing only in named marker spellings) into
byte-identical text, which the dedup/decontam stages then correctly reject
as duplicates. Canonicalization has to happen **after** dedup/decontamination
but **before** the final `records.jsonl` write — a real, multi-site change
to `build_train_data`'s pipeline ordering, not a two-line patch. Reverted the
speculative patch rather than land something that trades one bug for
another; this needs deliberate `improve-openui-harnesses` work with the full
pipeline-ordering picture in view.

## Separate, unrelated pre-existing failure

Independent of the above: `tests/test_harnesses/train_data` has 8
pre-existing failures on HEAD (`10b93a8`) unrelated to markers, including a
version-stamp drift —
`ValueError: generator version mismatch for 'pack.corpus_generator': plan='v18', active='v20'`
(`src/slm_training/harnesses/synthesis_plan.py:470`) — i.e. a locked
synthesis-plan JSON pins an older generator version than what's active.
Not investigated further this iteration; flagging so it isn't conflated with
the marker-canonicalization bug above.

## Status

- `ok`: **False** — no SFT/eval ran; this iteration produced no train_summary,
  no loss, no wall time. Not added to the `iter*` success ledger table.
- Blocking bug filed here for `improve-openui-harnesses` to fix properly
  (canonicalize post-dedup, pre-write, in `build_train_data`).
- Artifacts from this iteration: `outputs/data/train/wf_smoke_v1/`,
  `outputs/data/eval/wf_smoke_v1/` (both scratch, not published, not committed).
