# Annotations phase

Human feedback → typed training inputs. Owner:
`src/slm_training/harnesses/annotations/`; the collection UI and API live under
`src/slm_training/web/`.

## Prerequisites

- An annotation store populated via the Mission Control dashboard / playground
  (`python -m scripts.serve_playground`, see README "Annotate playground").

## Commands

```bash
# Export annotations for downstream consumers
slm annotations export --pairs <pairs-out.jsonl> \
  [--human-train <records-out.jsonl>] [--feedback <summary-out.json>]
```

(`slm annotations export` ≡ `python -m scripts.export_annotations`.)
Downstream: feed `--pairs` into the preference phase
(`slm preference train-events` / pair building).

## Key flags

`--pairs`, `--human-train`, `--feedback`.

## Outputs

Derived preference artifacts belong to the consuming run — raw stores stay
under their configured output root; no new root folders.

## Gates & invariants

- Stable annotation IDs, atomic appends, attempt provenance preserved.
- Validation/append safety is the harness's job — never hand-edit stores.

## Close out

- Shared duties: [contracts.md](contracts.md).
- Checks: `pytest -q tests/test_web/test_annotation_store.py
  tests/test_web/test_annotations.py tests/test_web/test_bad_outputs.py`.
- Changing the harness itself → `improve-openui-harnesses`.
