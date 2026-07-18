# VSS2-04: Represent stripped user regions semantically and splice them hygienically after solving

**Issue:** SLM-68
**Status:** wiring / fixture evidence. No train, eval, benchmark, model, checkpoint, or ship claim.

## What was added

A semantic opaque-region model in `src/slm_training/dsl/opaque_regions.py`:

- `OpaqueRegionKind` enum — `CONTENT_VALUE`, `IDENTIFIER`, `EXPRESSION`, `STATEMENT_BLOCK`, `AST_SUBTREE`, `COMMENT`.
- Immutable, JSON-safe dataclasses:
  - `OpaqueRegionSummary` — conservative contract (bindings, effects, pre/post-conditions).
  - `OpaqueRegion` — region identity, kind, AST path, placeholder, source digest, summary, and `required` flag.
  - `OpaqueRegionBinding` — concrete scalar/AST/source fragment supplied for a region.
  - `OpaqueRealizationResult` — splice outcome with status, source, AST, verifier report, source map, region digests, and errors.
- `realize_opaque_regions(program_or_spec, bindings, *, pack)`:
  - Accepts a `ProgramSpec` or raw source string.
  - Validates coverage, duplicates, unknown regions, and kind-appropriate binding shapes.
  - Splices `CONTENT_VALUE` placeholders deterministically; delegates richer kinds to pack hooks.
  - Canonicalizes the assembled source and re-runs the pack oracle.
  - Returns honest statuses: `"solved"` only when the oracle reports no failing gate, `"rejected"` when it does, `"unknown"` when no oracle is present, `"error"` on failure.

`ProgramSpec` persistence in `src/slm_training/data/progspec/schema.py`:

- Added `opaque_regions: tuple[OpaqueRegion, ...]`.
- `to_dict`/`from_dict` preserve the field and tolerate historical records that lack it.

Pack integration in `src/slm_training/dsl/pack.py`:

- Added optional slots to `DslPack`:
  - `opaque_region_extractor`
  - `fragment_parser`
  - `region_splicer`
- Registered `_openui_opaque_region_extractor` for the `openui` pack, which classifies `CONTENT_PROPS` placeholders (e.g. `:hero.title`) as `CONTENT_VALUE` regions.

Regression tests in `tests/test_dsl/test_opaque_regions.py`:

- ProgramSpec round-trip and historical compatibility.
- OpenUI region extraction.
- Successful splice/verify, missing required region, unknown binding, duplicate binding, unsupported kind fail-closed, and digest non-leakage.

## Verified

- `ruff check src/slm_training/data/progspec/schema.py src/slm_training/dsl/pack.py src/slm_training/dsl/opaque_regions.py tests/test_dsl/test_opaque_regions.py` passes.
- `python -m compileall` passes.
- `pytest tests/test_dsl/test_opaque_regions.py tests/test_data/test_progspec.py tests/test_dsl/test_pack.py -q` → 45 passed.
- `python -m scripts.repo_policy` ok.
- `git diff --check` clean.
- `.githooks/check-changed` → passes except for the pre-existing unrelated failure in `tests/test_data/test_verify.py::test_preview_runtime_and_behavior_seeded_failures` (missing `src/src/.../preview.js` build artifact).

## Design boundaries preserved

- The solver still operates on stable region IDs and conservative summaries; concrete content is spliced only after solving.
- `CONTENT_VALUE` splicing is pack-aware and quote-escapes conservatively for OpenUI string props.
- Unknown regions, duplicate bindings, and missing required regions fail closed.
- Region digests bind to the supplied binding value, not to raw source, so the realized source cannot be reconstructed from the digest alone.

## Caveats

- Only `CONTENT_VALUE` splicing is implemented end-to-end; the other `OpaqueRegionKind`s require pack-specific `fragment_parser` / `region_splicer` hooks that are not yet provided.
- No model, checkpoint, or ship gate is claimed; this is fixture wiring for the VSS2 solver stack.
