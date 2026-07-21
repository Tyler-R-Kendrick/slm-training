# VSS3-04: Typed late-surface-realization slots and deterministic verified baseline

**Issue:** SLM-72

**Status:** wiring / fixture evidence. No autoregressive model, no training run, no
model checkpoint, and no ship claim.

## What was added

A pack-owned surface-realization boundary in `src/slm_training/dsl/surface.py`:

- `SurfaceSlotKind` — `internal_identifier`, `decorative_text`, `comment`,
  `docstring`, `structured_string`, `externally_observable_name`.
- `SurfaceAuthority` — `surface_only`, `semantic`, `opaque_user_value`.
- Immutable, JSON-safe dataclasses:
  - `SurfaceConstraint` — pattern, max bytes, reserved names, uniqueness scope,
    case preservation.
  - `SurfaceSlot` — slot id, kind, authority, AST path, semantic symbol id,
    opaque region id, constraints, current value digest, and required flag.
  - `SurfaceRealizationRequest` — pack id, constraint version, semantic IR
    fingerprint, slots, and context.
  - `SurfaceAssignment` — slot id, realized value, provenance.
  - `SurfaceRealizationResult` — status, source, AST, verifier report,
    assignments, source map, semantic-equivalence evidence, fallback/repair
    counters, diagnostics, and errors.
- `SurfaceRealizer` protocol and `DeterministicSurfaceRealizer`:
  - assigns deterministic canonical binder names (`v0`, `v1`, ...) to
    `INTERNAL_IDENTIFIER` slots;
  - enforces identifier grammar, reserved-word avoidance, uniqueness, and
    deterministic collision repair;
  - skips `OPAQUE_USER_VALUE` slots (caller supplies validated bindings);
  - rejects `STRUCTURED_STRING`, `EXTERNALLY_OBSERVABLE_NAME`, `COMMENT`, and
    `DOCSTRING` slots if presented as freely assignable;
  - has no model or network dependency.
- `realize_surface_and_verify(solved_program, *, pack, realizer, opaque_bindings,
  semantic_ir_fingerprint, prior_status)` — the end-to-end pipeline:
  1. requires a solved semantic-IR fingerprint and prior `solved`/`verified`
     status;
  2. extracts/classifies surface slots via `resolve_surface_slot_extractor(pack)`
     (a pack-declared `surface_slot_extractor` slot when present, otherwise the
     built-in OpenUI extractor keyed by `pack.pack_id`);
  3. obtains assignments from the realizer;
  4. validates exact coverage and rejects unknown/duplicate/missing required
     slots;
  5. applies identifier substitutions (definitions + references, never inside
     placeholders);
  6. applies opaque content through the VSS2-04 `realize_opaque_regions` splice
     path;
  7. canonicalizes and re-runs the pack oracle;
  8. returns an honest status; a failed verifier yields no certified source.

Pack integration (surface-owned, no `pack.py` change):

The canonical `DslPack` on `main` does not declare a `surface_slot_extractor`
slot, so `surface.py` owns extractor resolution and stays a self-contained
increment over the canonical pack/opaque-region API:

- `resolve_surface_slot_extractor(pack)` returns a pack-declared
  `surface_slot_extractor` when one exists (forward-compatible if a pack later
  adds the slot), otherwise the built-in extractor registered by `pack.pack_id`.
- `_openui_surface_slot_extractor` is registered for the `openui` pack:
  - binder definitions (left-hand side of `name = ...`) are classified as
    `INTERNAL_IDENTIFIER` / `SURFACE_ONLY` (except `root`, which is syntactically
    required and therefore semantic);
  - content placeholders for user-facing string props (`text`, `label`, etc.)
    are classified as `DECORATIVE_TEXT` / `OPAQUE_USER_VALUE` and linked to the
    matching opaque region id (`openui:content:<placeholder>`), spliced through
    the canonical `dsl/opaque_regions.py` `realize_opaque_regions` path;
  - component names, property keys, operators, and other structured fields are
    left out of the surface slot set and remain semantic by default.
- A pack that provides neither a declared slot nor a built-in (for example
  `toy-layout`) resolves to `None` and `realize_surface_and_verify` fails closed.

Regression tests in `tests/test_dsl/test_surface_realization.py`:

- OpenUI surface slot extraction and root exclusion.
- Unknown fields remain semantic by omission.
- Deterministic canonical binder naming and reserved-word collision repair.
- Rejection of structured/observable/comment/docstring slots.
- Internal binder renaming is alpha-equivalent after canonicalization.
- Content placeholders route through the opaque-region splice path.
- Missing required opaque value, unknown assignment, duplicate assignment, and
  tampered assignment all fail closed.
- Failed verifier returns no certified source.
- Missing fingerprint or invalid prior status fails closed.
- Pack without surface extractor fails closed.
- JSON/dict round-trip of results and slots.
- Historical opaque-region behavior remains unchanged.

## Verified

- `ruff check src/slm_training/dsl/surface.py tests/test_dsl/test_surface_realization.py` passes.
- `python -m compileall src/slm_training/dsl/surface.py` passes.
- `pytest tests/test_dsl/test_surface_realization.py -q` → 22 passed.
- No regression: `pytest tests/test_dsl/test_opaque_regions.py tests/test_dsl/test_solver_closure.py -q` stays green.
- `python -m scripts.repo_policy` ok.
- `git diff --check` clean.

## Design boundaries preserved

- Unknown/structured/externally observable fields default to semantic and are
  not exposed as surface-only.
- The deterministic baseline does not generate missing required opaque/user
  values.
- Final output is always canonicalized and re-verified; failure returns no
  certified result.
- Binder substitution uses whole-word matching that excludes placeholders, so
  opaque content region ids are not corrupted during identifier renaming.
- Fixture wiring only; no autoregressive model, trained checkpoint, eval, or
  ship claim.

## Classification table (OpenUI V1)

| Source construct | Kind | Authority | Rationale |
| --- | --- | --- | --- |
| Non-`root` binder definitions (`title = ...`) | `internal_identifier` | `surface_only` | Identity is the symbol, not the spelling; canonicalizer normalizes names. |
| Content placeholders (`:hero.title`, `:cta.label`) | `decorative_text` | `opaque_user_value` | User-facing prose; semantic role is the region/schema, not the bytes. |
| `root` binder | *not extracted* | `semantic` | Syntactically required program entry point. |
| Component names (`Stack`, `TextContent`) | *not extracted* | `semantic` | Runtime/schema meaning depends on exact spelling. |
| Property keys / child refs | *not extracted* | `semantic` | References binders by symbol; structural keys have runtime meaning. |
| Comments / docstrings | *not extracted* | unsupported | OpenUI V1 has no comment syntax to realize into. |

## End-to-end example

Input (solved semantic IR, placeholders preserved):

```text
root = Stack([title], "column")
title = TextContent(":hero.title")
```

Surface slots extracted:

- `openui:binder:title` → `INTERNAL_IDENTIFIER` / `SURFACE_ONLY`
- `openui:content::hero.title` → `DECORATIVE_TEXT` / `OPAQUE_USER_VALUE`

Deterministic realization:

- `title` → `v0`
- `:hero.title` → caller-supplied `:user.title`

After substitution, opaque splicing, canonicalization, and re-verification:

```text
root = Stack([v0], "column")
v0 = TextContent(":user.title")
```

## Caveats

- Only `INTERNAL_IDENTIFIER` and `OPAQUE_USER_VALUE` realization is implemented
  end-to-end. `COMMENT`/`DOCSTRING` slots are rejected; `DECORATIVE_TEXT`
  without an opaque authority requires a pack-declared neutral default that is
  not yet wired.
- Per-language override of assignment validation, application, and the surface
  oracle is reserved for future work; the default pipeline uses direct
  substitution, VSS2-04 splicing, `pack.canonicalize`, and `pack.oracle`.
- No model, checkpoint, or ship gate is claimed; this is fixture wiring for the
  VSS3 solver stack.

## Production generation boundary

`generate_batch_requests(...) -> list[str]` remains unchanged. The opt-in
`generate_batch_bound_requests` path first generates and pack-verifies the exact
canonical template, then returns a `BoundGenerationResult` containing an ordered
typed binding envelope. Declared contract order deterministically maps external
keys such as `hero.title` to model-facing placeholders `:slot_0`, `:slot_1`, ...;
the result preserves the explicit bijection and external role/type metadata.
The model is never required to emit the external key spelling.

The current OpenUI pack rejects arbitrary quoted user-facing content. A focused
literal probe using `Welcome "back"\nToday` therefore selects the honest
template-plus-bindings outcome: `materialized_source` is absent and only the
template is labeled `pack_verified`. Missing, unknown, duplicate, or prefixed
alias bindings fail closed. One binding covers every occurrence of its declared
placeholder. Values remain typed transport data, so quotes, slashes, backslashes,
newlines, Unicode, and empty strings cannot inject structure. Result fingerprints
bind the template fingerprint and ordered value digests; generation evidence
never contains raw values.
