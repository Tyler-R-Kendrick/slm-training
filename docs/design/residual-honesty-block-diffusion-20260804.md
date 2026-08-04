# Residual honesty + block-diffusion wiring (2026-08-04)

Four-layer delivery triggered by adversarial review of three external
analyses of the templated-marker-blocks proposal. Claim class:
wiring/fixture — no quality or ship claim. Companion campaign:
[`precompiled-grammar-admissibility-20260804.md`](precompiled-grammar-admissibility-20260804.md).

## L-A — engine false-admits + honest residual authority (PR #1421, merged 36719aeb)

Two live legality bugs found while building a bounded oracle to verify an
external counterexample, both fixed and pinned:

1. **Retry false-admit**: `set_prefix(X)` → False, then the same `X` again →
   True (rejected `_prefix` satisfied the already-synced fast path). Fixed
   with `_synced_ok` (copied across forks; honored by `engine_in_sync` and
   the stateless already-synced check).
2. **Delta false-admit**: a rejected incremental delta returned the recovery
   resync's own success (`set_prefix("root = = ")` on an engine synced at
   `"root"` → True; fresh oracle False). Now resyncs AND reports False.

`joint_multi_hole_support` no longer stamps `authority="exact"` for
hole-bearing canvases (they carry `left_prefix_overapprox`): `admit_fill`
never validates tokens after the first hole, and the shipped default
`remask_ratio=0.0` means committed suffixes are never rewritten — the
historical "holes may rewrite the suffix" justification does not hold.
Counterexample `root [HOLE] \n )` (prefix admissible, joint canvas
uncompletable) proven by whole-vocabulary enumeration. En-route discovery:
comment-byte fills legalize any *same-line* suffix, so exact checkers must
model comment semantics.

## L-B — blast-radius measurement

Advisory counters `admit_probe_canvases` / `admit_probe_committed_suffix`
at the two live MaskGIT admit sites. Measured on the profile fixture
(s1_d64, 1 generate): **442/442 probes (100%)** are in the suffix-blind
configuration — the blind spot is the norm, not an edge case.

## L-C — exact multi-region support

`multi_region_support` (residual_support.py): bounded exact multi-region
completability under the system's own decode-then-parse semantics.
Memoized DFS on `(position, parser_state_key)`; hole branching over
currently-lexable candidates plus one epsilon fill (empty-piece tokens);
budget exhaustion is typed `authority="unknown"` (fail-closed, never
conflated with proven rejection). Differential tests: rejects the proven
counterexample the left-prefix probe admits; agrees on completable single-
and multi-hole canvases.

**Field measurement (HX2)**: 60 admitted canvases sampled from a real
fixture decode, re-checked exactly — **0 jointly impossible, 0 unknown**.
Combined with L-B: exposure is universal but sequential per-position
commits (with pick-prevalidated prefixes) rarely realize joint
impossibility. The checker's production consumer is parallel block
commits (L-D), where independent per-position admits genuinely compose.

## L-D — block_diffusion_decode lever

The block-diffusion layer (`block_noise.py`, `constrained_posterior.py`)
was previously **unreachable** (zero production importers) — the repo's
production decode is positionwise MaskGIT. Now wired behind a default-off
lever:

- `block_diffusion_decode` / `block_diffusion_block_size`: unmask
  selection groups positions into fixed blocks (`BlockNoiseSchedule`);
  positionwise commit/admit machinery unchanged.
- Parallel step commits jointly validated by `multi_region_support`;
  proven-impossible canvases revert to masks (`block_joint_rejections`
  counter). Unknown keeps commits — fail-closed direction only.
- `pick_constrained_production` confidence bug fixed: reports the
  SURVIVING candidate's probability, not the unconstrained argmax's
  (which could belong to a rejected token).

Evidence: block-lever smoke test generates grammar-valid output through
the full loop; lever-off profile outputs **byte-identical** across L-B,
L-C, L-D (7.32 → 7.26 s/gen, within the declared ±10% noise band).

## Honesty

Fixture-scale evidence on one local checkpoint (s1_d64; the committed demo
predates the output contract) and fixture prompts. The engine false-admits
were latent on this fixture (byte-identical outputs pre/post fix) but
reachable via retries and stateful callers. The block path is wiring only:
no quality suite was run for lever-on decode; enabling it by default would
require its own preregistered campaign. The 0/60 HX2 result bounds the
realized left-prefix FP rate on THIS fixture only; it is not a general
soundness claim for the over-approximation.

## Follow-ups filed

- HX3/HX4 (executable static control + acceptance-length bitsets) continue
  under the precompiled-admissibility campaign.
- HX5: cross-attempt session reuse in `_ensure_valid_openui` (3 fresh
  sessions per certify) — cheapest untried large win.
- Block-lever quality screening (preregistered) before any default flip.
- Duration-aware CI sharding (from the earlier delivery) remains open.
