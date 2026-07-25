# ProgramSpec coverage-scaling wiring evidence

SLM-267 (VSD2-02) asks for uniform-vs-coverage-targeted 10k/100k/1M ProgramSpec
corpora. The repository's existing coverage-guided generator (SLM-5,
`slm_training.data.progspec.generate.ProgramGenerator`) samples from a fixed,
exhaustible candidate grid — a few dozen root/topology/prop combinations per
small component set, not an open-ended program space — so those rungs are out
of reach without a materially different generator. This issue is scoped down
to what the existing generator actually supports: a genuine uniform-random
control arm, a deterministic `(global_seed, shard_id, worker_id)` sharding
primitive, and an honest measurement of the generator's own state-space
saturation.

The committed [disposition JSON](iter-slm267-programspec-coverage-scaling-20260725.json)
records the exact recipe, per-arm counters, and comparison. Rerun the exact
command to regenerate it:

```bash
python -m scripts.generate_programspec_corpus --mode fixture --target-count 80 --seed 0 --shards 2
```

## Measured fixture-scale campaign — 2026-07-25

Two shards of a 3-component, depth/width-3 candidate grid (64 target coverage
cells, 58 coverable) were sampled under two policies up to an 80-program
budget per shard:

- **`uniform`** — `ProgramGenerator.generate_uniform`, a new method added in
  this change that draws uniformly at random from the unused candidate grid
  (no coverage bias).
- **`coverage_targeted`** — the generator's existing `generate_one`, which
  greedily maximizes `CoverageTracker.score()` (always prefers a candidate
  touching an uncovered cell).

Both policies exhausted the finite candidate grid at 36 accepted programs per
shard (8 duplicate canonical roots via serialization collapse, 28 unique
roots) — the grid's exact saturation point at this component/depth/width
setting. `coverage_targeted` reached full coverable-cell coverage in 19
programs; `uniform` took 34–35. This reproduces deterministically across
shards and reruns (see `shard_seed` and the manifest hash in the disposition
JSON).

## Honest scope

This is wiring evidence only:

- It does **not** validate VSD-H7a (10k/100k/1M reachable unique roots),
  VSD-H7b (coverage-targeted beats uniform on strict semantic outcomes at
  matched exposure), or VSD-H7c (scale moves semantics) — those require
  corpora and trained models several orders of magnitude larger than the
  fixture budget used here, and no model training occurs in this issue.
- `coverage_targeted` reuses the generator's internal coverage bias; it does
  not yet consume an external `CoverageGapManifestV1` (SLM-265) gap manifest.
  Wiring that mapping honestly (rather than inventing an unverified feature
  correspondence between SLM-265's record-level domain-shift cells and this
  generator's grammar-coverage cells) is left as explicit follow-up work.
- Canonical-root dedup here is in-memory only; a disk-backed exact index
  required at 100k/1M scale is not implemented.
- No corpus is published to the `DataStore`, and no ship gate, checkpoint, or
  production default is touched.

Disposition: **`inconclusive`** — the mechanism (uniform control, deterministic
sharding, greedy coverage-targeting, in-memory canonical dedup) is real,
tested, and reproducible, but the scale required to resolve VSD-H7a-c remains
unaddressed.
