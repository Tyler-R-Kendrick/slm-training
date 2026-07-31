# LeverProof Lean

LeverProof Lean is a Lean 4 implementation of the experiment-resource metric
oracle used by `slm-training`. It consumes raw integer samples and provenance
digests, derives exact rational and interval metrics, selects a candidate with
the committed lexicographic policy, and emits a replayable certificate.

This project is part of the `slm-training` monorepo. The initial import preserves
external snapshot
`51a03f0cc410ef3e0591862f32b3393307280d1e`; the in-repo path is authoritative
for subsequent changes. The executable protocol is:

- input: replayable `metric_evidence/v1` or band-aware `metric_evidence/v2`;
- output: the matching `metric_certificate/v1` or `metric_certificate/v2`;
- replay: exact recomputation from the original evidence;
- arithmetic: natural numbers and unreduced rational pairs, never floats.

## Build and verify

Install Elan, then run:

```sh
lake build
make test
```

The project pins Lean in `lean-toolchain`. `make proofs` compiles the public
proof audit and rejects `sorry` or declared `axiom` in project sources.

```sh
.lake/build/bin/leverproof-lean check Test/resource.json
.lake/build/bin/leverproof-lean verify \
  Test/resource.json Test/resource.certificate.json
```

## Proven core

Lean proves:

- every observed sample lies in its derived minimum/maximum interval;
- a non-empty sample mean has a positive denominator;
- piecewise model evaluation uses a declared box;
- interval addition, multiplication, natural powers, and inversion contain
  every point admitted by their input intervals;
- complete metric-program evaluation contains the corresponding point
  evaluation for every dependency assignment inside the declared boxes;
- selection returns a member of the candidate set;
- the selected candidate globally minimizes the primary quality-failure key;
- a checked certificate's selected candidate beats or ties every candidate
  under the complete committed lexicographic policy;
- successful checking requires valid evidence;
- a checked selection belongs to the derived candidates and satisfies the
  primary and complete finite-candidate optimum theorems.

The v2 protocol also evaluates generic reverse-Polish metric programs over
named rational intervals, classifies every raw natural-number observation as
below, within, or above the calculated interval, and records whether the range
is theorem-backed or assumption-backed. `countPositions_total` proves that no
observation disappears during classification; the classifier theorems
characterize all three positions exactly, and `relation_inBand_iff` proves
that in-band is equivalent to having no below/above samples.

## Core formal claims (Mathlib-free)

Self-contained axiomatized theories for the structural safety layer (no Mathlib):

| Module | Claims |
| --- | --- |
| `Forest` | certified closure is monotone, idempotent, history-preserving, never adds live candidates; lossy history is refuted |
| `Trace` | accepted steps apply certified removals and prefix-extend history |
| `StructuralMetrics` | recall / structural-similarity / means are monotone under declared inequalities |
| `ExactClosure` | VSS exact-closure removes only replay-valid UNSUPPORTED; passes only shrink |
| `DecodeInvariants` | I1/I2/I6 commit rules: singleton bypass, empty=dead-end, ranker ⊆ legal |
| `EcosystemTier` | production-core vs ecosystem-library formal partition; core success ignores library size |

Design notes: [`docs/design/core-formal-claims.md`](../../docs/design/core-formal-claims.md),
[`docs/design/ecosystem-tier.md`](../../docs/design/ecosystem-tier.md).

## Trust boundary

Lean checks the arithmetic and selection propositions. Measurement truth,
sensor calibration, experiment design, JSON decoding, filesystem I/O,
SHA-256 implementations, the Lean runtime/compiler, and the operating system
remain trusted. A certificate binds conclusions to evidence; it does not prove
that the observations were unbiased.

The model language covers affine and polynomial expressions, box-selected
piecewise expressions, and integer inverse powers. Unsupported models fail
closed. Rational-power fits must be lowered to a certified piecewise envelope.
