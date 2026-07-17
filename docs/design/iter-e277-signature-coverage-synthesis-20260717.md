# E277 — signature-coverage synthesis

Date: 2026-07-17
Status: **completed; harness retained, candidate corpus rejected**

E277 tests whether broader grammar-state sampling repairs E276's sparse
preference data. The harness now:

- samples distinct compiler states by `decision_kind`, legal token set, and
  selected gold-AST token rather than repeatedly sampling one coarse kind;
- recursively consumes sharded trace stores;
- records support-signature counts and metadata in immutable manifests;
- requires train support for every held-out support signature before a
  counterfactual corpus can be admitted; and
- validates the manifest before writing events or evidence, so a failed gate
  cannot leave a partial training corpus.

Objective signatures still include judged bad-token sets for gradient
diagnostics. Corpus support signatures intentionally exclude bad-token sets,
because those depend on which legal alternatives were sampled. Admission is
based on the stable grammar state and judged positive:
`decision_kind + legal_token_ids + good_token_ids`.

The run used the unchanged E228 checkpoint, all 65 independently judged E230
document records, strict compiler-tree decoding, gold-AST state sourcing, seed
277, six signature-diverse states per record, and up to three legal candidates
per state. It began from a clean branch rebased on current `origin/main`,
proving `0 behind / 1 ahead` at `187ab5c`. Five CPU-capped source shards were
later resharded only at completed record boundaries; the final 65 record IDs are
unique.

## Measured result

| Measure | Result |
| --- | ---: |
| Accepted traces | 65 / 65 |
| Exact states replayed | 390 |
| Grammar-legal candidates | 943 |
| Independent-judge pass | 631 / 943 |
| Fully verified candidates | 516 / 943 |
| Qualified events | 362 |
| Train / held-out events | 301 / 61 |
| Qualified groups | 64 |
| Set-valued events | 161 |
| Train support signatures | 72 |
| Held-out support signatures | 23 |
| Covered held-out signatures | 14 |
| Uncovered held-out signatures | 9 |

The immutable-event manifest contains the 362 retained probes. Within those
qualified probes, 573/885 candidate outputs pass the independent judge and
471/885 are fully verified. The wider recipe improves event volume over E261
(239 to 362), but it does not satisfy semantic support coverage.

An initial 12-state/four-candidate recipe was stopped after four traces because
its measured per-record cost was unbounded for this campaign. Those traces are
excluded. The completed six-state recipe is the only source for the headline
numbers.

## Admission decision

Reject the candidate corpus. The strict build correctly raises
`decision event corpus lacks train support for 9 held-out signatures` before
writing events, evidence, or a manifest. The diagnostic manifest was emitted
only with the explicit `--allow-sparse-signatures` override and is not located
under committed training resources.

Seven missing support signatures already exist in train gold ASTs but were not
retained by bounded probing. Two are absent from the E230 train split:
bound `TextArea` selection and one bound-empty-list closing-bracket state. The
next repair must target grammar-state metadata, mine the seven existing train
states, and synthesize independently judged train records for the two absent
patterns without copying held-out prompts or programs.

Machine-readable evidence:
[`quality-matrix-v10-e277-signature-synthesis-results.json`](quality-matrix-v10-e277-signature-synthesis-results.json).
