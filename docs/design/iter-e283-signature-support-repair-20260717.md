# E283 — grammar-state support repair

Date: 2026-07-17
Status: **completed; corpus admitted for bounded preference training**

E283 repairs the nine uncovered held-out support signatures from E277 without
adding token, component, or test-case branches to the compiler. The collector
now accepts target states defined by compiler-derived metadata only:
`decision_kind + legal_token_ids + selected_token_id`. Exact record IDs can be
selected independently, and source manifests can be repeated when a corpus
spans multiple judged datasets.

The run started from merged `origin/main` at `9dff17d`. Six independently
judged E230 train-group records supplied seven target states already present in
their gold ASTs. Two remaining states occurred only in held-out preference
groups, so their prompts and programs were not copied. Instead, two
non-isomorphic generation records were produced from the component schema and
grammar, assigned stable train groups, and admitted only after:

1. independent prompt/output judging scored each pair `1.0`;
2. meaningful-program verification passed;
3. compiler decisions exactly matched the missing state metadata; and
4. all user-facing content was represented by placeholders.

The source records are committed at
`src/slm_training/resources/data/train/e283_signature_support_synth_v1`.
The web data reader exposes that version and both records. The admitted
decision corpus is committed at
`src/slm_training/resources/data/preference/e283_signature_support_repair_v1`;
the web data reader classifies it as semantic preference training.

## Measured result

Recipe: unchanged E228 checkpoint (`7a9be4a6…f5b093a`), CPU, strict
compiler-tree decoding, gold-AST state source, seed 283, at most three legal
candidates per exact target state. No model training ran in E283.

| Measure | E277 rejected candidate | E283 admitted corpus |
| --- | ---: | ---: |
| Decision events | 362 | 372 |
| Train / held-out | 301 / 61 | 311 / 61 |
| Train groups / held-out groups | 53 / 11 | 55 / 11 |
| Train support signatures | 72 | 81 |
| Held-out support signatures | 23 | 23 |
| Covered held-out signatures | 14 | 23 |
| Uncovered held-out signatures | 9 | **0** |
| Qualified judge probes | 362 | 372 |
| Candidates in qualified probes | 885 | 915 |
| Independent-judge pass | 573 | 597 |
| Fully verified | 471 | 488 |

The strict `--min-train-signature-support 1` gate passes. The immutable corpus
fingerprint is
`f4e4f5230488a5b58950fc890e41ca4897ff73350f6241737c39f26e96650090`;
its manifest records both source-corpus fingerprints and all 372 judge probes.

## Rejected intermediate attempts

- A first two-record collection used the seven-state train-existing target
  subset and correctly produced zero target states. It is excluded.
- The first all-nine-state synthetic pass found both targets, but meaningful
  verification rejected literal user-facing copy despite the independent
  pairing judge passing. The records were corrected to placeholders and
  re-judged; the rejected traces are excluded.
- Combining E277 with only the seven existing-train repairs left exactly the
  two held-only signatures uncovered. The fail-closed builder wrote no partial
  corpus.

## Decision

Admit the E283 source and preference datasets. The next model experiment may
use the committed E283 event corpus, but must first rerun the E275/E276
metric/signature profile. Any actual training command must fetch and reconcile
latest `origin/main`, prove a clean worktree, and prove zero commits behind
immediately before launch.

Machine-readable evidence:
[quality-matrix-v10-e283-signature-support-results.json](quality-matrix-v10-e283-signature-support-results.json).
