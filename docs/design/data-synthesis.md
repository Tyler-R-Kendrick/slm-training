# Data synthesis: final verification

Status: **bounded verification complete; production ship not claimed**.
Machine-readable evidence: [data-synthesis-results.json](data-synthesis-results.json).

Scope-graded families (identity anchors, canonical-form bias, scoped repair,
typed lexical maps) are documented separately in
[scope-graded-data-synthesis.md](scope-graded-data-synthesis.md).

This closes the P1-P13 corpus-integration program at CPU smoke scale. It
verifies the corpus contract, split isolation, governance, edit/task semantics,
and a matched training-data signal. It does not replace a full HF-context run,
the full 1,500-example `rico_held` suite, or the unchanged ship gates.

## Reproduction recipe

The accepted corpus was built twice with the pinned OpenUI 0.2.9 bridge:

```bash
python -m scripts.build_train_data \
  --source all --synthesizer none --programspec-count 1 --rico-limit 1 \
  --allow-missing-design-md --output-root /tmp/slm17-build-c --version v1

python -m scripts.build_train_data \
  --source all --synthesizer none --programspec-count 1 --rico-limit 1 \
  --allow-missing-design-md --output-root /tmp/slm17-build-d --version v1

python -m scripts.build_test_data \
  --source both --rico-limit 20 --max-children 3 \
  --output-root /tmp/slm17-test3 --version v1 \
  --train-manifest /tmp/slm17-build-c/v1/manifest.json

rtk python scripts/verify_data_synthesis.py \
  --first-train-dir /tmp/slm17-build-c/v1 \
  --second-train-dir /tmp/slm17-build-d/v1 \
  --test-dir /tmp/slm17-test3/v1 \
  --task-results docs/design/task-eval-wiring-results.json \
  --baseline-matrix /tmp/slm17-e50-fixture-final-summary.json \
  --champion-matrix /tmp/slm17-e50-integrated-final-summary.json \
  --out docs/design/data-synthesis-results.json
```

The RICO filter is a data-preparation constraint, not a relaxed metric gate.
Nineteen decontaminated `rico_held` rows remain with target-token p95/max 145,
under the 256-token E50 diagnostic budget. No ship threshold was changed.

## Corpus and integrity result

| Check | Result |
| --- | --- |
| Accepted train rows | 176 |
| Reproducibility | both builds `9359d47176c0ec75fce9389558af69884a6ce4afc47f89fc0f7412987ad4bf17` |
| Language contract | active `contract_id` on 176/176; supported component/ref/list/placeholder rows round-trip through `openui-langcore` |
| Deferred syntax | state/query/mutation/action/tool are explicitly unsupported until the pinned contract advances |
| Families | all 10 expected families are present and have root parents |
| Leakage | 0/35 accepted eval rows overlap train by exact, structural, or `split_group_id` checks |
| Eval ceiling | parse, placeholder fidelity/validity, structure, and component recall = 1.0 on smoke/held-out/adversarial/OOD |
| Length | train p95/max 93/117; RICO p95/max 145/145; no checked row exceeds 256 tokens |
| Verifier | Bronze 153, Silver 21, Gold 2; all Gold/Silver rows are green |
| Governance | Croissant, Data Card, and SPDX emitted; fresh PII/secret scan clean; 2 governed complete, 174 internal, 0 quarantined |
| Edits | all replayable edit trajectories satisfy patch application and inverse/undo identity |
| Generalization | held-out slices are populated and decontaminated |

The first verification pass found two source-owner defects rather than hiding
them in documentation: P12 normalization removed `contract_id` from legacy
rows, and P9 counted internal rows as quarantined. PRs #54 and #55 repaired
those defects before this final run.

## Task and equivalence wiring

The deterministic scoreboard covers generation, repair, edit, behavior, and
visual evidence. L3, L4, and L5 equivalence each have an eligible fixture; the
aggregate is 1.0 over three self-consistency cases. These are wiring checks,
not learned-model quality measurements.

## Matched quality signal

Both accepted arms use E50, CPU scratch context, 80 train steps, batch 4,
learning rate `3e-4`, seed 0, honest slot contract, no DESIGN.md context, no
template-fill decode, four effective generation steps, best-of-one, the same
five held-out/five RICO cases, and unchanged ship-gate policy. Only the train
corpus changes.

| Arm | Train corpus | held_out fidelity (n=5) | rico_held fidelity (n=5) |
| --- | --- | ---: | ---: |
| Fixture control | 25-row fixture, fingerprint `a1c82610…` | 0.08 | 0.0667 |
| Integrated candidate | 176-row corpus, fingerprint `9359d471…` | 0.12 | 0.10 |
| Delta | — | +0.04 | +0.0333 |

Both checkpoints still have parse rate 0.0 and fail the full ship gates. This
result is therefore a bounded data signal only. Earlier E0, E41, 200-step E41,
and template-filled E53 probes were negative or non-attributable; none is
promoted.

All comparison checkpoints are local CPU scratch artifacts under `/tmp`.
They intentionally use the repository's no-sync scratch path and are not
reusable champions, bucket artifacts, or production evidence.

## Decision

The integrated corpus is structurally sound and produces a strict matched
fidelity improvement on both required smoke suites. It does **not** ship.
Promotion still requires frozen decoding, rank stability at the largest two
ladder points, three-seed efficiency bounds, full HF context, full
`rico_held`, durable bucket sync, and the unchanged multi-suite ship gates in
[promotion-pipeline.md](promotion-pipeline.md).

## Strict fixture synthesis-feedback repair (2026-07-18)

Machine-readable evidence:
[data-quality-strict-feedback-20260718.json](data-quality-strict-feedback-20260718.json).

Recipe: CPU fixture build, strict profile, quality synthesizer, 20 seeds, and
an outer `timeout --signal=INT --kill-after=10s 170s` cap. This was a bounded
data-quality diagnostic, not an evaluation, checkpoint, or ship claim.

The initial report exposed 12 reserved-structure drops, but attributed all of
them to `family:unknown`. The harness now preserves source-family and
synthesizer lineage on those rejection-ledger rows. The corrected report
identified three leaking `human_curated` roots and nine derived
`prompt_paraphrase` / `template` candidates. The three canonical train seeds
were repaired to remain structurally disjoint from the frozen fixture suites;
no curation threshold or decontamination gate changed.

| Strict fixture build | Admitted | Rejected | Reserved-structure drops | Feedback recommendations |
| --- | ---: | ---: | ---: | ---: |
| Baseline `r1` | 97 | 23 | 12 | 2 unactionable (`unknown`) |
| Attributed control `r2` | 97 | 23 | 12 | 3 actionable |
| Repaired, stamped `r4` | 104 | 14 | 0 | 0 |

The accepted `r4` snapshot is committed at
`src/slm_training/resources/data/train/dq_strict_fixture_r4_20260718/` with
fingerprint `59bc139e…`, `quality_report.json`,
`synthesis_feedback.json`, and `version_stamp/v1` provenance for
`harness.train_data` v1. It is intentionally a tiny fixture corpus for
dashboard inspection and regression coverage; it is not a full HF-context
training corpus.

## Documentized-expression projection (E500, 2026-07-18)

E500 converts language-contract expression tasks into complete `Stack`
documents while preserving source and parent lineage. A separate target-kind
selector retains codec-compatible documents and records intentional exclusions
at the `selection` stage without turning them into false producer-yield
warnings.

The corrected candidate contains 260/260 choice-compatible rows, 87 root
parents, 72 program families, and 241 structural families, with zero quality
warnings or synthesis-feedback recommendations. It is committed at
`src/slm_training/resources/data/train/e500_documentized_expression_candidate_r2_20260718/`
with fingerprint `bc256915…463bc62`. The first singleton projection was
invalidated after 15 placeholder-contract failures.

Matched 1k and 5k frozen-context smoke trains did not improve model quality:
all four arms have meaningful rate, fidelity, recall, and reward 0.0, with
AgentV 0/1. The generalized projection and clean corpus remain useful data
infrastructure, but no E500 checkpoint is promoted or synced. See
[the E500 record](iter-e500-documentized-expression-corpus-20260718.md).
