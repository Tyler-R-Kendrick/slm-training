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

### E501 sampling follow-up

The E500 snapshot contains 246 generation, 13 repair, and one edit record.
Its published equal-task-group mixture deliberately changes that natural
94.6% generation composition: E501 measured only 65/192 generation examples
(33.9%) in a 5k-token continuation. Uniform record sampling measured 185/198
(93.4%) and recovered component recall from 0.0 to 0.1667, but regressed
structure from the frozen parent's 0.2117 to 0.0889 and left meaningful rate,
fidelity, and reward at zero.

The mixture is therefore not relabeled as defective—it serves multi-task
balance—but generation-only evaluations must report effective task exposure.
The 1k uniform arm avoids the 5k structural collapse but still moves no
semantic gate. See [E501](iter-e501-e396-e500-warm-start-20260719.md).

### E502 initialization-attribution follow-up

E502 shows that a new-corpus warm start can change behavior before meaningful
optimization if corpus-derived serving priors are rebuilt. Future matched data
experiments must report `initialized_prior_fields` alongside effective task
exposure and the complete slot-head recipe. The canonical initializer now
restores checkpoint lexeme/span priors; this raises E396→E500 1k structure
from `0.2317` to `0.3169`, but the retained-prior 5k arm still collapses to
`0.0927`. The corpus conclusion remains unchanged and no gate was relaxed.
See [E502](iter-e502-initialization-prior-retention-20260719.md).

### E503 weight-retention follow-up

The committed E500 feedback remains clean: 260 admitted records, no warnings,
recommendations, or experiment candidates. E503 therefore changes no producer
or acceptance gate. Its matched 5k continuations show that stronger checkpoint
anchoring recovers structure only by losing component recall; the data conclusion
is unchanged. Future parent replay must retain source and parent provenance and
report effective exposure separately from the E500 mixture policy. See
[E503](iter-e503-initialized-weight-retention-20260719.md).

### E504 parent-replay follow-up

E504 uses the exact 998-row E357 parent corpus rather than regenerating an
approximation. The eight-file snapshot is persisted at
`hf://buckets/TKendrick/OpenUI/data/train/e357_card_hierarchy_v1/`; an
independent download reproduced semantic manifest SHA
`a4f212a3444d0f219fe1b3604f70929fe1a1b91d4fdc11a73167cb74c55b6a51`
and records SHA
`b1b2c3d0c1965bd9829edfc6ae34b5dce916a68c33bb17497a6392c80d7ea6ef`.

The E500 and E357 feedback remains clean, so E504 changes no producer or
acceptance gate. Fifty-percent replay restores hierarchy but not placeholder or
semantic behavior; adding retention makes the interaction worse. This localizes
the next investigation to primary-versus-replay objective or output-codec
conflict rather than synthesis admission. See
[E504](iter-e504-parent-corpus-replay-20260719.md).

### E505 source-loss follow-up

E505 changes no synthesis producer or acceptance gate. Both governed corpora'
masked-token loss proxies decline during matched 50% replay; the E500 primary
examples remain about 14–15% harder than E357 replay examples. This falsifies
simple primary-loss divergence but does not resolve gradient alignment. The
next lever belongs in optimizer/decode attribution, not data admission. See
[E505](iter-e505-replay-loss-attribution-20260719.md).

### E521 visible-inventory follow-up

E521 audits the prompt authority mismatch exposed by E519. Only 13/260 E500
records—and 0/209 generation records—made every declared placeholder visible,
versus 998/998 E357 replay records. Rebuilding through the existing
`--prompt-slot-contract` path admits 244 strict-profile rows, all 244 with full
visible inventory, mean quality 0.9643, and zero quality rejects or warnings.

The immutable snapshot is committed at
`src/slm_training/resources/data/train/e521_visible_slot_contract_r2_20260719/`
with fingerprint `b6a44a1b…a853b7d5`. Semantic dedup removes 18 near-duplicate
variants and produces one ProgramSpec yield recommendation; the gate remains
unchanged and the emitted producer-yield hypothesis is retained for a future
matched synthesis experiment. See
[E521](iter-e521-visible-slot-contract-data-20260719.md).

### E522 visible-inventory train follow-up

The matched E522 continuation confirms that the E521 representation changes
learned behavior: OOD placeholder fidelity rises from 0.4083 to 0.8667 and
component recall from 0.2083 to 0.2708. The gain does not compose with
hierarchy—structure falls from 0.2250 to 0.1955, reward falls to 0.2093, and
meaningful rate and AgentV remain zero. Keep visible inventories, but pair them
with component-hierarchy supervision or data rather than stronger slot loss.
See [E522](iter-e522-visible-slot-continuation-20260719.md).

### E524 visible component-contract follow-up

E524 projects an exact component type/count inventory onto immutable E521
without changing its 244 IDs or OpenUI targets. All 244 component contracts are
exact, all declared placeholders remain visible, mean quality remains 0.9643,
and the projection emits zero rejects, warnings, recommendations, or experiment
candidates.

Semantic deduplication and n-gram decontamination are not rerun because E521
already passed them and reapplying content-sensitive dedup changed membership
in a diagnostic candidate. The gates are not weakened for new synthesis; this
is a projection-only matched snapshot. See
[E524](iter-e524-visible-component-contract-data-20260719.md).

### E525 visible component-contract train follow-up

The matched E525 continuation confirms that exact component counts are learned:
OOD component recall rises from 0.2708 to 0.4167. The signal does not compose
with E522’s slot grounding—fidelity falls from 0.8667 to 0.4667, structure to
0.1452, meaningful and strict meaning remain zero, and AgentV remains 0/1.
Keep E524 as conditional-contract data evidence, but reject stronger count
prompting and the E525 checkpoint. See
[E525](iter-e525-visible-component-continuation-20260719.md).

### E527 visible component-types follow-up

E527 weakens E524 from exact type/count inventory to unique component types
only. It preserves all 244 E521 IDs and targets, exposes exact type inventories
and all declared slots in 244/244 prompts, retains mean quality 0.9643, and
emits zero rejects, warnings, recommendations, or experiment candidates. See
[E527](iter-e527-visible-component-types-data-20260719.md).

### E528 visible component-types train follow-up

The matched E528 continuation recovers OOD meaningful rate from 0.0 to 0.25,
fidelity from 0.4667 to 0.55, and reward from 0.1668 to 0.5778 versus E525.
The weaker contract still does not compose into hierarchy: structure falls to
0.1136, strict meaning remains zero, and AgentV remains 0/1. Keep E527 as
conditional-contract evidence, reject the E528 checkpoint, and move the next
lever to semantic-role/reference-graph supervision rather than synthesis gates
or stronger inventory prompting. See
[E528](iter-e528-visible-component-types-continuation-20260719.md).

### E530 visible semantic-role follow-up

E530 groups already-visible slots into semantic namespaces and annotates only
schema-compatible component types already present in each prompt. The first
immutable build accidentally retained default producer expansions; its
176-row result, 239 rejects, high-rejection warning, and redundant-expansion
experiment candidates are preserved but invalid for training.

The corrected recipe explicitly disables all producers and preserves all 244
E521 IDs, targets, and placeholder lists. Every prompt gains a role contract,
174 include compatible visible type candidates, no exact counts are exposed,
and the strict reports contain zero rejects, warnings, recommendations, or
experiment candidates. No gate or producer changes are warranted by the valid
projection. See [E530](iter-e530-visible-semantic-roles-data-20260719.md).
