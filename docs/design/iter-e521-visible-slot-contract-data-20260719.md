# E521 — visible slot-contract data

E521 closes the prompt-authority mismatch exposed by E519. The E500 projected
corpus carried placeholder lists in record metadata but omitted them from
ordinary generation prompts: only 13/260 rows exposed every placeholder, and
0/209 generation rows did. The E357 replay corpus exposed every placeholder in
998/998 prompts.

## Build

The canonical train-data builder was rerun from the same committed
`remediated_roots` source with strict gates, layout synthesis,
document-only expression projection, a six-record parent cap, and the existing
`--prompt-slot-contract` option. The successful build finished in 4.77 seconds
under the 170-second process cap.

| Corpus | Rows | All slots visible | Mean quality | Quality rejects | Feedback |
| --- | ---: | ---: | ---: | ---: | --- |
| E500 projected control | 260 | 13 | 0.9644 | 0 | 0 recommendations |
| E521 visible-inventory r2 | 244 | 244 | 0.9643 | 0 | 1 recommendation |

The immutable E521 snapshot is published at
`src/slm_training/resources/data/train/e521_visible_slot_contract_r2_20260719/`
with fingerprint `b6a44a1b…a853b7d5`. It includes the quality report, rejected
ledger, synthesis feedback, governance bundle, and manifest.

The first local attempt (`r1`) lacked the OpenUI bridge dependency and correctly
quarantined all candidates at G2. It was not published and is not evidence.

## Feedback and decision

The successful build has no warnings. Its sole recommendation reports
`programspec_generated` yield 0.375 because unchanged semantic dedup removed
near-duplicate variants. Keep the gate unchanged and carry the emitted
producer-yield hypothesis as a future matched synthesis candidate; suppressing
already-rejected variants would reduce work but would not change admitted data.

Publish E521 for a matched bounded continuation. This result establishes prompt
visibility and data quality only; it makes no learned-quality or ship claim.
Machine-readable evidence is in
[the E521 JSON](iter-e521-visible-slot-contract-data-20260719.json).
