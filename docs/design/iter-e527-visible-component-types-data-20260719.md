# E527 — visible component-types data

E527 is the minimal follow-up to E525’s recall/fidelity tradeoff. It removes
exact output counts from E524’s prompt contract while retaining the unique
component type inventory and the E521 slot inventory.

The projection derives directly from immutable E521, performs no new
synthesis, and preserves all 244 IDs and OpenUI targets. E521 already passed
semantic deduplication and decontamination, so those content-sensitive gates
are not rerun.

| Check | Result |
| --- | ---: |
| Rows / IDs / targets preserved | 244 / 244 / 244 |
| Exact visible type contracts | 244 |
| All declared placeholders visible | 244 |
| Mean quality | 0.9643 |
| Rejects / warnings / recommendations | 0 / 0 / 0 |

The build completed in 4.39 seconds under the 170-second process cap. The
immutable snapshot is committed at
`src/slm_training/resources/data/train/e527_visible_component_types_slot_contract_r1_20260719/`
with fingerprint `84c4ee8e…fe90fc5a`.

Publish E527 for one matched bounded continuation. This is conditional-contract
data evidence only; learned behavior and ship readiness require the standard
honest evaluation. Machine-readable evidence is in
[the E527 JSON](iter-e527-visible-component-types-data-20260719.json).
