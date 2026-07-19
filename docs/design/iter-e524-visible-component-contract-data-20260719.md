# E524 — visible component-contract data

E524 isolates the next prompt-authority lever after E522 improved slot fidelity
but regressed hierarchy. Earlier component-edge and binder-topology objectives
learned hidden gold targets without improving aggregate decoding. E524 instead
makes the component-plan target observable by appending an exact component
type/count inventory to each training prompt.

## Matched projection

The successful candidate derives directly from immutable E521 and performs no
new synthesis. E521 already passed strict semantic deduplication and
decontamination, so those two content-sensitive gates are not rerun. All
quality and independent-verification gates still run.

| Check | E521 | E524 r4 |
| --- | ---: | ---: |
| Rows | 244 | 244 |
| Identical IDs | — | 244/244 |
| Identical OpenUI targets | — | 244/244 |
| Exact visible component contracts | 0 | 244 |
| All declared placeholders visible | 244 | 244 |
| Mean quality | 0.9643 | 0.9643 |
| Quality rejects / warnings | 0 / 0 | 0 / 0 |

The build completed in 4.63 seconds under the 170-second process cap. Its
immutable source-controlled snapshot is
`src/slm_training/resources/data/train/e524_visible_component_slot_contract_r4_20260719/`
with fingerprint `56c035c1…1825d37a`.

## Diagnostics and feedback

Three local candidates were rejected before publication. Applying component
text before semantic dedup admitted only 230 rows; projecting after admission
from pre-E521 roots restored 260 rows; re-deduplicating E521 removed one
additional row. None is evidence. The r4 projection preserves E521 membership
exactly and emits zero feedback warnings, recommendations, or experiment
candidates, so no producer or acceptance gate changes.

Publish E524 for one matched bounded continuation against E522. This is
data-quality evidence only. Component counts are a stronger conditional prompt
contract than ordinary OOD requests, so a learned gain would not establish
unconditional ship readiness. Machine-readable evidence is in
[the E524 JSON](iter-e524-visible-component-contract-data-20260719.json).
