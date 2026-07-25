# E932-E938 — role-aware opaque identifiers across every target

Date: 2026-07-22. These were deterministic CPU data builds under the repository
110-second command cap; no model training or evaluation ran.

## Finding

The prior symbol-only sanitizer was identity-safe but not property-role-safe. It
could turn required structural strings such as `Form.name`, `Input.name`,
`Slider.name`, and `TabItem.value` into content slots. It could also preserve an
English enum atom such as `email` merely because that atom was legal somewhere
else in the schema. E932 proved the first opaque-ID patch was incomplete: 116
role violations remained across 500 parseable structured targets, and the
sanitizer fell back 128 times. E932 is rejected evidence, not a promoted corpus.

The repaired contract assigns traversal-local quoted structural atoms (`"$0"`,
`"$1"`, ...) to required identifier/value fields, permits an enum only for the
exact property declaring it, and reserves `:slot_N` for content-bearing fields.
The same transform now covers document, statement, and expression targets.
The follow-up audit found that E933's primary targets were clean but document-kind
`accepted_outputs` had passed through the lexical fragment sanitizer. Evaluation
consumes those alternate targets, so E933/E934 are rejected as incomplete.

## Promoted evidence

| Run | Artifact | Result |
| --- | --- | --- |
| E933/E934 | prior role-aware pair | Rejected: primary-only audit missed role-unsafe accepted targets |
| E935/E936 | first all-target rebuild | Rejected: generated before the owning component version bumps |
| E937 | `data/train/e937_role_safe_all_targets_v2` | 524 admitted from 1,714 candidates; 0 sanitizer fallbacks; 0 role violations across all 582 primary and accepted targets |
| E938 | `data/eval/e938_role_safe_all_targets_v2` | 50 records across smoke/held-out/adversarial/OOD/RICO; 0 build errors; 6 train-overlap candidates rejected; 0 role violations across all 50 targets |

E937 fingerprint is
`23e81f05616592a29c755a96849da1af279e40ee075a4353277b36739af679e1`.
The train records SHA-256 is
`72b145eefa0b61d92e15ea89aaa159377b99cb2a5d8afbcb13eb0469ec24ae0f`.

The synthesis feedback emitted 26 recommendations and 26 experiment candidates.
Awwwards-derived rows remain at zero yield because their G10 provenance is
incomplete; this is an honest quarantine and no gate was weakened. Eight
language-contract expressions that could not satisfy the role-aware transform
were likewise rejected rather than relabeled.

## Decision

Promote E937 and E938 as the canonical train/eval defaults. Keep older snapshots
immutable for historical reproducibility, but do not select them for new model
training or evaluation. The shared model-data loader now independently rejects
role-unsafe primary and accepted targets before training or evaluation; a direct
audit loaded all 524 E937 records and all 50 E938 records, while correctly
blocking E933 and historical E826 on structural content-slot violations. This is a
data-contract promotion only: no checkpoint was written, AgentV is not applicable,
and no ship-gate claim is made.
