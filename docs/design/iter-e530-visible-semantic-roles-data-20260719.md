# E530 — visible semantic-role data

E530 tests the smallest prompt-side follow-up to E528's hierarchy failure:
group already-visible placeholder slots by namespace and, where the existing
schema allows it, name compatible component types that are already present in
the visible type inventory. The contract never exposes output counts or an
exact parent/child graph.

The first immutable build (`r1`) was invalid for the matched comparison.
Default producer expansions remained enabled, collecting 354 candidates and
admitting 176 rather than preserving the 244-row E521 parent. It is retained as
negative process evidence and must not be used for training. Its feedback
reported a 0.4972 rejection rate plus redundant `corruption_repair` and
`edit_trajectory` expansions. Those experiment candidates are valid for a
future synthesis-yield study, but do not answer this projection-only
hypothesis.

The corrected `r2` recipe explicitly disables every producer and derives only
from immutable E521. It preserves all 244 IDs, OpenUI targets, and placeholder
lists. E521 already passed semantic deduplication and decontamination, so those
content-sensitive gates are not rerun.

| Check | Result |
| --- | ---: |
| Rows / IDs / targets preserved | 244 / 244 / 244 |
| Rows with semantic-role contracts | 244 |
| Rows with schema-compatible visible type candidates | 174 |
| Type-candidate mentions | 274 |
| Count tokens exposed by the new contract | 0 |
| Mean quality | 0.9643 |
| Rejects / warnings / recommendations | 0 / 0 / 0 |

Both builds completed in under five seconds under the 170-second process cap.
The valid immutable snapshot is committed at
`src/slm_training/resources/data/train/e530_visible_semantic_roles_r2_20260719/`
with fingerprint `e65a6ac5…0eccee2`. The invalid recipe-drift snapshot remains
at the matching `r1` path so its rejection report and feedback are auditable.

Publish only E530 `r2` for one matched bounded continuation. This is
conditional-contract data evidence, not learned behavior or a ship claim.
Machine-readable evidence is in
[the E530 JSON](iter-e530-visible-semantic-roles-data-20260719.json).
