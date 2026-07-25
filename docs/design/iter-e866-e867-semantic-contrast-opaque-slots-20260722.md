# E866-E867: semantic-contrast opaque-slot producer repair

The repository changed-file gate exposed two stale data paths after the opaque
marker contract became fail-closed: a diffusion test fixture still used named
markers, and the semantic-contrast builder extracted plans from named generator
surfaces. The latter silently rejected every transformed candidate, leaving its
test corpora empty.

The producer now projects source programs before plan extraction, the shared
plan seed builder emits only ordinal `:slot_N` identities, and each compiled
candidate is reindexed from its actual output inventory. Admission now persists
only positive controls that pass and negative transforms that fail the frozen
meaningful-program evaluator; all other candidates remain visible in
`rejected.jsonl`. No gate or threshold changed.

E866 was a partial diagnostic: named markers were eliminated, but five transforms
retained noncontiguous ordinals after deleting an earlier slot. E867 added the
output-inventory reindex and admitted 10 pairs / 20 records. Positive controls
passed meaningful evaluation at 1.0000; admitted binding, content, contract, and
topology negatives passed syntax verification at 1.0000 and meaningful evaluation
at 0.0000, as intended. No named or noncontiguous markers remained. The 17 honest
rejects were four compilation/verifier failures and thirteen negatives that still
passed the evaluator. This is local data-harness smoke evidence, not model training
or a ship claim.

Canonical evidence:
[`iter-e866-e867-semantic-contrast-opaque-slots-20260722.json`](iter-e866-e867-semantic-contrast-opaque-slots-20260722.json).
