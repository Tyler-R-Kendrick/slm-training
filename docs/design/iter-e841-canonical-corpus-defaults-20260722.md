# E841: canonical corpus defaults

## Outcome

The tracked resource audit found 11,029 persisted record rows across 47
`records.jsonl` files. Only 1,345 rows use canonical opaque marker identities;
9,684 historical rows contain named markers and are intentionally ineligible.
They remain immutable provenance, but the shared record loader rejects them
before model construction or training.
The dashboard's audit row reports JSON parse rate 1.0 and marker-policy
fidelity 0.1220 over that full historical inventory.

The active E826 train corpus loads 350/350 rows and the active E827 eval corpus
loads all 23 rows across smoke (3), held-out (5), adversarial (4), OOD (4), and
RICO-held (7). No model converts marker names. Builders own canonicalization;
loaders enforce it.

## Harness correction

`DEFAULT_TRAIN_DATA_DIR` and `DEFAULT_EVAL_DATA_DIR` now live beside all other
discoverable levers in `slm_training.levers`. Training, evaluation, quality,
grammar, performance, phase, autoresearch, diagnosis, migration, baseline, and
model-cycle entrypoints consume those constants. Missing `outputs/data/*/v1`
paths and legacy `remediated` corpora are no longer implicit defaults.

Historical named-marker snapshots were not rewritten because that would falsify
their experiment provenance. They cannot be selected accidentally through a
canonical default and still fail closed if explicitly supplied. No training,
checkpoint, sync, remote compute, deployment, or ship evaluation occurred.
