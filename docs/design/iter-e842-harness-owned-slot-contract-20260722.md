# E842: harness-owned slot contract

## Outcome

Template-marker conversion is now exclusively a data-harness responsibility.
TwoTower, grammar diffusion, and tree-edit diffusion no longer add missing
colons, infer semantic marker names from prompt words, or surface hidden gold
markers. `GenerationRequest` rejects anything except a contiguous opaque
`:slot_N` inventory before a model sees the request.

The strict test-data builder produced the immutable
`e842_harness_owned_slots_v1` snapshot from the existing fixture and local RICO
sources against the E826 train manifest. It retained all 23 expected rows:
smoke 3, held-out 5, adversarial 4, OOD 4, and RICO-held 7. There were zero
normalization errors, zero sanitizer fallbacks, and all 23 rows passed the
shared persisted-record loader with no named target marker. Thirty-three
train-overlapping candidates were rejected.

Fixture normalization failures are now fatal rather than being recorded and
silently omitted. RICO conversion failures remain reported source-ingestion
rejects. The single `DEFAULT_EVAL_DATA_DIR` lever now selects E842 for every
canonical evaluation entrypoint.

No training, checkpoint, model evaluation, remote compute, sync, deployment,
or ship claim occurred. Focused harness, model, request-contract, integrity,
surface-realization, and test-data tests passed 59/59.
