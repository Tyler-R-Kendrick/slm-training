# Batch-attributed telemetry (2026-07-15)

Training metrics now retain the record IDs, normalized source families, and
prompt/target character lengths for every record contributing to an optimizer
step. For accumulation windows this includes all microbatches, not only the
last one. A two-step `grad_accum=2` probe emitted eight records per row and
persisted `run_insights.json`.

This makes future loss spikes actionable: feedback can be joined to concrete
examples and source families before changing the corpus or recipe.
