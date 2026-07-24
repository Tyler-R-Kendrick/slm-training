# Harness artifact audit (SLM-281)

SLM-281 provides a byte-preserving replay protocol for archived evaluation
failures. It preserves the stored prediction bytes locally, records their
SHA-256, and never regenerates a model output. Archive-derived reports must
remain under local `outputs/` unless the source data is explicitly approved for
publication.

Each replay record carries `HarnessProvenanceV1`: source-evaluation digest,
evaluation policy, timeout, canvas cap, parser fallback, repair policy,
target length, browser, runtime, verifier, and raw/constrained/repaired
identifiers. Canonical model-build evaluation artifacts emit one suite-level
provenance record plus a stable provenance ID on every detail row. Older
records without those fields retain `unknown_not_captured`; replay-time
feasibility is not attributed to the original decoder.

The harness can classify byte-preserving failures as stable, timeout-, canvas-,
or truncation-sensitive. It records `actual_decode_replayed=false`: archived
outputs lack the original decoder trace, so a feasibility sweep is not a
causal re-decode experiment.

No archive-derived result is committed here. Such a report is diagnostic only,
not an architecture or ship claim. A future publishable causal protocol must
freeze an authorized input corpus and capture complete provenance at evaluation
time before using perturbation results to support architecture conclusions. A
missing original decoder trace yields unknown flip and architecture-blocking
values, never a fabricated zero-percent result.
