# Attribution window (2026-07-15)

A six-step baseline window emitted 24 attributed records with persisted
telemetry and run insights. The rough source-family proxy was highest for
`rico+aug`, but the counts were small and each record inherited its optimizer
batch's aggregate loss. Length buckets also overlapped (`short` 62.77,
`medium` 73.11, `long` 66.07).

This is not enough evidence to rebalance the corpus. The next useful upgrade is
either a larger attribution window or per-example loss instrumentation; no
training-data change is justified by this sample.
