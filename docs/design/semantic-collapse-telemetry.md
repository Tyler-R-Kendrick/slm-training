# Semantic-collapse telemetry

`SemanticTrajectoryTelemetryV1` is a default-off, immutable observation layer.
It reuses existing constraint-debt, grammar-trace, decode-statistics, and
semantic-failure evidence; it neither changes training nor changes decoding.
`SemanticEarlyWarningPolicyV1` is advisory: unknown or partial features always
produce `unknown_no_action`.

Exact empty-program mass is reported only for a declared complete finite set.
Partial/unknown compiler coverage is never upgraded to exact; the direct
minimal-vs-populated score remains a separate feature.

## Measured inventory — 2026-07-24

The [result JSON](iter-slm264-semantic-collapse-telemetry-20260724.json)
records the local archive audit and one committed schema fixture. Historical
run directories retain final/best checkpoints and scalar loss telemetry, but
not the four immutable intermediate checkpoints per trajectory or raw
distribution telemetry required for an early-warning predictor. The result is
therefore `insufficient_trajectory_evidence`; no predictor, automatic stop, or
quality claim is made.
