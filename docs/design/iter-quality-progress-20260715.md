# Durable quality-matrix progress (2026-07-15)

An attempted six-step baseline produced its checkpoint and six training metric
rows, but the inline eight-step decode evaluation terminated before writing the
final matrix summary. This is incomplete evidence, not a quality result.

The matrix runner now writes `quality_matrix_progress.json` after every
completed experiment and records an exception as a failed result instead of
discarding all prior results. The final summary remains the authoritative
completed-run artifact.
