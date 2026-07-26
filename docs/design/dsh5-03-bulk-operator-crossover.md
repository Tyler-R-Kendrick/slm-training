# DSH5-03 bulk-operator crossover

SLM-411 owns the comparison contract for repeated primitives, one exact bulk
action, and full generation. It reuses the shared full-denominator serving
work contract from DSH3-32; it does not create a second latency accounting
path.

## Required matched arms

Every measured stratum must share target-model revision, device, precision,
batch/cache mode, context fingerprint, and meaningful-parse quality. The
required arms are `primitive_policy`, `oracle_primitive_sequence`,
`bulk_policy`, `bulk_disabled`, and `full_generation`. Legal-set construction,
dry runs, execution, validation, materialization, failures, timeouts, CPU,
device, wall time, caching, and model forwards remain in the denominator.

The only primary systems metric is target-model forward equivalents. A
crossover is reported only between observed fanout/state/heterogeneity strata;
the harness never interpolates a threshold or treats an incomplete arm as a
benefit.

## Exact bulk admission

`openui.map_set_property` enters the matrix only through the live bounded
operator legal set. Selector candidates are now enumerated from the exact
state-bound `ReferenceTableV1.selectors` collection and appear in the
sanitized policy input as an opaque row carrying only its closed selector
kind, allowlisted compiler facts, cardinality, and fanout. It carries no
target identity, target order, AST payload, scope fingerprint, or opaque ID.

The bounded local preflight at
[`dsh5-03-bulk-operator-crossover-20260726-local/report.json`](dsh5-03-bulk-operator-crossover-20260726-local/report.json)
executes real pack/legal-set/apply/replay/lowering checks at fanout 1/2/4/8.
Each fixture row has a complete legal action and exact replay plus
primitive-lowering equality. This is wiring evidence only, never model or
serving evidence.

## Current disposition

The report is **unavailable**, not a crossover result. DSH3-28 rejected the
compatible typed-policy candidate, and no matched five-arm serving rows exist.
No fixture result substitutes for held-out meaningful parse, no human rating
is a gate, and no checkpoint, remote/HF workload, or ship claim is made.

The next run must supply compatible local measurements for every required arm
and retain all no-effect, fallback, timeout, and invalid rows. If no powered
observed stratum improves matched meaningful quality and full serving work,
the learned bulk-policy claim is rejected while the exact executor remains a
compiler utility.
