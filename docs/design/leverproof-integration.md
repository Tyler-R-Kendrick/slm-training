# LeverProof integration

LeverProof Lean is the executable oracle for resource-derived experiment
metrics. This repository remains the owner of raw measurement, feature-flag
snapshots, campaign governance, model-quality gates, and checkpoint promotion.

```text
raw integer samples + content digests
  -> metric_evidence/v1
  -> native Lean checker
  -> metric_certificate/v1
  -> replay at the promotion boundary
```

No floating-point mean, fitted optimum, or dashboard summary is accepted as
proof input. Durations are nanoseconds, energy is microjoules, cost is
micro-USD, and counts are natural numbers. `CycleTelemetry` retains
`samples_ns` so callers can export observations rather than reverse-engineer
rounded summaries.

## Provenance and derivation

`metric_evidence/v1` binds:

- the existing `EvidenceBundleV1` bytes;
- the persisted OpenFeature snapshot bytes;
- the locked campaign manifest bytes, when present;
- candidate lever-snapshot digests;
- one hardware identity shared by compared arms;
- the declared cold/warm workload and every raw sample.

The checker derives cold/warm min-max intervals, the workload-weighted
expected-latency interval, exact rational means for input size, passes, energy
and cost, success rate, and parameter count. It selects by the committed
lexicographic policy: quality failures, success rate, parameters, latency upper
bound, passes, energy, cost, then candidate id.

Promotion does not trust the certificate's `verified` field alone. It invokes
the pinned Lean binary over the evidence and certificate, checks the selected
candidate, and independently binds the campaign-manifest digest. Missing
evidence, a missing binary, timeouts, digest drift, replay failure, or candidate
mismatch fail closed.

## Usage

```bash
python -m scripts.leverproof_metrics export \
  --run-id RUN \
  --evidence-bundle outputs/runs/RUN/evidence-bundle.json \
  --feature-flags outputs/runs/RUN/feature_flags.json \
  --campaign-manifest outputs/campaigns/CAMPAIGN/manifest.json \
  --cold-requests 1 --warm-requests 9 \
  --candidate-json outputs/runs/RUN/raw-resource-candidates.json \
  --out outputs/campaigns/CAMPAIGN/metric-evidence.json

python -m scripts.leverproof_metrics certify \
  --evidence outputs/campaigns/CAMPAIGN/metric-evidence.json \
  --certificate outputs/campaigns/CAMPAIGN/metric-certificate.json \
  --leverproof-bin ../leverproof-lean/.lake/build/bin/leverproof-lean
```

`register_promoted_checkpoint` looks for `metric-evidence.json` and
`metric-certificate.json` in the campaign artifact root by default. The
lineage `model_cycle promote` command requires explicit paths. Set
`LEVERPROOF_BIN` when the checker is not on `PATH`.

## Proof and trust boundary

Lean proves sample bounds, a nonzero mean denominator, declared-box model
selection, deterministic candidate membership, primary quality-failure
optimality, valid evidence, and checked-selection soundness. The standalone
audit reports standard logical axioms explicitly and rejects unfinished proof
placeholders.

Measurement truth, timers, JSON parsing, SHA-256, filesystem I/O, the Lean
runtime/compiler, and the operating system remain trusted. A certificate says
the checked result follows from the bound raw evidence; it does not assert that
a sensor was calibrated or that the experiment design was unbiased.

The v1 model language includes affine and polynomial expressions, declared-box
piecewise models, and integer inverse powers. A rational power fit must be
lowered to a certified piecewise polynomial envelope. Unsupported models are
rejected rather than approximated silently.
