# LeverProof integration

The in-repo [`src/leverproof_lean/`](../../src/leverproof_lean/) project is the
executable oracle for resource metrics and preregistered calculated ranges.
Python owns files, hashes, process execution, and typed cycle policy. Lean owns
exact arithmetic, selection, band calculation, and observation classification.

```text
locked campaign + expectation digest + raw integer observations
  -> metric_evidence/v2
  -> src/leverproof_lean/.lake/build/bin/leverproof-lean
  -> metric_certificate/v2
  -> typed OptimumFeedbackV1
  -> continue | stop | block promotion and diagnose
```

Version 1 remains replayable for historical evidence, but cannot authorize a
new checkpoint promotion. New promotions require v2 and a campaign-locked
`metric_expectations_sha256`.

## Generic calculated ranges

Each expectation names a metric and unit, an authority
(`theorem` or `assumption_backed`), named rational dependency intervals, and a
small reverse-Polish program. Supported instructions are `constant`,
`variable`, `add`, `multiply`, `power`, and `inverse`. The checker rejects
unknown variables, malformed stacks, zero-denominator inverses, invalid
intervals, duplicate metrics, and empty observations.

Observed values remain raw natural numbers. Lean emits the calculated interval,
observed min/max interval, below/within/above counts, and an `in_band`, `below`,
`above`, or `mixed` relation. This protocol is metric-generic: latency, memory,
energy, token counts, loss-scaled integers, or another preregistered quantity
use the same structure.

No floating-point summary is proof input. The expectation manifest digest is
locked before outcomes are visible and is embedded in evidence and the
certificate. Bounds are never widened after a miss.

## Cycle policy

An out-of-band metric is a cycle-level research signal, not an auxiliary loss:

- all metrics in band: continue;
- theorem-backed miss: stop the campaign;
- assumption-backed miss: keep the terminal run as evidence, block promotion,
  and require a successor hypothesis matrix;
- v1 certificate: historical replay only.

A miss does not identify its own cause. The successor matrix must cover all
five controlled diagnosis lanes: measurement/control, training method,
architecture, Lean model, and assumptions/dependent variables. Automatic
source rewriting, gate weakening, and post-hoc range edits are forbidden.

## Usage

```bash
make -C src/leverproof_lean test

python -m scripts.leverproof_metrics export \
  --run-id RUN \
  --evidence-bundle outputs/runs/RUN/evidence-bundle.json \
  --feature-flags outputs/runs/RUN/feature_flags.json \
  --campaign-manifest outputs/campaigns/CAMPAIGN/manifest.json \
  --cold-requests 1 --warm-requests 9 \
  --candidate-json outputs/runs/RUN/raw-resource-candidates.json \
  --expectations outputs/campaigns/CAMPAIGN/metric-expectations.json \
  --observations outputs/runs/RUN/metric-observations.json \
  --out outputs/campaigns/CAMPAIGN/metric-evidence.json

python -m scripts.leverproof_metrics certify \
  --evidence outputs/campaigns/CAMPAIGN/metric-evidence.json \
  --certificate outputs/campaigns/CAMPAIGN/metric-certificate.json
```

The CLI and promotion paths default to the pinned in-repo executable; there is
no environment or `PATH` fallback. An explicit binary is retained only for
controlled tests. Missing evidence, digest drift, replay failure, timeout,
candidate mismatch, v1 evidence at a new promotion, or any band breach fails
promotion closed.

## Trust boundary

Lean proves the declared calculation and classification. Measurement truth,
sensor calibration, experiment design, JSON decoding, SHA-256, filesystem I/O,
the Lean runtime/compiler, and the operating system remain trusted. Therefore
the diagnosis matrix always includes a measurement-control lane.

The candidate selector still derives exact rational resource summaries and
replays its committed lexicographic policy: quality failures, success rate,
parameters, latency upper bound, passes, energy, cost, then candidate id.
Current global theorems cover membership and the primary quality-failure
optimum; the remaining tie-breaks are executable replayed definitions and are
not overstated as fully proved.
