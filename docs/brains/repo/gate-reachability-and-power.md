# Gate reachability and prospective power

Status: active measurement boundary

Linear: [SLM-286](https://linear.app/quickdeploy-ai/issue/SLM-286)

## Thesis

A failed ship verdict is not automatically a model-quality result. Gate
reachability must be decided first: insufficient suite volume makes the
quality claim inconclusive, while missing telemetry makes it invalid or
confounded. Only a sufficiently measured threshold failure can support a
negative model-quality conclusion.

Every canonical binomial rate therefore carries its exact numerator,
denominator, seed count, Wilson interval, and evidence class. Bounded means
such as structural similarity are not relabeled as binomial rates.

## Power boundary

Power belongs in preregistration. The target delta, alpha, target power, sample
size, and seeds are frozen before confirmation data is read. Seeds are reported
separately rather than multiplied into an artificial sample size. Observed or
post-hoc power is not success evidence.

## Repository evidence

- [`slm286-ship-gate-evidence-census-20260723.md`](../../design/slm286-ship-gate-evidence-census-20260723.md)
- `src/slm_training/evals/power_protocol.py`
- `src/slm_training/harnesses/model_build/evidence_census.py`

The canonical-reader census found 869 of 904 suite rows below the current
minimum before quality was considered. Historical records remain immutable;
the census preserves its verified ledger prefix and appends hash-bound
superseding adjudications.

## Sources

- Wilson, *Probable Inference, the Law of Succession, and Statistical
  Inference* (1927), DOI
  [10.1080/01621459.1927.10502953](https://doi.org/10.1080/01621459.1927.10502953).
- Hoenig and Heisey, *The Abuse of Power* (2001), DOI
  [10.1198/000313001300339897](https://doi.org/10.1198/000313001300339897).

These are adapted measurement practices, not a claim that interval overlap or
non-overlap is a significance test.
