---
name: improve-lean-optimums
description: Diagnose and improve Lean4-calculated metric ranges when an autotraining, evaluation, benchmark, or promotion observation is outside its preregistered band. Use for metric_evidence/v2 and metric_certificate/v2 work, LeverProof theorem or assumption changes, out-of-band cycle feedback, or the required five-lane successor experiment matrix.
---

# Improve Lean Optimums

Treat a certified band miss as information about the research system. Never turn
it into a model loss, silently widen a bound, weaken a gate, or let an agent edit
code automatically from one observation.

## Workflow

1. Read `docs/design/leverproof-integration.md`, the locked
   `ExperimentCampaignV1`, its expectation manifest, raw observations, evidence,
   and certificate. Confirm all SHA-256 bindings before interpreting numbers.
2. Replay the certificate with the in-repo checker:

   ```bash
   make -C src/leverproof_lean test
   python -m scripts.leverproof_metrics verify \
     --evidence <metric-evidence.json> \
     --certificate <metric-certificate.json>
   ```

3. Apply the tiered disposition exactly:
   - `continue`: all observations are in band.
   - `stop`: a theorem-backed range is contradicted. Stop this campaign; repair
     measurement or the formal model before further training.
   - `block_promotion_and_diagnose`: preserve the completed run as evidence,
     block promotion, and preregister a successor matrix.
   - `historical_only`: v1 evidence remains replayable but cannot authorize a
     new promotion.
4. For every miss, cover all five diagnosis lanes with controlled hypotheses:
   `measurement_control`, `training_method`, `architecture`, `lean_model`, and
   `assumptions`. Do not assign causality from the miss alone.
5. Change only the lane supported by new evidence. If the Lean model or an
   assumption changes, create a new expectation manifest and bind its digest in
   the next campaign before outcomes are visible. Preserve old artifacts.
6. Run the smallest applicable checks and update measured-results docs for any
   train, eval, bench, profile, telemetry, matrix, or reproduction run.

## Lean discipline

- Keep generic interval arithmetic and metric programs in
  `src/leverproof_lean/`; Python only handles files, hashes, process execution,
  and typed cycle policy.
- Prove reusable claims. Do not use `sorry`, `admit`, `axiom`, `native_decide`,
  or post-hoc constants derived from the observations under evaluation.
- Raw observations remain outside Lean's trusted measurement boundary. Lean
  proves the declared calculation and classification, not sensor truth.
- Preserve v1 replay compatibility while requiring v2 bands for new promotion.
- Respect the repository run cap for every command.

## Completion checks

Run at least:

```bash
make -C src/leverproof_lean test
python -m scripts.verify_agent_surfaces
python -m scripts.verify_version_stamps --check
python -m scripts.repo_policy
```

Also run focused Python tests for `verified_metrics`, autoresearch feedback, and
promotion. If an experiment ran, follow `documenting-experiment-results`; if a
training-data build ran, follow `synthesis-feedback`.
