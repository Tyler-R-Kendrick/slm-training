# SLM-261: bounded memorization probe (slm261-79f2e77b)

- **Matrix set:** slm261_memorization_probe
- **Matrix version:** vsd0-02-v1
- **Status:** fixture
- **Claim class:** wiring
- **Disposition:** inconclusive
- **Timestamp:** 2026-07-21T10:12:48.792805Z

## Hypothesis
A correctly wired TwoTower trainer can memorize a small verified corpus: principal masked CE falls, exact target accuracy rises, and every active loss term reconciles with the reported total.

## Falsifier
Principal loss cannot fall below 0.10 nats/token, exact target accuracy cannot exceed 0.99, canonical reconstruction cannot exceed 0.98, or the loss ledger fails reconciliation.

## Arms

| arm | seed | steps | final loss | ledger error | raw NLL | legal NLL | exact acc | recon rate | wall s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0_principal_only | 0 | 3 | 21.5996 | 0.0000 | 21.9668 | 7.5223 | 0.9515 | n/a | 2.51 |
| M1_current_recipe | 0 | 3 | 41.0030 | 0.0000 | 21.9933 | 7.5043 | 0.9515 | n/a | 1.78 |

## Disposition rationale
No arm reached the strict fixture memorization thresholds (exact target accuracy >= 0.99 and ledger reconciliation error < 1e-3). This is expected for a tiny diagnostic fixture; it is not a falsification.

## Honest caveats
- Fixture-only diagnostic: tiny model, tiny corpus, CPU run, no ship claim.
- VSD0-01 semantic scorer prerequisite is not enforced by this fixture.
- Candidate-normalized CE (M2) is not implemented in this iteration.
- Exact canonical reconstruction is measured by string match, not the full binding-aware meaning pipeline.

## Reproducibility
```bash
python -m scripts.run_memorization_probe --corpus <path> --output-dir outputs/experiments/slm261-79f2e77b
```