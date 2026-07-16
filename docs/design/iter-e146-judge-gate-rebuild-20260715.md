# E146 — Judge-gate rebuild verification (2026-07-15)

## Question

Does the independent prompt/output judge actually reach the persisted verification metadata used by training-data admission?

## Rebuild

The existing 405-record `remediated_roots_judged` corpus was normalized into a temporary `judged_gate` snapshot with synthesis disabled. The rebuild used the repository’s normal record normalization, quality filtering, and verification stamping path.

## Results

| Metric | Result |
| --- | ---: |
| input records | 405 |
| collected records | 530 |
| persisted records | 498 |
| quality rejected | 0 |
| normalization errors | 0 |
| G11 pass | 498 |
| G11 fail | 0 |
| G11 skipped | 0 |
| mean quality score | 0.9661 |

Before the harness fix, records carried a nested quality judge result while verification G11 remained `skip` because the result was not copied into `VerificationContext`. E146 confirms the fix at the artifact boundary: all persisted rows now carry an authoritative G11 pass.

This does not prove semantic prompt/output quality. The deterministic judge remains the current gate; a stronger semantic judge can be added as a separate measured experiment rather than silently conflating it with G11.
