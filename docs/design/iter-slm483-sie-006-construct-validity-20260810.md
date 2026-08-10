# SLM-483 (SIE-006): Construct-validity + human calibration (EXP-SR-3)

**Claim class:** `fixture` execution on catalogue `diagnostic`

**Status:** `external_blocked` / `external_blocked`

**Catalogue:** `exp-sr-3`

**Frozen SIE-005 pin:** `docs/design/iter-slm478-sie-005-blinded-packet-20260810.json`

## Acceptance snapshot

| Check | Value |
| --- | --- |
| pair_n | 8 |
| training_excluded | True |
| audit_holdout | True |
| external_blocked | True |
| evaluator_human_agreement_kappa | None |
| promotion | False |

## Blocked claims

- `evaluator_human_agreement_kappa`
- `external_judge_calibration`
- `blinded_human_adjudication`
- `construct_validity_promotion`

## Qualified claims

- `evaluator_human_agreement_kappa`: external_blocked
- `external_judge_calibration`: external_blocked
- `blinded_human_adjudication`: external_blocked
- `construct_validity_promotion`: external_blocked

## Scope

External-blocked EXP-SR-3 run: verifies the frozen SIE-005 packet pin and locks the campaign manifest, but does not fabricate human ratings or score primary construct-validity claims. Audit sample remains training-excluded. Fixture plumbing (mock transport + synthetic labels) is validated separately via `python -m scripts.run_sie006_construct_validity --mode fixture`.

Command: `python -m scripts.run_sie006_construct_validity --mode external-blocked --seed 0`
