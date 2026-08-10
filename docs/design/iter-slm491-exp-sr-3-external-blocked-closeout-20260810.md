# SLM-491: EXP-SR-3 external-blocked prepared-package closeout

**Status:** `external_blocked` / incomplete — not a construct-validity claim.

**Catalogue:** `exp-sr-3`

**Primary metric (`evaluator_human_agreement_kappa`):** `evaluator_human_agreement_kappa` = `None`

## Environment probe

- EXTERNAL_JUDGE_API_KEY set: `False`
- Real human raters available: `False`
- Live calibration ready: `False`

## Prepared package pointers

- SIE-005 frozen packet: `docs/design/iter-slm478-sie-005-blinded-packet-20260810.json`
- SIE-006 replay: `python -m scripts.run_sie006_construct_validity --mode external-blocked --seed 0`
- VCE-010 owner: `src/slm_training/evals/evaluator_calibration_protocol.py`

## SIE-006 external-blocked snapshot

- pair_n: `8`
- external_blocked: `True`
- evaluator_human_agreement_kappa: `None`

## Scope

SLM-491 closes EXP-SR-3 follow-up as an honest external-blocked prepared package: verifies the frozen SIE-005 pin and SIE-006 replay surface, but does not claim construct-validity pass/fail on real humans. Kappa and calibration metrics remain null — never mocked-as-real.

Command: `python -m scripts.run_slm491_external_blocked_closeout`

Command: `python -m scripts.run_slm491_external_blocked_closeout`
