# SLM-478 (SIE-005): Blinded construct-validity packet

**Claim class:** `fixture` only

**Status:** `fixture`

**Related catalogue:** `exp-sr-3` (SIE-006 executes)

## Acceptance snapshot

| Check | Value |
| --- | --- |
| pair_n | 8 |
| strata | ambiguity_case, hard_valid_contrast, metamorphic_pair, real_model_output |
| blinding_ok | True |
| training_excluded | True |
| independence_ok | True |
| import_complete_pair_n | 8 |
| mock_judge_modes | ok, refusal, cost_overrun |
| rubric_sha256 | `829adbc7adba9366…` |

## Owners reused

- `judge_audit.freeze_blinded_pairs` / `import_blinded_labels` (SLM-106)
- `judge_independence.ExternalJudgeConfig` / `ExternalJudgeAdapter`
- VCE-010 calibration protocol (feeds this packet; not reimplemented)

## Notes

- Packet prepared for EXP-SR-3 / SIE-006; no campaign locked here.
- Reuse SLM-106 judge_audit + judge_independence; VCE-010 owns calibration protocol.
- training_records_sha256_pin=d117d385afe9688ecf2bc41547d3081ee4c43832d277e2f41b049bf50044af0a
- conflicting duplicate labels rejected deterministically.

Command: `python -m scripts.run_sie005_blinded_packet --mode fixture`
