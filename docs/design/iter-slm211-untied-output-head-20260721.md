# SLM-211 output-head tying fixture (slm211-untied-output-head-20260721)

## Claim class
wiring

## Hypothesis
At matched starting function, an untied output head uses distinct storage and receives unambiguous optimizer updates, while a tied head shares storage exactly as before.

## Falsifier
Untied copy-init does not match tied initial logits, or optimizer groups contain duplicate tied storage, or spectral tooling cannot tell the modes apart.

## Settings
- tie_output_embedding: False
- tied_storage: False
- copy_init_logits_match: True
- optimizer_groups_unique: True
- spectral_tie_recorded: True
- n_trainable_parameters: 48642
- n_optimizer_group_params: 48642

## Honest caveats

- Fixture/wiring evidence only: no trained model, rare-action campaign, or GPU run.
- The H0-H3 matched experiment (capacity/exposure-matched rare-action debt/recall) is a follow-up requiring local E224+ checkpoints and the rare/focal weighting owner.

## Version stamp

```json
{
  "stamp_schema": "version_stamp/v1",
  "code_commit": "83d12c301de4d96eef7129340359940637b647af",
  "code_dirty": true,
  "components": {
    "harness.experiments": "v55",
    "harness.experiments.slm211_untied_output_head": "v1"
  },
  "stamped_at": "2026-07-21T03:11:54.334439+00:00"
}
```
