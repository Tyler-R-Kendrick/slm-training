# SLM-272 DCA1-01 — DCA activation gate (program-preference-activation-gate-v1)

Verdict: **blocked** (disposition: `inconclusive`)

DCA1-01 (SLM-272) activation requires: SLM-260 scorer oracle clean; SLM-268 selected verified data/semantic baseline or explicit floor; SLM-269 selected target policy; SLM-270 selected objective set; SLM-271 selected mask policy or no-effect disposition. Audited prerequisite artifacts are unqualified: SLM-261_loss_ledger=missing_or_unqualified, SLM-268_data_regime=blocked, SLM-269_target_policy=blocked_or_inconclusive. No training, surgery, curriculum, or tournament is launched; closing blocked/inconclusive.

## Prerequisite legs

- `SLM-261_loss_ledger`: `missing_or_unqualified`
- `SLM-268_data_regime`: `blocked`
- `SLM-269_target_policy`: `blocked_or_inconclusive`

## Successor conditions

- qualified LossLedgerV1 artifact (SLM-261 successor)
- SLM-268 successor with activation_status != blocked (selected regime or negative result)
- SLM-269 successor with a selected target policy or no-effect disposition
- then re-run scripts/audit_dca_activation --issue SLM-272 (reads artifacts live) and re-file the campaign

## Non-goals honored

No training, objective surgery, curriculum, tournament, checkpoint, or production default change. This disposition is itself the deliverable while the prerequisites are unqualified.
