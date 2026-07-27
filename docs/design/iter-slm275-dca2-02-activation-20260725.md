# SLM-275 DCA2-02 — DCA activation gate (diffusion-rl-activation-v1)

Verdict: **blocked** (disposition: `inconclusive`)

DCA2-02 (SLM-275) activation requires: SLM-260 scorer oracle clean; SLM-272 offline program preference disposition; SLM-274 selected diffusion SFT parent and rollout implementation; plus a nonzero semantic baseline with usable within-group reward variance. Otherwise close reward_variance_insufficient without training. Audited prerequisite artifacts are unqualified: SLM-261_loss_ledger=missing_or_unqualified, SLM-268_data_regime=blocked, SLM-269_target_policy=blocked_or_inconclusive. No training, surgery, curriculum, or tournament is launched; closing blocked/inconclusive.

## Prerequisite legs

- `SLM-261_loss_ledger`: `missing_or_unqualified`
- `SLM-268_data_regime`: `blocked`
- `SLM-269_target_policy`: `blocked_or_inconclusive`

## Successor conditions

- qualified LossLedgerV1 artifact (SLM-261 successor)
- SLM-268 successor with activation_status != blocked (selected regime or negative result)
- SLM-269 successor with a selected target policy or no-effect disposition
- then re-run scripts/audit_dca_activation --issue SLM-275 (reads artifacts live) and re-file the campaign

## Non-goals honored

No training, objective surgery, curriculum, tournament, checkpoint, or production default change. This disposition is itself the deliverable while the prerequisites are unqualified.
