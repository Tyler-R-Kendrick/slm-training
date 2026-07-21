# SLM-238 (RSC-A02): depth-aux-mode factorial (rsc_a02_depth_aux_mode_factorial)

Matrix set: `rsc-a02-depth-aux-mode-factorial`  
Version: `rsc-a02-v1`  
Status: **calibration_only**

**No quality or LOTUS-transfer claim is made from this factorial** -- it is calibration/semantics work only, ahead of a future SLM-233 control-matrix campaign.

Preregistered lambda: `0.3`

0.3 (< 1) is chosen a priori, not tuned post-hoc: the final recursion depth already receives full weight once via the primary reconstruction term (rec_out['logits'] == rec_out['depth_logits'][-1]), so a partial (0.3x) extra credit in all_depths mode avoids letting the auxiliary term dominate or swamp the primary gradient signal, while remaining large enough to matter. It is also a simple, easy-to-reproduce round number for a first calibration pass.

## Arm A: mode=`off` aux_weight=`0.0` schedule=`n/a`

Deterministic fixture (final step):

- primary_final_reconstruction_loss: `23.041454`
- recursive_intermediate_aux_loss: `0.000000`
- recursive_final_depth_aux_contribution: `0.000000`
- recursive_depth_aux_weight: `0.0`
- recursive_depth_supervision_loss: `0.000000`
- combined_training_loss: `23.041454`
- first/last-depth contribution ratio: `n/a (aux disabled -- no per-depth contributions)`
- final update grad norm: `24.354876`

Per-depth gradient diagnostics (synthetic isolated tower batch):

- depth 0: grad_norm=`2.257631` cosine_vs_final_depth=`0.9730`
- depth 1: grad_norm=`2.210759` cosine_vs_final_depth=`0.9962`
- depth 2: grad_norm=`2.171908` cosine_vs_final_depth=`1.0000`

Bounded safety smoke (bridge_healthy=False): openui grammar bridge unavailable in this environment (e.g. a NODE_OPTIONS incompatibility) -- syntax/structure/semantic smoke skipped/vacuous here, not because of this issue's changes; rerun in a bridge-healthy environment for a real read

Bounded real-corpus smoke (final step):

- combined_training_loss: `21.394547`
- final update grad norm: `19.412747`

## Arm B: mode=`intermediate_only` aux_weight=`1.0` schedule=`uniform_normalized`

Deterministic fixture (final step):

- primary_final_reconstruction_loss: `23.048304`
- recursive_intermediate_aux_loss: `26.997311`
- recursive_final_depth_aux_contribution: `0.000000`
- recursive_depth_aux_weight: `1.0`
- recursive_depth_supervision_loss: `26.997311`
- combined_training_loss: `50.045614`
- first/last-depth contribution ratio: `n/a (final depth structurally excluded -- intermediate_only)`
- final update grad norm: `44.723161`

Per-depth gradient diagnostics (synthetic isolated tower batch):

- depth 0: grad_norm=`2.257631` cosine_vs_final_depth=`0.9730`
- depth 1: grad_norm=`2.210759` cosine_vs_final_depth=`0.9962`
- depth 2: grad_norm=`2.171908` cosine_vs_final_depth=`1.0000`

Bounded safety smoke (bridge_healthy=False): openui grammar bridge unavailable in this environment (e.g. a NODE_OPTIONS incompatibility) -- syntax/structure/semantic smoke skipped/vacuous here, not because of this issue's changes; rerun in a bridge-healthy environment for a real read

Bounded real-corpus smoke (final step):

- combined_training_loss: `45.966850`
- final update grad norm: `35.562350`

## Arm C: mode=`all_depths` aux_weight=`1.0` schedule=`uniform_normalized`

Deterministic fixture (final step):

- primary_final_reconstruction_loss: `23.044403`
- recursive_intermediate_aux_loss: `17.999525`
- recursive_final_depth_aux_contribution: `7.681468`
- recursive_depth_aux_weight: `1.0`
- recursive_depth_supervision_loss: `25.680992`
- combined_training_loss: `48.725395`
- first/last-depth contribution ratio: `1.2522495522125672`
- final update grad norm: `45.991463`

Per-depth gradient diagnostics (synthetic isolated tower batch):

- depth 0: grad_norm=`2.257631` cosine_vs_final_depth=`0.9730`
- depth 1: grad_norm=`2.210759` cosine_vs_final_depth=`0.9962`
- depth 2: grad_norm=`2.171908` cosine_vs_final_depth=`1.0000`

Bounded safety smoke (bridge_healthy=False): openui grammar bridge unavailable in this environment (e.g. a NODE_OPTIONS incompatibility) -- syntax/structure/semantic smoke skipped/vacuous here, not because of this issue's changes; rerun in a bridge-healthy environment for a real read

Bounded real-corpus smoke (final step):

- combined_training_loss: `44.907421`
- final update grad norm: `36.598845`

## Arm D: mode=`intermediate_only` aux_weight=`0.3` schedule=`uniform_normalized`

Deterministic fixture (final step):

- primary_final_reconstruction_loss: `23.043634`
- recursive_intermediate_aux_loss: `8.100473`
- recursive_final_depth_aux_contribution: `0.000000`
- recursive_depth_aux_weight: `0.3`
- recursive_depth_supervision_loss: `8.100473`
- combined_training_loss: `31.144108`
- first/last-depth contribution ratio: `n/a (final depth structurally excluded -- intermediate_only)`
- final update grad norm: `30.414565`

Per-depth gradient diagnostics (synthetic isolated tower batch):

- depth 0: grad_norm=`2.257631` cosine_vs_final_depth=`0.9730`
- depth 1: grad_norm=`2.210759` cosine_vs_final_depth=`0.9962`
- depth 2: grad_norm=`2.171908` cosine_vs_final_depth=`1.0000`

Bounded safety smoke (bridge_healthy=False): openui grammar bridge unavailable in this environment (e.g. a NODE_OPTIONS incompatibility) -- syntax/structure/semantic smoke skipped/vacuous here, not because of this issue's changes; rerun in a bridge-healthy environment for a real read

Bounded real-corpus smoke (final step):

- combined_training_loss: `28.766763`
- final update grad norm: `24.219577`

## Arm E: mode=`all_depths` aux_weight=`0.3` schedule=`uniform_normalized`

Deterministic fixture (final step):

- primary_final_reconstruction_loss: `23.043045`
- recursive_intermediate_aux_loss: `5.400579`
- recursive_final_depth_aux_contribution: `2.304305`
- recursive_depth_aux_weight: `0.3`
- recursive_depth_supervision_loss: `7.704884`
- combined_training_loss: `30.747929`
- first/last-depth contribution ratio: `1.2525688690321286`
- final update grad norm: `30.821360`

Per-depth gradient diagnostics (synthetic isolated tower batch):

- depth 0: grad_norm=`2.257631` cosine_vs_final_depth=`0.9730`
- depth 1: grad_norm=`2.210759` cosine_vs_final_depth=`0.9962`
- depth 2: grad_norm=`2.171908` cosine_vs_final_depth=`1.0000`

Bounded safety smoke (bridge_healthy=False): openui grammar bridge unavailable in this environment (e.g. a NODE_OPTIONS incompatibility) -- syntax/structure/semantic smoke skipped/vacuous here, not because of this issue's changes; rerun in a bridge-healthy environment for a real read

Bounded real-corpus smoke (final step):

- combined_training_loss: `28.449054`
- final update grad norm: `24.551611`

## Caveat

Bounded calibration/semantics factorial only (deterministic 2-record fixture + a 6-record real-corpus smoke, 3 training steps each). No promotion or broad GPU campaign; no quality or LOTUS-transfer claim is made. See docs/design/iter-rsc-a02-*.md for the recommended semantic mode and the SLM-233 control-matrix recommendation.
