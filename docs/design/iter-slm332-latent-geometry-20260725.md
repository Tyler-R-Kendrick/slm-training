# SLM-332 local latent-geometry diagnostic

This bounded local run compares a reconstruction-only control with the combined
reconstruction, hard-valid binding contrast, and local-noise objective. It is
**not a ship result**.

| arm | contrast distance | reconstruction MSE | promotion |
| --- | ---: | ---: | --- |
| control | 0.019898 | 0.004581 | no |
| combined | 5.996696 | 0.005570 | no |

- Local CPU; 366 admitted AP-009 binding/reference pairs; seed 332; 40 steps.
- Combined minus control contrast distance: 5.976798.
- AP-030's factor proxy is not a semantic oracle. There is no verified
  latent-to-program decoder, so no interpolation traversal or strict semantic
  claim was produced and the result is fail-closed/non-promotable.
- AgentEvals/AgentV records the non-promotion assertion alongside the JSON.
