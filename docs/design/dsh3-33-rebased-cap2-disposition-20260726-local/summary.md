# DSH3-33 rebased CAP2 operator-policy disposition

Status: **DSH5 closed** — no learned operator-policy claim is supported.

| Claim | Verdict | Evidence |
| --- | --- | --- |
| `runtime_correctness` | `fixture_only` | SLM-381.cap2-v2-fixture |
| `model_view_hygiene` | `fixture_only` | SLM-403.cap2-v2 |
| `training_validity` | `negative` | SLM-403.cap2-v2 |
| `complete_partial_safety` | `negative` | SLM-404.coverage |
| `stop_calibration` | `unrun_conditional` | none |
| `permutation_invariance` | `fixture_only` | SLM-402.permutation |
| `held_out_semantics` | `negative` | SLM-403.cap2-v2, SLM-405.negative-ablation |
| `systems_efficiency` | `unavailable` | SLM-407.systems |
| `cap0_retention` | `unrun_conditional` | none |
| `cap1_retention` | `unavailable` | none |

The frozen DSH3-17 disposition remains historical and unchanged. The local evidence retains the typed-policy negative stop rule, the unavailable systems comparison, and fixture-only runtime/permutation evidence. No checkpoint, model-card update, remote run, human-rating gate, production change, or DSH5 authorization follows.
