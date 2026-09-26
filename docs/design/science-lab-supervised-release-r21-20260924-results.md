# Supervised science-lab fixture remeasurement — R21

R21 completed a two-arm diagnostic through the ordinary autotrain supervisor,
including a fresh-process resume of the same locked driver cursor. The measured
result is a six-pair fixture-screening win; it is not a ship result or evidence
for promotion.

| Field | Value |
| --- | --- |
| Campaign / loop | `science-lab-supervised-release-r21-20260924` |
| Locked measurement plan | `science-lab-measurement-inputs-20260921/measurement_fixture.json`; SHA-256 `78a4cab366bd0b724325109538f166c433c7654a1172de22b748797ff3aafeda` |
| Retained evidence | `science-lab-current/new-trained-evidence.json`; SHA-256 `890c709e10db015fab2208abc6727931be8ab8229736ad85b83c4fd65b0e87a0` |
| Execution source | `/tmp/slm-production-lab-r19/execution`; source digest `65013ca8a8b497dd7d6eba2be05114130350a1426163f22a50a0668a9d097b3c` |
| Execution cwd / campaign store | `/tmp/slm-production-lab-r19/execution` / `/tmp/slm-production-lab-r21/outputs/science-lab-campaigns` |
| Recipe | CPU, scratch context backend, TwoTower, seed `7301`, three-step retained checkpoints, `65,826` trainable parameters per arm; R21 performed inference and evaluation only |
| Evaluation | Six decoded smoke cases and six paired masked-token denoising loss observations per arm; AgentV SDK bundles recorded for loss-suites and smoke ship-gate evaluation |
| Version stamp | `version_stamp/v1`, code commit `e56115002dd3f2872d2d24d1a5ef85ff36ce49cf`, `code_dirty=true`; see the JSON artifact for component versions |
| Honesty class | Fixture screening only; no promotion or shipment authority |

## Results

| Metric | Control (`lr=0.0003`) | Candidate (`lr=0.0006`) | Interpretation |
| --- | ---: | ---: | --- |
| Paired denoising loss | `24.23453` | `22.46921` | Candidate lower by `1.76532`; sign test `p=0.03125`, six non-tied pairs, locked `alpha=0.05` |
| Parse rate | `1.000` | `1.000` | Equal on six smoke cases |
| Meaningful-program rate | `0.167` | `0.167` | Equal; below ship bar |
| Structural similarity | `0.24445` | `0.24445` | Equal; below ship bar |
| Binder-reference F1 | `0.9444` | `1.0000` | Diagnostic secondary metric |
| Median latency | `670.0 ms` | `556.2 ms` | Fixture-only observation |

The paired loss is masked-token denoising cross-entropy under the locked loss
suite, not exact sequence-generation NLL. The paired test supports a screening
diagnostic only. Both arms have the same six declared root families and the same
checkpoint parameter count; no capacity difference is being credited.

## Supervisor and acceptance evidence

The supervisor first yielded after a bounded attempt. A fresh invocation
resumed the same campaign and driver cursor; both arms then exited successfully,
and closeout and inspection completed. The retained operation requests bind both
attempts to the same immutable execution source digest above. This demonstrates
supervisor replay and continuation for this bounded measurement.

The `--ship-gates` scoreboards correctly failed for both arms with **11
failures**: smoke volume was `6` (minimum `20`), smoke meaningful-program rate,
structural similarity, component recall, AST/canonical equivalence and reward
were below their bars, and `held_out`, `adversarial`, `ood`, and `rico_held`
were absent. There was no promotion. The serialized result records
`promotion_allowed=false` and `ship_eligible=false`.

R21 predates commit `b39d725ae627f6af2b621d631f95ed300997d03d`, which binds the
installed JavaScript runtime trees and package locks into merge-verification
identity. Therefore R21 is measurement and supervisor evidence for its recorded
source digest; it is **not** current-source acceptance for that later fix. The
separate merge-verification regression tests cover the new fingerprint.

## Artifacts

- Machine-readable result and version stamp: [JSON](science-lab-supervised-release-r21-20260924-results.json)
- Campaign: `/tmp/slm-production-lab-r21/outputs/science-lab-campaigns/science-lab-supervised-release-r21-20260924/`
- Supervisor attempts: `loops/science-lab-supervised-release-r21-20260924/runtime/attempts/`
- Per-arm AgentV JSONL and SDK bundles: `runs/{control,candidate}/agentv/`
- Checkpoints: retained from `science-lab-training-20260921`; R21 created or promoted none.
