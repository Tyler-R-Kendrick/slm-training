# EFS0-04 X22 audit-sample reproduction (2026-07-17)

SLM-106 requires the topology/tree-edit family in its frozen cross-family judge
sample. The historical X22 checkpoint and raw predictions were not retained in
the local artifact store, so this run reproduced the existing X22 recipe before
freezing any study pairs. Machine-readable matrix evidence is in
[`grammar-matrix-results-iter-efs0-04-x22-replay-20260717.json`](grammar-matrix-results-iter-efs0-04-x22-replay-20260717.json).

## Recipe and provenance

- Code/training/evaluation source: `7e2f5605e8fb9a874587f14fbe113aee904c9050`
- Model: X22 `tree_edit_diffusion`
- Data: committed `remediated_roots` train and `remediated` evaluation suites
- Device/backend: CPU, scratch context
- Training: seed 0, 80 steps, batch 4, learning rate 0.0003
- Checkpoint: local-only `last.pt`, SHA-256
  `a9cfb450e8146089cb26b6df84e90a5073627c4e59a2933d16f69034ec802ff6`
- Decode: 8 generation steps; RICO limited to 3 records
- Suites: smoke 3, held-out 5, adversarial 4, OOD 4, RICO-held 3
- Honesty: local fixture-grade reproduction only; canonical ship gates unchanged
- Checkpoint policy: scratch audit material, intentionally not synced or promoted

The run wrote a local checkpoint and all 19 raw prediction envelopes under
`outputs/runs/gx_x22_kapur_tree_edit_s0/`. It also emitted AgentEvals JSONL and
an AgentV SDK result bundle under that run's `agentv/` directory. Those ignored
runtime artifacts are inputs to the bounded SLM-106 capture step, not durable
ship evidence by themselves.

The durable JSON pins the AgentEvals JSONL digest
`d86d4ed6…e3ab03` and AgentV benchmark digest `f130b06b…7e621` so the
local bundle can be checked without treating its integrity score as a semantic
judge label.

## Result

| Suite | n | syntax parse | meaningful parse | structural similarity |
| --- | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.00 | 0.333 | 0.322 |
| held-out | 5 | 1.00 | 0.200 | 0.339 |
| adversarial | 4 | 1.00 | 0.000 | 0.301 |
| OOD | 4 | 1.00 | 0.000 | 0.259 |
| RICO-held | 3 | 1.00 | 0.667 | 0.183 |

The row fails the unchanged multi-suite ship gates. Syntax remains 1.0 by the
all-valid search construction; meaningful parse is the primary quality signal
and remains weak. The reproduction therefore restores raw X22 study material
without promoting the checkpoint or changing the historical X22 conclusion.

## EFS0-04 use and caveats

The 19 complete predictions may enter the blinded cross-family audit only after
their checkpoint and envelope hashes are captured in the frozen manifest. They
must not be interpreted as independent-judge or human evidence. External and
human decisions remain unavailable until genuinely independent participants run
the blinded package.
