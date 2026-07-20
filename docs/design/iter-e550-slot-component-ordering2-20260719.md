# E550 — learned slot-component midpoint

E550 holds the E547 checkpoint and OOD `n=4` recipe fixed, changing learned
slot-component decode weight from 4 to 2. It completed on CPU under the
170-second cap and creates no checkpoint.

Weight 2 produces the same predictions, metrics, and 28 learned interventions
as weight 4: fidelity 0.2583, structure 0.2248, component recall 0.2083, reward
0.5403, AST node F1 0.3270, meaningful-v1 0.0, strict-v2 0.0, and AgentV 0/1.

**Verdict:** close scalar tuning. Weight 0 collapses semantic density, while
tested positive weights 2 and 4 have identical ordering. Next address learned
supervision, calibration, or candidate ordering directly. Evidence:
[JSON](iter-e550-slot-component-ordering2-20260719.json).
