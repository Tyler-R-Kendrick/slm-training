# E561 — midpoint rare slot-owner coverage

E561 keeps twofold exposure and moves the owner eligibility ceiling from four
to seven labels. Ten classes select 42 of 244 records, expanding the sampling
pool to 286.

The clean run processed 1,284 target tokens in 41.47 seconds under
`max_wall_minutes=3` and wrote checkpoint SHA
`35a4fe6dd1b0eb2f59c33cb6d4ae11472c693f43a15fa3e6abc46db323a127f9`.

Against E555 on matched OOD `n=4`, E561 improves every non-semantic headline:
fidelity 0.3000→0.5750, structure 0.1594→0.2419, component recall
0.1250→0.1458, reward 0.5453→0.5753, AST-node F1 0.2389→0.3125, and AST-edge
F1 0→0.0385. Meaning-v1/v2 remain 0 and AgentV remains 0/1.

**Verdict:** retain threshold 7 at twofold exposure and close the sampling
ladder. The checkpoint is not promotable; use it only for the next semantic
decode investigation. Evidence:
[JSON](iter-e561-owner-threshold7-20260720.json).
