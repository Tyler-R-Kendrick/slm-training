# E237 — detached binder-topology context

Date: 2026-07-16
Status: completed; no-op hypothesis rejected; checkpoint not promotable or ship

E237 tests whether E236's topology loss damaged shared prompt features. The
only implementation change detaches the pooled context before the auxiliary
binder-topology head. The grammar-derived targets, compiler-legal candidate
restriction, decoder, data, seed, and all weights remain matched.

Immediately before training, the isolated branch fetched and rebased onto
`origin/main` at `e9fea69`, was zero commits behind, and was clean. The
changed-file hook passed 189 tests with 15 deselected; Ruff and compile checks
also passed.

The matched 126-row E230 corpus, CPU, 32-step, batch-4, learning-rate 0.0003,
seed-0 frozen SmolLM2 recipe took 142.67 s. Trace:
`2bfa5b6dc890012a1496b2cb2bd89a8b`; checkpoint SHA:
`edcbad06be36962d51c81e0ae5af913b9049f0f67f24bfdb9338446cd8c4b59d`.

The topology head still did not learn: sampled-batch loss rose 1.1498 → 1.3684
and accuracy moved 0.5455 → 0.5238, with 11 → 21 rows and 5.0 → 7.7619 mean
legal candidates. Final total loss was 28.3716.

All five strict suite aggregates exactly reproduce E236: syntax is 1.0, but
meaningful program, component recall, fidelity, and reward are 0 throughout.
Structure is 0.3094/0.2514/0.2905/0.2369/0.0901 for
smoke/held-out/adversarial/OOD/RICO. Twelve thresholds fail and AgentV is 0/5.
The topology head is applied 38 times and changes zero choices. Evaluation
trace: `33c3ffd3a2d3f9a50a4cc1a669aee13b`.

The decode-weight-zero ablation is again identical; trace:
`565eeff271ea13d174873189e576fe4a`.

Because the HF context tower is frozen, its pooled features already carry no
trainable gradient path; detaching them is a no-op in this recipe. Retain the
defensive detach for future unfrozen-context experiments, but reject the
hypothesis and checkpoint. E236's collapse cannot be attributed to context-head
gradient interference. Do not spend another run on topology calibration until
the target is reformulated to include grammar-derived reference arity/stop
decisions and a stable learning diagnostic is demonstrated.

Machine-readable evidence:
[iter-e237-detached-topology-20260716.json](iter-e237-detached-topology-20260716.json).
