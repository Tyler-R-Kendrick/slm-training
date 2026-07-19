# CAP5 claim ledger

Machine-readable source: [cap5-claim-ledger.json](cap5-claim-ledger.json).

| Claim | Hypothesis | Status | Evidence kind | Primary artifacts | Transfer limits |
| --- | --- | --- | --- | --- | --- |
| CAP0-02-001 | CAP0-02 | supported | exact_local | docs/design/cap0-02-arity-analyzer-20260718.json, tests/test_dsl/test_arity_analysis.py | Does not transfer to unbounded grammar, different template classes, or later parser/codec versions without re-enumeration. |
| CAP0-03-001 | CAP0-03 | supported | exact_local | docs/design/cap0-03-coding-precision-20260718.md, tests/test_dsl/test_arity_coding.py | Not a general bound solver; each construction must be re-verified for new parameters. |
| CAP1-03-001 | CAP1-03 | supported | exact_local | docs/design/cap1-03-task-quotient-20260718.md, tests/test_dsl/test_task_quotient.py | Quotient is not M_epsilon without declared distortion and traffic distribution. |
| CAP2-04-001 | CAP2-04 | supported | estimated | docs/design/iter-cap2-04-state-ablation-20260718.md, docs/design/iter-cap2-04-state-ablation-20260718.json | Does not imply compiler-owned state alone is sufficient for ship quality. |
| CAP3-03-001 | CAP3-03 | falsified | estimated | docs/design/iter-cap3-03-ternary-falsification-20260718.md, docs/design/iter-cap3-03-ternary-falsification-20260718.json | Does not rule out ternary with larger budgets, optimized kernels, or different state signatures. |
| CAP4-03-001 | CAP4-03 | diagnostic | estimated | docs/design/iter-cap4-03-quantized-energy-inference-20260719.md, docs/design/iter-cap4-03-quantized-energy-inference-20260719.json | Not a deployment recommendation; exact inference is infeasible at full scale. |
| CAP5-01-001 | CAP5-01 | diagnostic | estimated | docs/design/calculated-arity-adaptive-precision-results.md, docs/design/quality-experiment-matrix.md | Not a universal law; extrapolation outside declared families is unsupported. |
