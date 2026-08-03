# Autotrain c1822: compiler-decision token fixture positive

**Verdict:** queue `compiler-decision-token` for a fresh-seed confirmation. The
exact frozen c1821 checkpoints completed under the repaired evaluator, and the
size-matched candidate improves the declared structural primary, meaningful
program rate, component recall, reward, and p50 latency without regressing
parse, binder F1, or placeholder fidelity.

| Arm | Params | Smoke structure | MPR | Component recall | Binder F1 | Fidelity | Reward | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,608,962 | .052367 | 0 | .08333 | 1.0 | 1.0 | .941 | 14011.85 |
| compiler-decision token | 1,608,962 | .146233 | .6667 | .33333 | 1.0 | 1.0 | .945 | 12386.03 |

The primary improvement is `.093867`; MPR improves by `.6667`, component
recall by `.25`, reward by `.004`, and p50 latency by 11.6%. Both arms parse all
three documents, use the same 1,608,962 trainable parameters, and report no
decode timeout. This closes the c1821 infrastructure gap and shows that the
dense compiler-decision objective can affect meaningful OpenUI structure at
this seed.

The result is still fixture screening, not confirmation or ship evidence.
Both arms have AST and canonical equality `0`, the candidate structure remains
below `.35`, component recall remains below `.35`, and the suite has only three
documents instead of the required 20. Held-out, adversarial, OOD, and full Rico
were not run. The candidate therefore fails unchanged ship gates, remains
blocked from promotion and RL, and must reproduce on a fresh seed before its
protected promotion cadence can open.

The replay reused c1821's explicit no-sync scratch checkpoints: candidate SHA
`95c5846a...815b1` and control SHA `e02e53d3...9d029`. They are never reusable,
promotable, syncable, or shippable. Lean is
`not_applicable:retry_measurement`; no theorem or promotion claim is made.

Machine evidence:
[`autotrain-cycle-1822-compiler-decision-token-positive.json`](autotrain-cycle-1822-compiler-decision-token-positive.json).
