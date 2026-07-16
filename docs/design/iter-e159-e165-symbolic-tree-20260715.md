# E159–E165 symbolic-tree enforcement (2026-07-16)

This follow-up tests the deterministic layer independently of training duration.
The same three-record smoke subset was used with local HF context and one
decode attempt. E159 is an existing lexer-native checkpoint control; E160 is a
new 32-step lexer-native train. E161–E165 progressively exposed and repaired
the compiler path.

| Run | Output alphabet / path | Parse | Raw syntax | Structure | Key evidence |
| --- | --- | ---: | ---: | ---: | --- |
| E159 | existing lexer control | 0.3333 | 0.6667 | 0.3417 | deterministic layer can produce valid output |
| E160 | new lexer train, ordinary LTR | 0.0 | 0.3333 | 0.1256 | lexer is necessary but model still weak |
| E161–E163 | tree path before final fixes | 0.0 | 0.3333 | 0.1256–0.1789 | BOS root and contract/path fallback bugs found |
| E165 | tree path, certified-prefix only | 0.0 | 0.3333 | 0.1789 | compiler candidates 938; seeded fallbacks 0; p50 1694 ms |

Code fixes:

- BOS-only prefixes now count as statement start, preserving `<BIND_0>`/root.
- Partial forests with valid paths are consumed instead of discarded.
- Tree/restricted decode stops at a certified prefix when no next path exists;
  it no longer appends unconstrained MaskGIT output.
- Legacy LTR repair is skipped when compiler decode owns the constrained path.
- Evaluation policy persists `compiler_decode_mode`.

E165 still fails parse and ship gates. That is now an honest result from the
symbolic path: no seeded unconstrained fallback occurred. The next research
lever is model-choice/semantic completion quality inside the certified tree,
not more training steps or a relaxed parse gate.

Evidence: [result JSON](iter-e159-e165-symbolic-tree-20260715.json) and the
raw AgentEvals bundles under the run paths listed there.
