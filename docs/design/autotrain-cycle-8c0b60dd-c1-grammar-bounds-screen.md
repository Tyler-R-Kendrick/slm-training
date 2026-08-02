# Autotrain local-loop c1: grammar_completion_bounds screen

**Verdict:** reject `grammar_completion_bounds` as a standalone lever. On a
freshly seeded, exactly size-matched fixture arm, the bounds candidate cuts
p50 latency 4.08% and trains 18.36% faster wall time, but every measured
quality signal ties the control exactly (parse, meaning, structure, recall,
AST/canonical BEq, reward). The primary metric (`smoke.structural_similarity`)
shows zero improvement, so this is a null-delta screen, not a win.

| Arm | Params | Steps | n / complete / timeout | Parse | Meaning | Structure | Recall | p50 | Loss / train wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,608,962 | 21 | 3 / 3 / 0 | 1 | 0 | .05750 | 0 | 1,846.50 ms | 22.6219 / 4.974 s |
| grammar_completion_bounds | 1,608,962 | 21 | 3 / 3 / 0 | 1 | 0 | .05750 | 0 | 1,771.30 ms | 22.6219 / 4.060 s |

Both CPU scratch arms used seed 100001, batch size 2, `wf_smoke_v2` fixture
data, and strict compiler-tree constrained evaluation
(`slot_contract_constrained_decode=True`). This is cycle 1 of a fresh local
continuous-loop identity (`continuous-openui-local`): the repo's
prior `continuous-openui-202607-98199209` lineage (through c1759) has no
recoverable local state in this ephemeral checkout — `outputs/` is
gitignored and its champion queue / loop state do not survive a container
recycle — so this session starts a new, honestly-labeled local loop rather
than fabricating continuity with lost state.

AgentV completed both arms without execution errors (`@agentv/core`,
previously blocked by a missing `npm ci` in this checkout — installed as
part of this cycle to unblock the eval stage). `--ship-gates` fails at
fixture `n=3` on meaning, structure, recall, AST BEq, canonical BEq, and
reward; held-out, adversarial, OOD, and `rico_held` were not run. Lean is
`not_applicable:screening` because no promotion claim is made. Neither
checkpoint is synced, reusable, served, promoted, or ship evidence.

Next: treat the `grammar_completion_bounds` fingerprint as exhausted at this
scale (tied quality, marginal latency-only gain). Prioritize the
ranked successor — a distinct, size-matched "component-plan" quality
hypothesis — per the driver's `cycle_handoff.json` priorities; keep the
matched control fixed every cycle to prevent recipe drift.

Machine evidence:
[`autotrain-cycle-8c0b60dd-c1-grammar-bounds-screen.json`](autotrain-cycle-8c0b60dd-c1-grammar-bounds-screen.json).
