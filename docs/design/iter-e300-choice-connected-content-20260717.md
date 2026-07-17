# E300 choice connected content (2026-07-17)

E300 reevaluates the unchanged E297 checkpoint (SHA-256
`a78193f91ee12d07791cab008a75267e3f6e19cfd223fbc726b3896dd98d14ee`)
after strengthening the opt-in choice-native `decode_min_content=-1` policy.
Unlike E299's direct content root, E300 requires one or more string-bearing
bound components, a Stack root with an explicit child list, and references
from that list to the required content declarations. The CPU scratch,
prompt-only, five-suite ship-gate recipe and no-fallback policy are unchanged.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 0.6667 | 0.3333 | 0.3889 | 0.4597 | 0.1667 | 0.2830 |
| held_out | 5 | 1.0000 | 0.0000 | 0.5600 | 0.2996 | 0.0000 | 0.0000 |
| adversarial | 4 | 1.0000 | 0.2500 | 0.8333 | 0.4635 | 0.1250 | 0.2123 |
| ood | 4 | 1.0000 | 0.0000 | 0.5167 | 0.4346 | 0.0000 | 0.0000 |
| rico_held | 3 | 1.0000 | 0.6667 | 0.2083 | 0.3038 | 0.3333 | 0.5067 |

Relative to E299, connected content restores structure substantially and
reduces failed thresholds from 12 to 9. AgentV improves from 0/5 to 1/5 with
zero execution errors. This is still not a ship result: held-out and OOD
meaningful rates remain zero, smoke parse falls to 0.6667, and adversarial
recall misses. The open variadic Stack list allows many unnecessary primitive
children; one adversarial record hits the 4× pathological-overgeneration guard
and one smoke record fails to produce a root.

**Verdict:** connectivity is the right direction, but E300 is diagnostic-only
and not promotable. E301 should close the root list immediately after the
required connected references, testing whether concise topology preserves the
structural gain while restoring parse and preventing overgeneration.

Artifacts:

- `outputs/runs/e300-choice-connected-content-honest-r1/`
- [machine-readable result](choice-connected-content-results-iter-e300-20260717.json)
