# E301 concise connected content (2026-07-17)

E301 closes the E300 Stack child list and Stack component immediately after
the prompt-derived required content references. The checkpoint, SHA, CPU
scratch five-suite recipe, prompt-only honesty policy, and
`decode_min_content=-1` setting are unchanged. This isolates concise topology
from training and component selection.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Recall | Reward | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.3333 | 0.5278 | 0.4642 | 0.1667 | 0.2497 | 258.46 |
| held_out | 5 | 1.0000 | 0.0000 | 0.2800 | 0.3369 | 0.0000 | 0.0000 | 248.22 |
| adversarial | 4 | 1.0000 | 0.5000 | 0.5417 | 0.4744 | 0.3750 | 0.4245 | 252.56 |
| ood | 4 | 1.0000 | 0.0000 | 0.2583 | 0.3750 | 0.0000 | 0.0000 | 233.64 |
| rico_held | 3 | 1.0000 | 0.6667 | 0.2083 | 0.3038 | 0.3333 | 0.5067 | 426.71 |

Relative to E300, parse returns to 1.0 everywhere, pathological
overgeneration disappears, failed thresholds fall 9→7, and AgentV improves
1/5→2/5 with zero execution errors. Adversarial and the limited RICO suite now
clear their complete suite rows. Smoke still misses meaningful, component
recall, and reward; held-out and OOD remain meaningful/recall 0.0.

The remaining failure is component selection, not syntax or connectivity:
the frozen checkpoint selects TextContent for every required declaration.
That is sometimes present in gold layouts, but it cannot recover forms,
inputs, tabs, buttons, cards, galleries, modals, or auth structures.

**Verdict:** concise connected topology is the best choice decode policy so
far and should replace E300's open-tail diagnostic when the opt-in floor is
used. It remains non-default and non-ship until component-type selection
improves and full gates pass.

Artifacts:

- `outputs/runs/e301-choice-connected-content-close-honest-r1/`
- [machine-readable result](choice-connected-close-results-iter-e301-20260717.json)
