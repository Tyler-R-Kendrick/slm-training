# E340 bounded HF-context explicit content floor — 2026-07-17

E340 replaces E339's inert automatic floor with an explicit minimum of one
content component on the unchanged E337 checkpoint, retaining plan-off. The
four-suite evaluation completed in 23.2s under the hard 300-second cap.

| Suite | n | Parse | Fidelity | Structure | Meaningful | Recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| held_out | 5 | 0.6 | 0.0 | 0.2083 | 0.0 | 0.0 | 0.0 |
| adversarial | 4 | 0.25 | 0.0 | 0.0738 | 0.0 | 0.0 | 0.0 |
| ood | 4 | 0.5 | 0.0 | 0.0977 | 0.0 | 0.0 | 0.0 |

AgentV passes 0/4 with no execution errors. The explicit floor prevents the
model's preferred termination but mostly yields invalid reconstructions and
does not recover any fidelity, meaningful-program, component-recall, or reward
signal. RICO was intentionally omitted.

**Verdict:** reject E340. Premature termination is not the missing semantic
mechanism. No checkpoint was written or promoted.

