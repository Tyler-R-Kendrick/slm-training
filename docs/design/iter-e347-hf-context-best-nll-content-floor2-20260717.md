# E347 bounded best-NLL two-content floor — 2026-07-17

E347 raises E346's minimum content-component floor from one to two while
keeping the same best-weighted-NLL checkpoint and honest visible-slot policy.
The four-suite evaluation completed in 22.9 seconds, under the hard
300-second cap.

The local unsynced checkpoint SHA is
`6f0ecf7cce2ebc7c61f133c13456ac91bcd4861bd3e2f4f70a3a72473c211985`.
Every example parses. Smoke/held/adversarial/OOD meaningful rate is
0.6667/0.40/0.75/0.75; component recall is 0.50/0.20/0.50/0.50; reward is
0.6407/0.3844/0.6583/0.6008. AgentV passes smoke, adversarial, and OOD, for
3/4 overall, with no execution errors. Held-out misses only its component
recall gate: 0.20 versus the required 0.30. RICO was omitted.

**Verdict:** retain E347 over E346 as the strongest bounded HF-context decode
policy. It is not a checkpoint promotion and not a ship result: held-out still
fails, the sample is only 16 examples, and the full RICO gate was not run.
