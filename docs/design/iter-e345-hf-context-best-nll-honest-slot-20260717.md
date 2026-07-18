# E345 bounded best-NLL honest-slot evaluation — 2026-07-17

E345 evaluates E337's `best_weighted_nll.pt` checkpoint rather than its final
checkpoint, retaining the honest visible-slot context and constrained-decoding
policy from E341. The four-suite evaluation completed in about 56 seconds,
under the hard 300-second cap.

The local unsynced checkpoint SHA is
`6f0ecf7cce2ebc7c61f133c13456ac91bcd4861bd3e2f4f70a3a72473c211985`.
All 16 examples parse. Held-out meaningful rate, component recall, and reward
recover to 0.20/0.1333/0.1658; the other three suites remain at zero on all
three semantic metrics. Smoke/held/adversarial/OOD fidelity is
0.2778/0.4533/0.1667/0.1667. AgentV passes 0/4 with no execution errors. RICO
was omitted.

**Verdict:** reject E345 as a checkpoint, but retain best-NLL selection as the
stronger bounded selection policy for this HF-context run. It recovers limited
held-out semantic signal compared with E337's final checkpoint, but the signal
does not generalize across suites and clears no gate. Do not promote or sync.
