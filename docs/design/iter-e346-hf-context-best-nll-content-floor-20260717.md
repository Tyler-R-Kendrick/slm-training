# E346 bounded best-NLL content-floor evaluation — 2026-07-17

E346 combines E345's best-weighted-NLL checkpoint and honest visible-slot
policy with a minimum of one content component before EOS. The four-suite
evaluation completed in 22.6 seconds, under the hard 300-second cap.

The local unsynced checkpoint SHA is
`6f0ecf7cce2ebc7c61f133c13456ac91bcd4861bd3e2f4f70a3a72473c211985`.
Every example parses. Smoke/held/adversarial/OOD meaningful rate is
0.6667/0.40/0.75/0.50; component recall is 0.3333/0.20/0.50/0.25; reward is
0.6327/0.3496/0.6055/0.3645. AgentV passes adversarial and OOD, for 2/4
overall, with no execution errors. Smoke misses its 0.35 recall gate at
0.3333, while held-out misses its 0.30 recall gate at 0.20. RICO was omitted.

The floor changes the decode path materially: the slot-component policy is
applied once per example across all suites and changes six of 16 choices. In
E345, without the floor, it was never applied.

**Verdict:** retain E346 as the strongest bounded HF-context decode policy.
It is not a checkpoint promotion and not a ship result: two suites still fail,
the sample is only 16 examples, and the full RICO gate was not run. The next
lever should target component recall without weakening the honest contract.
