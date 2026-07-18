# E350 bounded two-content floor plus slot weight 8 — 2026-07-17

E350 doubles E347's slot-component decode weight from 4 to 8 while retaining
the two-content floor, best-weighted-NLL checkpoint, and honest visible-slot
policy. The four-suite evaluation completed in about 31 seconds, under the
hard 300-second cap.

The local unsynced checkpoint SHA is
`6f0ecf7cce2ebc7c61f133c13456ac91bcd4861bd3e2f4f70a3a72473c211985`.
Every example parses. Smoke/held/adversarial/OOD meaningful rate is
0.6667/0.40/0.75/0.75; component recall is 0.50/0.30/0.50/0.50; reward is
0.6407/0.3844/0.6583/0.6008. AgentV passes all four bounded suites with no
execution errors.

The stronger slot prior changes one additional held-out component choice:
`held_out_input_01` now emits both `Input` and `Button`, raising that example's
recall from 0.5 to 1.0 and aggregate held-out recall from 0.20 to the 0.30
gate. Other suite aggregates match E347 except held-out structure, which rises
from 0.4375 to 0.5075.

**Verdict:** retain E350 as the strongest bounded HF-context decode policy.
This is not a checkpoint promotion or a production ship result: it evaluates
only 16 examples, omits full `rico_held`, and uses a local unsynced checkpoint.
