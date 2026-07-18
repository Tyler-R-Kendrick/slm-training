# E343 bounded visible-slot HF adaptation — 2026-07-17

E343 resumes E337 for 5,028 additional target tokens with slot-contract
context and constrained slot decoding enabled during training, component-plan
decode weight zero, and the same pinned frozen SmolLM2 context. The train
completed in 120.1s at 35,044 total tokens under the hard 300-second cap.

Final weighted/broad NLL are 6.0674/6.1001; best inherited weighted NLL remains
5.7512. Final-20 slot accuracy is 0.9200 versus a 0.6250 majority baseline.
Loss AgentV passes 2/2. The local unsynced checkpoint SHA is
`8be4ce4ceeb0e84d3891c789f3ac687eb4d38ccb94c29acac68f5b59e4f8610d`.

| Suite | n | Parse | Fidelity | Structure | Meaningful | Recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 0.1111 | 0.3150 | 0.0 | 0.0 | 0.0 |
| held_out | 5 | 1.0 | 0.1400 | 0.2884 | 0.0 | 0.0 | 0.0 |
| adversarial | 4 | 1.0 | 0.1458 | 0.1986 | 0.0 | 0.0 | 0.0 |
| ood | 4 | 1.0 | 0.2167 | 0.2366 | 0.0 | 0.0 | 0.0 |

The honest four-suite evaluation completed in 55.4s; AgentV passes 0/4 with
no execution errors. RICO was intentionally omitted. Relative to E341's
decode-only visible-slot policy, adaptation reduces smoke fidelity
0.6389→0.1111 and erases OOD meaningful/recall/reward
0.50/0.25/0.3435→0/0/0.

This run also exposed a provenance omission: `train_summary.json` did not
include `slot_contract_in_context` or `slot_contract_constrained_decode` even
though checkpoint metadata did. The summary recipe now records both flags and
a focused train-loop regression test covers them.

**Verdict:** reject E343. Short visible-slot adaptation does not preserve
E341's recovered signal. Do not promote, sync, or claim RICO/ship readiness.

