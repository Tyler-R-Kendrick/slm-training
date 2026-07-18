# E337 bounded HF-context continuation to 30k — 2026-07-17

E337 resumes E336's pinned, frozen SmolLM2 context checkpoint and extends the
same E333-derived recipe from 20,008 to 30,016 target tokens. The train
completed in 256.33s under the hard 300-second command cap.

Best weighted NLL improves from E336's 6.2014 to 5.7512; final
weighted/broad NLL are 5.9240/5.9070. Final-20 slot-owner accuracy is 0.9381
versus a 0.6387 majority baseline. The local unsynced checkpoint SHA is
`6f3c1fda0048dfe85ed254edf6adb0801b872d1ad8acfed16443309af959c9f6`.
Loss AgentV passes 3/3 for the continuation's emitted loss evaluations.

| Suite | n | Parse | Fidelity | Structure | Meaningful | Recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 0.6667 | 0.0 | 0.2511 | 0.0 | 0.0 | 0.0 |
| held_out | 5 | 0.8 | 0.0 | 0.2044 | 0.0 | 0.0 | 0.0 |
| adversarial | 4 | 0.75 | 0.0 | 0.2340 | 0.0 | 0.0 | 0.0 |
| ood | 4 | 1.0 | 0.0 | 0.2884 | 0.0 | 0.0 | 0.0 |

The bounded four-suite evaluation completed in 49.5s; AgentV passes 0/4 with
no execution errors. RICO was intentionally omitted because it cannot finish
inside the five-minute policy, so this is not a full ship evaluation.

The first evaluation attempt exposed a generalized harness defect: an invalid
model-generated literal payload raised `JSONDecodeError` and aborted the run.
The choice codec now quotes invalid payloads as strings, letting the evaluator
score the malformed output. Thirty-nine focused codec/tokenizer tests pass,
and the rerun completed without execution errors.

**Verdict:** reject E337. Additional tokens improve loss and auxiliary slot
accuracy but do not recover fidelity, meaningful programs, component recall,
or reward. Do not promote, sync, or claim RICO/production readiness.

