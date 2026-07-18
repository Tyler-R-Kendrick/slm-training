# E360–E362 Card-hierarchy adaptation — 2026-07-17

E360 tests whether E357's train-only Card hierarchy examples repair the
dominant full-RICO structural miss. It initializes from E337's
best-weighted-NLL serving checkpoint, starts a new optimizer and token counter
on the changed corpus, and consumes 5,039 target tokens in 97 CPU steps.
Training completes in 96.7 seconds with final weighted NLL 5.8091. The frozen
SmolLM2 revision, loss recipe, and E350 decode policy are otherwise retained.
The local checkpoint was explicitly not synced.

E361 evaluates the checkpoint on the complete bounded smoke, held-out,
adversarial, and OOD suites in 28.7 seconds. AgentV passes 4/4. Smoke,
held-out, and adversarial metrics exactly match E350. OOD meaningful rate,
component recall, and reward also match, but placeholder fidelity regresses
from 0.5167 to 0.3500 and structure from 0.5235 to 0.4814.

E362 evaluates RICO rows 0–63 in 63.1 seconds. All 64 serialized predictions
are byte-for-byte identical to the E353 control. The adaptation therefore
recovers `Card` in 0 additional examples and leaves every aggregate metric
unchanged: parse 1.0, meaningful rate 0.9844, fidelity 0.2553, structure
0.2435, component recall 0.5104, and reward 0.7249. AgentV correctly reports
0/1 because this is only 64/1500 rows.

Every command used both the trainer's 4.5-minute graceful deadline where
applicable and an external interrupt at 290 seconds with a hard kill by 300
seconds.

**Verdict:** reject E360. The proposed data intervention does not change the
target RICO behavior and regresses OOD fidelity/structure. Retain E357 as
useful diagnosed data coverage and retain the explicit weight-only
initialization harness, but do not promote or sync the checkpoint.
