# E421–E422 exact continuation-step boundary — 2026-07-18

E421 completes the deterministic bisection from E396. It runs exactly two
power-zero optimizer steps, ending at step 429 / 22,127 target tokens in 8.2
seconds. Checkpoint SHA is
`81d5db009a9ab1d56337886713d54075522cf6000b8f63fdf83c77bac51e1940`.
It is local-only, inherits best weighted NLL 5.8091 without a fresh loss
evaluation, and is not promoted.

E422 passes all four complete bounded suites with AgentV 4/4. Smoke remains
fully meaningful with type recall 0.5. The only global failure is missing full
RICO. Combined with failed E419 at step 430, this identifies the exact
transition: step 429 passes; step 430 fails smoke recall.

The triggering step-430 batch contains a LineChart prompt paraphrase and a
semantic-slot edit trajectory. Its total loss is 12.5489, component-plan loss
2.4499, root loss 0.9788 with accuracy 0.5, and bound top-k recall 0.125.
Those are diagnostic associations, not proof that either record alone causes
the regression.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 1.0 | 0.5600 | 0.5000 | 0.9770 |
| held_out | 5 | 1.0 | 0.6 | 1.0 | 0.5933 | 0.4833 | 0.5916 |
| adversarial | 4 | 1.0 | 0.75 | 1.0 | 0.6304 | 0.6250 | 0.7238 |
| ood | 4 | 1.0 | 0.75 | 1.0 | 0.5352 | 0.6042 | 0.7335 |

Every command used the external 290-second interrupt / ten-second forced kill;
training also used the internal 4.5-minute limit. E421 stopped normally on
token budget. E422 completed in 24.8 seconds and returned exit 8 only because
full RICO is absent. No timed-out process contributes evidence.

**Verdict:** step 429 is bounded-safe and step 430 is the first failing
checkpoint. E421 is a diagnostic boundary checkpoint, not a new champion.
Retain E396 while diagnosing step 430; do not spend full RICO yet.
