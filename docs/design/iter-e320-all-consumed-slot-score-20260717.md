# E320 all-consumed-slot scoring — 2026-07-17

E320 replaces next-slot-only component bias with the mean owner score across
every upcoming slot the candidate's existing schema contract requires. The
unchanged E318 r2 checkpoint was evaluated after accepted E319 slot assignment.

All five suites exactly reproduce E319: parse/fidelity are 1.0; smoke,
held-out, adversarial, OOD, and limited-RICO meaningful/recall/reward are
0.6667/0.3333/0.6407, 0.40/0.20/0.3916, 0.50/0.375/0.4805,
0.50/0.25/0.4992, and 1.0/0.5556/1.0. The 2/5/4/4/9 slot-choice changes are
also identical. AgentV remains 3/5; smoke and held recall fail.

**Verdict:** retain the schema-correct scorer but record no quality gain. Do
not promote or claim ship. Candidate arity is no longer the limiting factor;
future work should improve leakage-free semantic role representation/data.
