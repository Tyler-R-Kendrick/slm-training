# E902-E903: focused typed-role continuation with retention

E902 repeats E900's 20-step E891 continuation with the same 61 E851 base and
19 E899 focused examples, changing only initialization-weight retention from 0
to 5%. It completes in 4.30 seconds. Retention reduces RMS drift from
0.00128895 to 0.00082110 and prevents E900's four-timeout collapse.

| Run | Checkpoint / suite | n | parse | meaning-v1 | strict-v2 | fidelity | structure | recall | reward | timeout / fallback | AgentV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E892 | E891 / held_out | 5 | 0.8000 | 0.8000 | 0.2000 | 0.8000 | 0.3298 | 0.7143 | 0.7844 | 1 / 0 | 0/1 |
| E901 | E900 / held_out | 5 | 0.2000 | 0.0000 | 0.0000 | 0.0333 | 0.0293 | 0.0667 | 0.1314 | 4 / 2 | 0/1 |
| E903 | E902 / held_out | 5 | 0.8000 | 0.4000 | 0.0000 | 0.6400 | 0.1824 | 0.4524 | 0.7202 | 1 / 1 | 0/1 |

Weight anchoring restores parse and most surface metrics relative to E900, but
the candidate remains worse than its E891 parent on every semantic/topology
metric and loses the only strict-v2 pass. Reject E902 without smoke. Never
sync, promote, serve, resume, or use it as a parent. Focused typed-role
continuation is falsified at both 0% and 5% retention; the next work should use
the existing schema-typed decoder path rather than more weight updates.

The version stamp is dirty solely because unrelated untracked SLM experiment
files remain in the shared worktree and were left untouched. This is a
five-row diagnostic failure, not ship evidence, and AgentV fails.
