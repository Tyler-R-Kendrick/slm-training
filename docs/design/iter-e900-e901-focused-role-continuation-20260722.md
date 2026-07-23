# E900-E901: focused typed-role continuation

E900 tested 20-step continuation from E891 with 75% E851 base exposure and
25% E899 focused typed-role exposure. `r1` failed before training because using
the two-row focus corpus as primary inferred a 256-token context shape that was
incompatible with E891's 308-position checkpoint; it emitted no checkpoint.
`r2` corrected the canonical data roles—E851 primary, E899 replay—and completed
20/20 CPU steps in 4.70 seconds with 61 base and 19 focused examples.

| Run | Checkpoint / suite | n | parse | meaning-v1 | strict-v2 | fidelity | structure | recall | reward | timeout / fallback | AgentV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E892 | E891 / held_out | 5 | 0.8000 | 0.8000 | 0.2000 | 0.8000 | 0.3298 | 0.7143 | 0.7844 | 1 / 0 | 0/1 |
| E901 | E900 / held_out | 5 | 0.2000 | 0.0000 | 0.0000 | 0.0333 | 0.0293 | 0.0667 | 0.1314 | 4 / 2 | 0/1 |

Focused repetition catastrophically destabilizes generation: four rows time
out and return empty, and the fifth collapses to one `TextContent`. Reject E900
without a smoke run. Never sync, promote, serve, resume, or use it as a parent.
The next matched arm may test strong initialization-weight retention with the
same exposure, but only E891 remains a valid parent.

The version stamp is dirty solely because unrelated untracked SLM189 result
files remain in the shared worktree; they were left untouched. This is a
five-row diagnostic failure, not ship evidence, and AgentV fails.
