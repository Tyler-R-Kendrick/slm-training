# E898: typed-role scratch attempts

E898 attempted to isolate E897 against E886's 600-step scratch recipe. Neither
attempt is valid model evidence. `r1` was externally interrupted by the command
session at roughly 30 seconds and emitted only trace/metric fragments with no
summary or checkpoint. `r2` exited cleanly on the harness wall budget after
279/600 steps in 95.09 seconds, with loss 4.47709608 and checkpoint SHA
`a1683be8…cf2161`.

The `r2` version stamp is dirty only because unrelated untracked SLM189 result
files appeared in the shared worktree; they were not read, modified, staged, or
included in this experiment. More importantly, `stopped_on=wall_time_budget`
makes the serialized checkpoint invalid regardless of its stamp. Do not
evaluate, sync, promote, serve, resume, or use either attempt as a parent.

The E897 corpus remains valid. Replace the overlong full scratch arm with a
strict two-row typed-role corpus mixed against the committed base distribution
in a short warm continuation. No ship gate or AgentV evaluation ran.
