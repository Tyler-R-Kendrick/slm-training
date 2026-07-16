# E118 post-hookfix small control (2026-07-15)

E118 reran the previously completing small recipe after fixing changed-test
selection in `scripts/check_changed.py`. The command transport detached before
showing completion, but the child process continued and finished all **64**
steps. The durable summary reports final loss **9.9858**, 1,417 training
records, and a local scratch checkpoint.

Telemetry and the checkpoint are under
`outputs/runs/iter-e118-small-after-hookfix-20260715/e118_small_after_hookfix/`.
No evaluation or quality claim is made. This verifies that changed-test
selection no longer blocks the run; the apparent interruption was a detached
command-session reporting problem, not trainer termination.
