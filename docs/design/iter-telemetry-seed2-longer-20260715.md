# Iteration: seed-2 longer feedback (2026-07-15)

The seed-2 batch-8 scratch run was extended to four steps with final-only loss
feedback. It reached complete weighted held-out NLL **30.283** at **5,581**
target tokens. Effective batch size 8 and all required telemetry/insight/loss
artifacts were persisted; loss suites consumed 76.54% of wall time.

Compared with the same recipe, seed 0 reached 27.977 and seed 1 reached
31.165. Seed 2 is closer to seed 1 than the earlier two-step result suggested,
so the current evidence still supports seed variance rather than a stable
hard-example or corpus defect. No data reweighting or recipe change is made.
