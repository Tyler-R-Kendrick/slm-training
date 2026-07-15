# Iteration: 128-step generation feedback and regression (2026-07-15)

Seed 0 continued the same scratch recipe to **128 steps** (batch size `8`,
learning rate `6e-4`, effective batch size `8`, unchanged 585-record corpus).
The run consumed **172,553 target tokens** and persisted complete telemetry.
Final weighted held-out NLL was **7.312**, regressing from **7.057** at 64
steps.

The bounded constrained smoke evaluation used 16 generation steps and one
attempt. It completed in **3,464.44 ms**, but partial generation also regressed:
structural similarity **0.1917**, placeholder validity **0**, and component
recall **0**. Parse and reward remained **0**. The persisted parse diagnostic
identified malformed structure: `Stack` received 9 arguments where 6 are
expected and its required `children` field was null.

Decision: do not promote the 128-step final checkpoint. The 64-step checkpoint
is the current generation-quality best. Repeat the 128-step budget with
interval loss feedback (every 32 steps) to select a measured checkpoint rather
than trusting the final step; this is a scratch diagnostic, not a ship claim.
