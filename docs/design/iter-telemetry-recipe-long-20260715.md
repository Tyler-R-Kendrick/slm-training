# Longer recipe control (2026-07-15)

At the same 8,845-target-token budget, the baseline (`grad_accum=1,
lr=3e-4`) reached weighted held-out NLL **21.79** in **137.15s**. The tuned
control (`grad_accum=2, lr=6e-4`) reached **22.59** in **109.62s**. The control
was approximately **20.08% faster** with a **3.66% NLL regression**.

This is a concrete speed/quality tradeoff, not a default replacement. Keep it
as an explicit candidate and require generated-quality validation before using
it for any ship or promotion decision.
