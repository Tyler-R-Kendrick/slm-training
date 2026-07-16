# E107 no-LTR-primary control (2026-07-15)

E107 evaluated the unchanged E104 checkpoint with grammar LTR primary
disabled, keeping constrained decoding, repair, slot contracts, and
no-fallback behavior enabled. This isolates the learned denoiser/MaskGIT path
from the LTR-primary route.

The control was effectively unchanged: parse/raw syntax `0.0/0.0`, structural
similarity `0.425`, contract precision/recall `1.0/1.0`, placeholder fidelity
`1.0`, component recall `0.25`, and latency `10280.75 ms`. AgentV failed all
five checks.

Decision: the E104 corruption is not specific to LTR-primary selection. The
next iteration should inspect the learned denoiser logits/target corruption
for Stack-list positions.
