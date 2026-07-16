# Longer tuned accumulation quality check (2026-07-15)

The tuned `grad_accum=2, lr=6e-4` candidate was evaluated after six training
steps with the proper eight-step decode path. Smoke `n=3` still scored zero for
parse rate, placeholder fidelity, structural similarity, and reward. The E1
matrix row failed, and the run emitted the pinned AgentV JSONL artifacts.

The candidate is rejected for generated quality. Its improved held-out loss is
useful telemetry, but it does not support a recipe change or a ship claim.
