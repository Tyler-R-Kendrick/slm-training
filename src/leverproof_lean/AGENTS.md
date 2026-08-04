# Agent instructions

This is a Lean 4 proof project.

- Use the installed official `lean4-setup` skill for toolchain or project setup.
- Use the installed official `lean-proof` skill for theorem work.
- Keep `lean-toolchain` pinned and use Lake for builds.
- Never finish with `sorry`, `admit`, `axiom`, or `unsafe` in the proof core.
- Change one proof step at a time and run `make proofs` after theorem changes.
- Keep `metric_evidence/v1` and `metric_certificate/v1` backward compatible.
- Run `make test` before handing off a change.
