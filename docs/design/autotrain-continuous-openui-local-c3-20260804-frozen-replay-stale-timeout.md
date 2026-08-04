# Autotrain continuous-openui-local c3 (2026-08-04): frozen replay reproduced the pre-repair timeout

**Verdict:** measurement incomplete again, no model attribution. A proposed
knob-resample repair was landed, found to violate a separate hard authority
invariant, and reverted the same cycle. This lineage cannot be unblocked by a
policy-level timeout change; it needs a fresh (non-replay) hypothesis.

Cycle 3 replayed the frozen c2 manifest
(`frozen_manifest_sha256=2fd7771dab610c8ddc6c6d32cba0eedfc24bd549471b7d946b26d7be5df70581`)
under the v178 repair (`screening_decode_timeout_seconds` 8s → 10s, commit
`a17419c`). It reproduced the **identical** `smoke:decode_timeout_count=3/3`
failure, `compiler_ms_mean` ~23,128–23,198ms — statistically the same cost as
c2, because `_apply_frozen_replay` (`scripts/run_autotrain_continuous.py`)
copies the frozen `knobs` dict verbatim, including `decode_timeout_seconds`.

**First attempt (reverted):** commit `218c9a3` changed `_apply_frozen_replay`
to re-sample `decode_timeout_seconds` from the freshly built matrix instead of
freezing it, reasoning from `_LEVER_KNOB_KEYS`'s comment ("Measurement knobs
(seed, decode_timeout, eval_suites) are re-sampled from role policy") and
`_thrash_lever_signature`'s exclusion of `decode_timeout_seconds` from lever
identity. Attempting cycle 4 under that change hit a **separate, independent**
authority gate: `scripts/autoresearch.py::_authorized_replay_configs` asserts
`current[0].knobs != source_experiments[0].knobs` and raises `"frozen replay
manifest must authorize one unique arm"` on any mismatch — including a
deliberately changed `decode_timeout_seconds`. This is a second, independent
enforcement of full knob-identity across a frozen replay, which the first fix
did not account for. Since two separate mechanisms agree that a frozen
replay's knobs (all of them, not just the registered lever subset) must stay
byte-identical, the `_LEVER_KNOB_KEYS` comment's "re-sampled from role policy"
describes a *different* action (fresh-seed confirmatory retest, which builds
a new matrix rather than replaying one), not this frozen-replay path. Commit
`218c9a3` was reverted in `5c53a64` (tests and policy version restored to
v178); the v178 `screening_decode_timeout_seconds` 8s→10s repair itself is
still correct and stands for every freshly built hypothesis.

**Conclusion:** a frozen replay of this lineage (`continuous-openui-local`,
component-plan family, originated at c2) cannot be repaired by any policy
knob — `decode_timeout_seconds` is locked to whatever value the arm was
originally minted with, by design, for both the copy logic and the separate
authority check. Continuing to replay it will reproduce the identical
`decode_timeout_count=3/3` failure indefinitely. The productive next step is
a **fresh, non-replay** screening hypothesis (not a replay of c2/c3), which
correctly picks up the current `screening_decode_timeout_seconds=10s` default
confirmed live in c3's own `matrix-proposal.json` for every non-replayed
candidate.

No checkpoint from c2 or c3 is promotable, reusable, or ship-eligible.
