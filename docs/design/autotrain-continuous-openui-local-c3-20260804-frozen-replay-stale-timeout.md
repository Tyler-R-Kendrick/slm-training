# Autotrain continuous-openui-local c3 (2026-08-04): frozen replay reproduced the pre-repair timeout

**Verdict:** measurement incomplete again, no model attribution. Deeper
harness gap found and repaired; second replay queued.

Cycle 3 replayed the frozen c2 manifest
(`frozen_manifest_sha256=2fd7771dab610c8ddc6c6d32cba0eedfc24bd549471b7d946b26d7be5df70581`)
under the v178 repair (`screening_decode_timeout_seconds` 8s → 10s, commit
`a17419c`). It reproduced the **identical** `smoke:decode_timeout_count=3/3`
failure, `compiler_ms_mean` ~23,128–23,198ms — statistically the same cost as
c2.

Cause: `matrix-proposal.json` for c3 shows every **freshly generated**
hypothesis using `decode_timeout_seconds=10.0` (the repair took effect), but
the two arms actually replayed (`control`, `component-plan`) both carried
`decode_timeout_seconds=8.0` — the frozen c2 value. `_apply_frozen_replay`
(`scripts/run_autotrain_continuous.py`) copies the entire frozen `knobs` dict
verbatim onto the replay target, and `decode_timeout_seconds` is part of that
dict. But this file documents `decode_timeout_seconds` elsewhere
(`_LEVER_KNOB_KEYS`'s comment, `_thrash_lever_signature`) as a **measurement
knob**, explicitly excluded from an experiment's lever/scientific identity —
so freezing it across a replay silently reintroduces whatever timeout bug the
repair just fixed, defeating the entire "repair harness, then replay" cycle
law in [`continuous.md`](../../.agents/skills/autotrain/references/continuous.md).

Repair (v178 → v179, commit `218c9a3`): `_apply_frozen_replay` now re-samples
`decode_timeout_seconds` from the freshly built matrix for every replay type
(screening, promotion, confirmation); every other knob (model levers, steps,
seed, batch_size) still copies byte-identical from the frozen arm, so the
actual experiment stays reproducible. Updated
`test_frozen_replay_preserves_recipe_and_links_current_main_successor` to
prove decode_timeout_seconds resamples while everything else stays frozen
(previously it asserted full knobs-dict equality, which is what let this slip
through).

Per continuous-loop law the identical frozen arm must be replayed again under
this second repair before any new hypothesis. No checkpoint from c2 or c3 is
promotable, reusable, or ship-eligible.
