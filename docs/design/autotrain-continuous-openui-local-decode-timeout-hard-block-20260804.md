# Autotrain continuous-openui-local: component-plan lineage hard-blocked (2026-08-04)

**Verdict:** three consecutive completed cycles (c2, c3, c5) reproduced the
identical `smoke:decode_timeout_count=3/3` measurement failure on the same
frozen `component-plan`/`control` (`structural_aux_head_profile="component-plan"`)
lineage, with in-pipeline self-heal exhausted. Per
[`continuous.md`](../../.agents/skills/autotrain/references/continuous.md)'s
hard-block rule ("Report blocked only after the same hard blocker has failed
three consecutive cycles with no new information and in-pipeline self-heal
could not recover"), this lineage is now blocked. This does not stop the
loop — the next cycle uses a fresh, non-replay hypothesis instead.

## Timeline

| Cycle | Action | Result |
| --- | --- | --- |
| c2 | Fresh matched-pair screen, `component_plan_decode_weight` candidate vs matched `structural_aux_head_profile="component-plan"` control | `decode_timeout_count=3/3` both arms; `compiler_ms_mean` ~23.15–23.2s |
| — | Repair: `screening_decode_timeout_seconds` 8s→10s (`harness.autoresearch.experiment_campaign` v177→v178, commit `a17419c`) | Landed, tested, confirmed live for every freshly built hypothesis (`matrix-proposal.json` shows 10.0 for all non-replay arms from c3 onward) |
| c3 | Frozen replay of c2's manifest under the v178 repair | Identical `decode_timeout_count=3/3`, `compiler_ms_mean` ~23.13–23.2s — the repair did not reach the replay because `_apply_frozen_replay` copies the frozen `knobs` dict (including `decode_timeout_seconds`) verbatim |
| — | Attempted repair: resample `decode_timeout_seconds` on replay (v178→v179, commit `218c9a3`) | Reverted same cycle (commit `5c53a64`) — broke a separate, independent authority invariant in `scripts/autoresearch.py::_authorized_replay_configs` (`current[0].knobs != source_experiments[0].knobs` → `"frozen replay manifest must authorize one unique arm"`). Two independent mechanisms agree a frozen replay's knobs, decode_timeout_seconds included, must stay byte-identical to the source arm; this is enforced design, not a bug |
| c5 (c4 crashed pre-handoff on the reverted-fix's authority violation, no valid cycle) | Frozen replay of c3's manifest, code reverted to v178 | Identical `decode_timeout_count=3/3` a third time, `compiler_ms_mean` ~23.17–23.25s |

## Why this is a genuine hard block, not a soft failure

- The v178 policy repair (8s→10s default) is real, tested, and live for
  every freshly generated screening hypothesis — confirmed directly in
  `matrix-proposal.json` for c3 and c5.
- It cannot reach an already-frozen replay by design: `decode_timeout_seconds`
  is locked to the value the arm was originally minted with, enforced by both
  `_apply_frozen_replay`'s knob copy and the independent
  `_authorized_replay_configs` authority check in `scripts/autoresearch.py`.
- Three consecutive completed cycles (c2, c3, c5) reproduced the exact same
  `decode_timeout_count=3/3` failure with statistically identical
  `compiler_ms_mean` (23.13–23.25s band) — no new information arrived between
  attempts once the mechanism was understood.
- No further in-pipeline knob change can unblock this specific lineage.
  Unblocking it for real would need either a faster decode path for
  `structural_aux_head_profile != "none"` arms (a model/runtime change well
  beyond a policy knob) or accepting that this exact frozen manifest chain is
  permanently unscoreable and must be abandoned in favor of a fresh hypothesis.

## Disposition

- The `component-plan` frozen lineage (c2 → c3 → c5) is abandoned. No
  checkpoint from any of these cycles is promotable, reusable, or
  ship-eligible.
- `retry_measurement` and `repair_harness` for c5
  (`frozen_manifest_sha256=98f49277d10dc0fa151c2d438e97e93f92da731cdb606be3be0dea9ad75dc2eb`)
  are acknowledged `status=blocked` with this document as evidence, so the
  next cycle proceeds with a fresh, non-replay hypothesis instead of a fourth
  replay of the same manifest.
- The v178 `screening_decode_timeout_seconds` repair stands and remains
  correct — it benefits every future `structural_aux_head_profile`-bearing
  arm that is *not* a frozen replay of this specific poisoned lineage.
- A future, deliberate design change to `_apply_frozen_replay`/
  `_authorized_replay_configs` (letting `decode_timeout_seconds` resample on
  replay while still enforcing exact lever-knob identity) is a legitimate
  follow-up, but it is a bigger contract change than a single autonomous
  supervised cycle should make unilaterally — it touches a security-relevant
  invariant (`"frozen replay manifest authority is invalid"`) shared by
  promotion and confirmation replays, not just screening.
