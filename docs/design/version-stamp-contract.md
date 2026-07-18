# Version-stamp contract (normalized component versioning)

**Date:** 2026-07-18
**Status:** active contract. Enforced by `scripts/verify_version_stamps.py` in
CI, the pre-commit changed-file check, and agent PostToolUse hooks.

The eval/smoke/checkpoint stack is self-improving: metric definitions, gate
thresholds, harness implementations, matrices, and the eval-suite builder all
change over time. This contract makes every result say **which revision of
those constraints produced it**, and makes results produced under since-changed
constraints **discoverable for re-testing** instead of silently incomparable.

It complements — never replaces — the checkpoint-provenance contract
([checkpoint-provenance.md](checkpoint-provenance.md)): checkpoint references
pin *artifacts* (commits + content hashes); version stamps pin the *constraint
stack* that scored them.

## The registry

`src/slm_training/resources/versions.json` (schema `version_registry/v1`) is
the single committed source of truth:

```json
{
  "schema": "version_registry/v1",
  "components": {
    "gates.ship": {
      "version": "openui_ship_gates_v1",
      "kind": "gate",
      "paths": ["src/slm_training/harnesses/model_build/ship_gates.py"],
      "history": [
        {"version": "openui_ship_gates_v1", "date": "2026-07-18", "note": "initial registration"}
      ]
    }
  }
}
```

- **Component ids** are dotted lowercase (`harness.model_build.eval`,
  `evals.meaningful_program`, `gates.ship`, `matrix.quality`,
  `data.test_build`, …); `kind` ∈ harness | metric | gate | matrix |
  data_builder.
- **`paths`** are repo-relative watched files; a trailing `/` claims a
  directory prefix. When one file is matched by several claims the **longest
  prefix wins** (`evals/meaningful_program.py` belongs to
  `evals.meaningful_program`, not the `evals.scoring` directory catch-all).
  No two components may claim the identical path string.
- **`history`** is append-only, newest first; `history[0].version` must equal
  `version`. **Ordering authority is the position in `history`, never string
  comparison** — legacy encodings (`2.0.0`, `openui_ship_gates_v1`,
  `vss4-02-v1`) order correctly next to new monotonic `v1, v2, …` labels.
- Pre-existing in-code constants (`LOSS_SUITE_VERSION`, `METRIC_VERSION`,
  `MEANINGFUL_METRIC_POLICY["threshold_version"]`, `MATRIX_VERSION`) stay
  canonical at runtime; `tests/test_versioning/test_mirrors.py` pins the
  registry to them so neither can drift.

## The stamp

Canonical result writers embed a `version_stamp` envelope built by
`slm_training.versioning.build_version_stamp(*component_ids)`:

```json
"version_stamp": {
  "stamp_schema": "version_stamp/v1",
  "code_commit": "903b8a8…",
  "code_dirty": false,
  "components": {
    "harness.model_build.eval": "v1",
    "evals.meaningful_program": "2.0.0",
    "evals.scoring": "v1"
  },
  "stamped_at": "2026-07-18T20:15:00+00:00"
}
```

Stamped payloads: `eval_<suite>.json` / `eval.json` / `scoreboard.json`
(eval runner), `gates.json` (ship gates), `train_summary.json` and
`loss_suites.json` (train loop), the quality/grammar/perf matrix summaries and
their `docs/design/*-results.json` mirrors, the verified-solver matrix and
VSS4 campaign reports, `bench_summary.json` (reasoning bench), the solver
bench CLI payload, and the slop-forensics report.

Stamping is provenance, not a gate: on environmental failure (no git, wheel
install, unreadable registry) fields degrade to the explicit `UNKNOWN` / null
sentinels — a provenance failure never kills a run. Durable checkpoints pin
their stamp transitively: stamped `train_summary.json` / `eval.json` are
SHA-256-hashed companion files of every frontier/ship checkpoint reference.

## The bump rule

**Any change to a file under a component's `paths` must touch that component's
registry entry in the same change:**

- **Behavioral change** → bump `version` (new ids and future bumps use
  monotonic `v1, v2, …`) and prepend a history entry
  (`{"version", "date", "note"}`).
- **Behavior-neutral change** (comments, type hints, log text, refactor with
  proven-identical outputs) → prepend a **same-version** history entry whose
  note starts with **`no-bump:`** and states why. This is the audited escape
  hatch: it is a reviewable file diff, survives squash-merges, and is
  greppable later.
- History is append-only in every change — existing entries are never edited
  or dropped.

When in doubt, bump: a false bump costs one registry line; a missed bump
poisons every later comparison against the old numbers.

## Enforcement map

| Surface | Mode | Command |
| --- | --- | --- |
| CI (`.github/workflows/ci.yml`) | blocking | `verify_version_stamps --check --base <PR base sha>` (PRs) / `--check` (push) |
| Pre-commit (`.githooks/check-changed` → `scripts/check_changed.py`) | blocking | `verify_version_stamps --check --staged` |
| Agent PostToolUse hook (`.claude/settings.json`) | advisory nudge | `verify_version_stamps --post-tool-use` |
| Agents directly (skills, AGENTS.md iron law item 8) | instruction | `verify_version_stamps --check` before finishing |

`--check` also lints the registry shape and requires every **newly added**
`docs/design/*.json` experiment result (structural sniffing on result-shaped
keys, never filename-only) to carry a valid stamp. **Modified legacy files
only warn** — the pre-contract ledger is grandfathered, not rewritten.

## Re-test discovery

```bash
# All results whose stamped versions are behind the current registry:
python -m scripts.verify_version_stamps --stale

# Scoped to one component, including gitignored run artifacts, machine-readable:
python -m scripts.verify_version_stamps --stale --component gates.ship \
  --include-outputs --json
```

The report groups retest candidates per component with `behind_by` (history
distance), flags `unrecognized_version` / `component_retired` stamps, and
counts `legacy_unstamped` pre-contract files. After bumping a component,
run `--stale --component <id>` and either re-run the listed experiments or
label their rows invalidated in the matrix markdown
(`running-experiment-matrices` skill).

## Worked examples

**Tightening a ship gate** (`ship_gates.py` threshold change):
1. Edit `ship_gates.py` (and `MEANINGFUL_METRIC_POLICY["threshold_version"]`
   → e.g. `openui_ship_gates_v2` — the mirror test forces the pair).
2. In `versions.json`, set `gates.ship.version` to `openui_ship_gates_v2` and
   prepend `{"version": "openui_ship_gates_v2", "date": …, "note": "raised
   held_out parse bar to 0.45"}`.
3. `python -m scripts.verify_version_stamps --stale --component gates.ship`
   → decide which historical clears to re-run or label invalidated.

**Ruff-only cleanup of `eval_runner.py`:**
1. Prepend to `harness.model_build.eval.history`:
   `{"version": "v1", "date": …, "note": "no-bump: ruff formatting only"}`.

## Non-goals

- No rewriting of pre-contract `docs/design` history (grandfathered).
- No change to `CheckpointReferenceV1` (byte-stable `sha` contract); linkage
  is via hashed companion files.
- `run_id` normalization and dashboard surfacing of stamps are follow-ups,
  not part of this contract.
