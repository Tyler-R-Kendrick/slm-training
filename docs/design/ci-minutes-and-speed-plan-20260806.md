# CI minutes burn + speed optimization plan (2026-08-06)

**Status:** hosted GHA automatic triggers **disabled** (workflow_dispatch only).
**Claim class:** operations / cost control. Not a model quality claim.

## Immediate action (done)

| Workflow | Before | After |
| --- | --- | --- |
| `.github/workflows/ci.yml` | `push: main` + every `pull_request` | `workflow_dispatch` only |
| `.github/workflows/openwiki-update.yml` | daily cron + dispatch | `workflow_dispatch` only |

Local gates stay the authority while Actions is off:

```bash
# Pre-commit / agent stop hooks already run the changed suite:
.githooks/check-changed
python -m scripts.check_changed --changed-tests-only
python -m scripts.repo_policy
python -m scripts.verify_version_stamps --check
python -m scripts.verify_decode_invariants
python -m scripts.verify_agent_surfaces
```

Manual full GHA (if minutes become available again): Actions → CI → Run workflow.

## Why it was unaffordable

Measured on a green PR run (`31062109361`, decode closeout 2026-08-06):

| Job family | Count | ~wall each | ~bill each (ceil min) | Setup-dominated? |
| --- | ---: | ---: | ---: | --- |
| `python (0..11)` | **12** | 1.1–1.2 min | **2 min × 12 = 24** | Yes — install ~40–50s, tests ~20s |
| `lean-formal` | 1 | 2.2 min | 3 | Yes — lake build ~120s |
| `data-build` | 1 | 1.4 min | 2 | Mix — pip+npm + data build |
| `python-static` | 1 | 0.5 min | 1 | Setup ~17s |
| `lean` | 1 | 0.5 min | 1 | lean-action ~24s |
| **Total / run** | **16 jobs** | | **~31 billed minutes** | |

Notes:

- GitHub bills **per job**, rounded **up to the next minute**. A 65s job = 2 minutes.
- The 12 python shards **each** reinstall the world (CPU torch wheel, pip packages, often npm). Most of the bill is **duplicated setup**, not test execution.
- Every PR force-push / rebase / base retarget starts another full matrix. One stacked-PR closeout easily burned **5–10×31 ≈ 150–300 minutes** in a single session.
- Sample of recent history: **100 consecutive successful API pages of runs were all named `CI`** — the workflow is the dominant consumer.

Root cost formula:

```text
billable ≈ runs × (12 × ceil(setup_s + test_s) + ceil(lean_formal) + ceil(data_build) + …)
```

## What is slow (ranked)

### 1. Fan-out that multiplies setup (dominant)

`ci.yml` matrix `test-shard: [0..11]` was grown so broad diffs fit under
`MAX_RUN_MINUTES=3` per job. That solved wall-clock timeouts by **paying 12×
setup**. Net: tests are a minority of billed time.

Evidence from the same run: install step ~44–48s vs regression step ~19–26s on
typical python shards.

### 2. Torch install on nearly every shard

Comments in `ci.yml` already note the 3 GB pip cache restore cost ~40s, so the
workflow downloads a pinned CPU wheel instead. Even so, **every shard** that
needs torch pays download/install. The `need_torch` heuristic is inverted
awkwardly and still defaults broad diffs to torch.

### 3. lean-formal cold lake build (~2 min)

`leanprover/lean-action` with `build: true` on `src/slm_training/formal/lean`
dominates that job. Little caching benefit visible in the measured run.

### 4. data-build always-on path

Unconditional `pip install -e .` + `npm ci` + full `build_train_data` /
`build_test_data` on every CI, even when the PR only touches docs or thrash
helpers.

### 5. npm install side-work on python shards

`npm --prefix src/apps/openui_bridge ci` always starts in the install step
(background), even when node is not needed for selected tests.

### 6. Test selection breadth (`check_changed`)

Prefix tables map one script touch → large suite trees
(`tests/test_scripts/test_run_autotrain_continuous.py` alone is huge).
Duration-aware packing helps **makespan**, not **billable job count**.

### 7. Residual heavy tests (already marked slow)

Live `test_topology_apply` classes and live dsh5 preflight remain multi-minute
when selected. Default CI deselects them; local full runs still feel them.

## Optimization plan (when re-enabling Actions)

Ordered by **minutes saved per engineering hour**. Do not re-enable automatic
PR/main triggers until **P0** is done.

### P0 — Cut billed minutes by 5–10× without losing signal

1. **One python job by default**, not 12.
   - Default `test-shard-count: 1` (or 2) for PR.
   - Raise shard count only when selected weight > budget (dynamic matrix or a
     follow-up `workflow_call` job).
   - Keep `MAX_RUN_MINUTES` law; if the single job cannot finish, **narrow
     selection** (required paths / risk tiers) instead of paying 12× setup.

2. **Path-filtered workflows** (`paths` / `paths-ignore`).
   - Docs-only / `versions.json` history-only → `python-static` only.
   - Skip `lean` / `lean-formal` unless Lean trees change.
   - Skip `data-build` unless data builders, openui bridge, or dashboard DSL
     paths change.

3. **Shared install artifact or composite action once per run.**
   - Build a venv (and optional torch wheel) in one job; cache by lock hash;
     download into shards. Goal: setup <15s/shard after first hit.
   - Or: self-hosted runner with warm env (zero GHA minutes if self-hosted).

4. **Keep automatic triggers off until (1)+(2) land** on a measured sample PR.

### P1 — Make remaining jobs honest and cheap

5. **Fix `need_torch` heuristic** so torch-free suites never download torch.
6. **Conditional npm** — only when openui/dashboard/agentv paths change.
7. **Cache lake build** for lean-formal (`actions/cache` on `.lake` / toolchain).
8. **Drop duplicate tokenizer-grammar cert from every python shard** — run once
   in `python-static` (or one designated job).
9. **Required checks reconfiguration** on GitHub: when re-enabling, mark only
   the reduced job set as required (avoid stuck PRs waiting for deleted matrix
   legs).

### P2 — Test runtime (local + CI)

10. **Tiered selection in `check_changed`:**
    - L0: ruff + version stamps + decode invariants (always)
    - L1: direct test modules for changed files
    - L2: domain suites (current prefix table)
    - L3: full / slow markers (nightly or manual)
11. **Cheapen remaining `@pytest.mark.slow` suites** (topology live apply, live
    dsh5 preflight) rather than leaving them forever deselected.
12. **Staleness automation** for `test_durations_v1.json` (already has integrity
    metadata; add a weekly local job, not GHA, to refresh weights).

### P3 — Optional architecture

13. **Self-hosted runner** or free-tier alternative for the heavy matrix.
14. **Merge queue / batch CI** so stacked PR closeout does not re-run full CI on
    every intermediate force-push (only on the tip or on merge to main).
15. **OpenWiki** stays manual or moves to a machine with API keys and no GHA bill.

## Re-enable checklist

- [ ] P0 items 1–2 implemented and measured on a representative PR (report
      billed minutes before/after in this doc or a successor).
- [ ] Required status checks updated to match the new job names/counts.
- [ ] Document local-vs-CI parity: what GHA still covers that hooks do not.
- [ ] Restore `pull_request` / `push` triggers deliberately (not by accident).
- [ ] Cap concurrency and keep `cancel-in-progress: true`.

## Local-only operating mode (current)

| Gate | Owner |
| --- | --- |
| Changed tests | `.githooks/check-changed` / `scripts.check_changed` |
| Repo layout + workflow timeout adapters | `scripts.repo_policy` |
| Version stamps | `scripts.verify_version_stamps` |
| Decode invariants | `scripts.verify_decode_invariants` |
| Agent surface parity | `scripts.verify_agent_surfaces` |
| Formal Lean | local `lake` / `make -C src/leverproof_lean test` when touching proofs |
| Data builders | local `build_train_data` / `build_test_data` when touching those paths |

A PR with red hooks must not merge just because GHA is silent.

## Honesty

- Numbers above are from one successful PR run and ceiling-minute billing math.
  Account-level free quota and included minutes vary by plan.
- Disabling Actions removes a remote second opinion; local hooks are necessary
  but not identical (no Lean lake build, no full data-build on every PR).
- This is a **cost and latency** intervention, not a quality claim about models.

## Implemented: local merge gate (2026-08-09)

The documented hole — *"A PR with red hooks must not merge just because GHA
is silent"* — is now closed locally by a zero-minute merge gate:

| Surface | Role |
| --- | --- |
| `scripts/verify_merge_ready.py` | Single local gate command mirroring the dormant `python-static` CI job plus the changed-test fan-out. `--fast` = static only; `--json` = machine-readable summary; per-step wall budget defaults to `slm_training.levers.MAX_RUN_MINUTES` (a timed out step is `timeout` = failure). |
| `.githooks/pre-push` | Runs `python3 -m scripts.verify_merge_ready --fast` on every push (`core.hooksPath=.githooks`). Escape hatch: `SLM_SKIP_PREPUSH=1` skips but prints a loud multi-line UNVERIFIED warning. |
| `scripts/report_merge_readiness.sh` | Read-only one-screen reporter over `verify_merge_ready --json`, for agents' status checks. |
| `.grok/workflows/unblock-in-review.rhai` | Workers must run the **full** `python3 -m scripts.verify_merge_ready` and get exit 0 before any squash-merge. |
| `tests/test_scripts/test_merge_ready_gate.py` | Certifies the gate embeds every `python-static` invocation in workflow order (parity by construction), the failure/skip semantics, and both shell surfaces. |

Exact command list mirrored from `.github/workflows/ci.yml` `python-static`
(same order), then the `python` job's regression fan-out:

```bash
python -m scripts.repo_policy
python -m scripts.verify_decode_invariants
python -m scripts.verify_agent_surfaces
python -m scripts.verify_ownership_map
python -m scripts.extract_test_cases            # read-only sweep (no --write)
python -m scripts.refresh_test_cases --check --changed   # local analogue of CI's --base-ref form
ruff check .
python -m compileall -q src scripts tests
python -m scripts.verify_checkpoint_references --check
python -m scripts.verify_version_stamps --check
python -m scripts.check_changed --changed-tests-only     # skipped by --fast
```

Failure behavior (chosen and documented in the script): every static step
always runs even after a failure — each is seconds-cheap, so one pass
collects every red step instead of one rerun per fix. The changed-test step
is the only expensive one and is skipped when any static step failed; the
gate exits non-zero either way.

Pre-push behavior: the hook runs the fast (static) profile so pushes stay
seconds-cheap; the **full** gate (including changed tests) is required before
merge — by agents' merge workflows and by anyone squash-merging manually.
`SLM_SKIP_PREPUSH=1` exists for emergencies only and marks the push
UNVERIFIED in loud stderr output; it never marks anything green.

**Rule: red `verify_merge_ready` ⇒ no merge. GHA silence is not approval.**
