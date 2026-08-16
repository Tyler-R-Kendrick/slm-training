# Self-healing escalation layer for the continuous autotrain loop

**Status:** delivered on branch `claude/hill-climbing-self-healing-24taaw`.
Parent analysis:
[`harness-evolution-architecture-review-20260809.md`](harness-evolution-architecture-review-20260809.md)
(RC4: "engineered never to stop, therefore never to conclude") and
[`autotrain-self-heal-pipeline-20260803.md`](autotrain-self-heal-pipeline-20260803.md)
(the in-driver soft-heal owner). This layer was designed through an explicit
adversarial-critique + rubber-duck pass; the skeptic dispositions below are
binding design constraints, and the two rejected proposals are recorded as
closed approaches (I14: a rejected experiment closes an approach, never a
goal).

## Problem

The loop is honest about *what* blocked it — typed `AutotrainActionV1` kinds,
blocker fingerprints, content-bound receipts where `status="blocked"` never
satisfies an action — but had **no executor for the blocked class**:

- **FP1** — the supervisor answered every `hard_pending` blocker with a blind
  fixed 30 s sleep, forever (`run_autotrain_supervisor.py`). No attempt
  counting, no dispatch, no escalation artifact.
- **FP2** — the most common true harness crash (missing AgentV / JS bridge
  installs) has a documented deterministic fix (the `npm ci` triple in
  `continuous.md`) that was never attempted without a human-opened agent
  session.
- **FP5** — `foreign_dirty_tree` on a dedicated loop worktree (almost always
  crashed-agent droppings) hard-blocked with no reversible quarantine path.
- **FP7** — escalation state lived in a truncated `state.json` string + logs;
  agents arriving later re-derived the diagnosis from scratch, and the
  campaign-bound `blocker_fingerprint` minted a fresh identity every cycle so
  attempt counts never accumulated.
- **FP11** — `conclusions.conclude()` (WP-4 family closure) had a read-side
  enforcement gate (`concluded_family` preflight) but **no production
  caller**: `closed_approaches.v1.json` stayed empty and dead hypothesis
  families kept consuming cycles.

## Law

```text
The legal verbs for a blocked loop are: repair with proof, conclude, or
escalate with a typed record. Anything that acks, waives, or downgrades a
hard action from inside the loop is RC4 wearing a receipt.
```

"Repair with proof" transplants the revmath self-healing idiom
(`harnesses/reasoning/revmath/self_healing.py`): frozen identity, bounded
budgets, cycle detection over content signatures, and verify-before-commit —
`healed` is decided by re-running the original failing probe, never by the
repair steps succeeding.

## Delivered mechanisms

### L1 — bounded heal-playbook runner (`slm_training.autoresearch.heal`)

Preflight-idiom discovery: every module under `heal/playbooks/` exposing a
module-level `PLAYBOOK` attaches; absence is never an error; a crashing
playbook yields a failed receipt, never an exception and never a claimed heal.
Runner laws (enforced centrally, tested in
`tests/test_autoresearch/test_heal_playbooks.py`):

| Law | Mechanism |
| --- | --- |
| Frozen identity | Steps write only under per-step `writes_allowed` prefixes; `reject_forbidden_heal` fail-closes any allowlist overlapping `src/`, `scripts/`, `tests/`, `docs/design/`, skills, or `versions.json`; post-step porcelain diff yields `refused_scope` |
| Verify decides | `outcome="healed"` iff the plan's verify probe (the original failing resolution, e.g. `node -e require.resolve(<missing module>)`) exits 0 |
| Kill is not evidence | A wall-capped step yields `step_timeout` — neither heal nor counted failure — and escalates immediately with zero retries (`MAX_RUN_MINUTES` law) |
| Cycle detection | `plan_sha256` is the equivalent-signature; an identical plan never re-executes (`cycle_detected`) |
| Budget | Attempts per cross-campaign fingerprint capped (default 2) → `budget_exhausted` → escalation |
| No acks | The heal package cannot construct action receipts (source-level test) |

Shipped playbooks:

- **`npm_bridges/v1`** (class `environment`) — the documented `npm ci`
  repair in `src/apps/openui_bridge`, `src/apps/design_md_bridge`, and the
  repo root with `NODE_OPTIONS` cleared; verify probes the missing JS module
  named in the crash reason (fallback `@agentv/core`).
- **`quarantine_dirt/v1`** (class `dirty_tree`) — double-guarded reversible
  stash: refuses without an operator-placed
  `loops/<loop_id>/WORKTREE_DEDICATED` sentinel matching the worktree's
  resolved git dir, and during any in-progress merge / rebase / cherry-pick.
  The stash commit SHA (stable, unlike `stash@{0}`) plus the restore command
  are cross-linked into the escalation record; no playbook ever drops a
  stash.

### L2 — escalation ledger + backoff governor (`heal/escalation.py`)

`loops/<loop_id>/escalations.jsonl`, append-only, event-sourced (latest row
per fingerprint wins, like `action_receipts.jsonl`). The fingerprint is
**cross-campaign** — `sha256(kind + normalized reason)` with hashes,
timestamps, and cycle numbers stripped — so a durable blocker keeps one
identity and attempt budgets actually accumulate (the campaign-bound
`state.json` fingerprint never dedupped). Records carry `blocker_class`,
`owner_skill`, `needed_authority`, seen/attempt counts, and the governed
backoff `min(30 · 2^(seen−1), 3600)` s. The supervisor sleeps the max open
backoff instead of a blind constant.

**Advisory-only, by construction and by test:** the ledger schedules and
routes; it exposes no API that can acknowledge, soften, or expire a hard
action. `evidence-brief` and arriving agents read it as the machine-readable
diagnosis surface.

### Emission-time rewrite (`SELF_HEAL_ENV_REPAIR`, driver)

Receipt evidence for `repair_harness` legally requires a git commit after the
campaign, and continuous law forbids faking repair commits — so a heal can
never "ack" the action. Instead the driver follows its own
`SELF_HEAL_THRASH_TIMEOUT_REPAIR` precedent: a pending `repair_harness` whose
reason classifies as **environment-incomplete** (`heal.classify` — the
single marker source shared with `_delivery_is_thrash_timeout_residual`) and
whose cross-campaign fingerprint has a **verified healed receipt** in
`heal_receipts.jsonl` is rewritten to `next_experiment`, with the receipt
sha in the handoff reasons. The successor replays the frozen arm on the
restored environment. Code-class crashes (repo-internal import errors —
`npm ci` cannot fix those) keep the whole handoff hard even when a spurious
healed receipt exists (`tests/test_scripts/test_env_repair_rewrite.py`).

### L4 — family-closure write path (WP-4 production caller)

`heal/conclusion_writer.py` maps durable evidence-store rows
(`EvidenceRecordV1`) into `ConclusionEvidenceRecord` — conservatively:
`decidable` only with a finite p-value, so a measurement that could not have
rejected anything can never help close a family — and calls the existing
append-only, idempotent, policy-thresholded `conclusions.conclude()`. The
supervisor runs it fail-soft after every cycle; newly appended closures are
committed through the continuous-closeout path
(`closed_approaches.v1.json` is registered as driver-committable machine
science, like continuous results docs). The read-side `concluded_family`
preflight now polices a ledger that actually fills.

### Blocker classes (`heal/classify.py`)

`environment` (playbook-eligible), `code` (owner skill, stays hard),
`formal_infra` / `formal_contradiction` (both routed to
`improve-lean-optimums`; contradictions are evidence and are never
playbook-eligible), `delivery` (sdlc), `dirty_tree` (guarded quarantine),
`data` (driver-owned local rebuild), `authority` (human forever). Ambiguity
fails toward `code` / `formal_contradiction`: the cost of a wrong "stays
hard" is a waiting agent; the cost of a wrong "heal it" is an
install-verify-fail thrash loop.

## Skeptic dispositions (binding)

| Proposal | Verdict | Consequence in this delivery |
| --- | --- | --- |
| Escalation ledger | Approved with guards | Cross-campaign fingerprints; advisory-only (no downgrade API, tested) |
| Env-repair playbook | Approved, redesigned | Emission-time rewrite, never receipt ack; original-probe verification; attempt caps; timeout ⇒ escalate with zero retries |
| Formal triage auto-repair | Deferred | Only *classification* shipped (`formal_infra` vs `formal_contradiction` routing); auto-reverification needs certificate-replay evidence under the run cap first (skeptic O9) |
| Foreign-dirt quarantine | Approved with guards | Operator sentinel + git-dir match + mid-operation refusal + stash-SHA receipt |
| **Delivery decoupling (`deliver_stack` → debt ledger)** | **Rejected (closed approach)** | Unbounded delivery debt poisons successor provenance (`integration_commit` lineage) and deletes the loop's last human review gate. Successor approach if ever needed: bounded WIP ≤ 1 undelivered positive layer with the merged-into-origin/main receipt rule unchanged |
| **`stop_campaign` family scoping** | **Rejected (closed approach)** | Contradicts continuous law ("ordinary training waits until repaired"); requires blast-radius evidence the formal layer cannot yet produce (RC5). A theorem contradiction correctly stops ordinary training — that stop *is* the loop's highest-value evidence. Successor approach: RC5 remediation + typed contradiction-scope verdicts first |

## What stays human, deliberately

- Paid GPU / remote jobs / HF writes (authority class `authority`).
- Genuine scientific judgment: theorem contradictions, judge/threshold
  changes (separate preregistered meta-campaign), goal changes.
- Un-quarantining stashed WIP and executing owner-skill code repairs
  (`code`, `formal_*`, `delivery` classes) — now with a typed, deduplicated,
  attempt-counted escalation record to start from instead of log archaeology.

## Operator surface

- `outputs/autoresearch/loops/<id>/escalations.jsonl` — open blockers with
  class, owner skill, needed authority, attempts, backoff.
- `outputs/autoresearch/loops/<id>/heal_receipts.jsonl` — every heal attempt
  with step results, verify result, and outcome.
- `loops/<id>/WORKTREE_DEDICATED` — operator opt-in sentinel enabling the
  quarantine playbook on exactly one worktree (content = resolved git dir).
- `python -m scripts.run_autotrain_supervisor --max-heal-attempts N
  [--no-playbooks]` — playbook dispatch is on by default; `--no-playbooks`
  keeps ledger + governed backoff only.

## Non-goals

- Weakening ship gates, receipts, multi-seed close, or decode invariants.
- Auto-acking `repair_formal`, `stop_campaign`, or `deliver_stack` — still
  forbidden, now with the environment/code distinction written into the
  marker source both consumers share.
- Faking repair commits or synthesizing evidence: every heal receipt hashes
  real command output, and `healed` without a passing verify probe is
  unrepresentable in the schema.
