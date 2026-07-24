"""SLM-312 (LAR2-04): seedward and on-policy state sources for tree-edit repair.

Builds immutable, content-addressed training-state snapshots from three
sources over the SLM-305 extended edit space, labels every state with the
SLM-308 bounded distance oracle and an SLM-305 legal supervised edit, and
mixes them into matched ``{gold_only, seedward, on_policy, mixed}`` arms:

- ``gold_only`` — the existing near-gold corruption chain (``sample_mutation``
  forward noise; the inverse edit is the supervised repair target).
- ``seedward`` — OFFLINE valid intermediates from the minimal decode seed,
  reached by greedy oracle-guided edits TOWARD each gold target (SLM-308
  distance strictly decreasing). These cover the seed side of the
  distribution that gold corruption never visits.
- ``on_policy`` — immutable snapshots of actual seed-originated beam
  trajectories (``_decode_one`` states/proposals telemetry): visited states,
  final states, and verifier-failure-cone states, each oracle-labeled.

Honesty invariants:

1. Every row carries explicit provenance (parent checkpoint, source commit,
   rollout config, prompt/suite hashes, gold-visibility policy). On-policy
   ROLLOUTS are gold-blind; oracle LABELS read gold at training time only —
   the policy string says exactly which.
2. Leak guard is fail-closed: any AST-fingerprint overlap between a train row
   and a held-out evaluation state raises :class:`LeakageError`.
3. Snapshots are content-addressed (per-row sha256 + manifest sha256); a
   tampered row or manifest fails :func:`read_snapshot` with
   :class:`SnapshotIntegrityError`.
4. UNKNOWN / unbounded oracle labels are excluded from value supervision,
   never coerced (same rule as SLM-308).

This module is torch-free at import time; the loss helper imports torch
lazily so acquisition/guard tests stay cheap.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.experiments.slm299_edit_reachability import (
    _canonical_key,
)
from slm_training.harnesses.experiments.slm308_distance_oracle import (
    DistanceKind,
    DistanceLabel,
    distance_to_target,
    effective_distance,
)
from slm_training.models.tree_edit_diffusion import (
    ACTION_ADD,
    ACTION_ADD_CONTAINER,
    ACTION_BIND_PLACEHOLDER,
    ACTION_INSERT_STATEMENT,
    ACTION_INSERT_SUBTREE,
    ACTION_NAMES,
    ACTION_REMOVE,
    ACTION_REMOVE_CONTAINER,
    ACTION_REPLACE,
    ACTION_REPLACE_STATEMENT,
    ACTION_REPLACE_SUBTREE,
    CONTAINER_RESTS,
    Edit,
    Statement,
    TreeEditSpace,
    V05_TEMPLATES,
    MAX_SLOTS,
    parse_statements,
    render_statements,
)

EXPERIMENT_ID = "slm312-state-sources"
SNAPSHOT_SCHEMA_VERSION = 1

SOURCE_GOLD_ONLY = "gold_only"
SOURCE_SEEDWARD = "seedward"
SOURCE_ON_POLICY = "on_policy"
SOURCES = (SOURCE_GOLD_ONLY, SOURCE_SEEDWARD, SOURCE_ON_POLICY)

# Gold-visibility policies (per row; the on-policy ROLLOUT is gold-blind —
# decode never reads gold — while oracle labels read gold at training time
# only, so the policy string states both phases explicitly).
GOLD_VISIBILITY_ACQUISITION_AND_LABELS = "acquisition_and_labels"
GOLD_VISIBILITY_LABELS_ONLY = "rollout_gold_blind_labels_gold_visible"

# Oracle budgets for state-source labels (same shallow budgets as SLM-308
# training-time labels; UNKNOWN is excluded downstream, never coerced).
ORACLE_MAX_DEPTH = 8
ORACLE_NODE_BUDGET = 8

# PREREGISTERED mixture arms (locked before any training; deviations are
# append-only and exploratory). Weights are per-source row-mixture weights.
PREREGISTERED_ARMS: dict[str, dict[str, float]] = {
    "gold_only": {SOURCE_GOLD_ONLY: 1.0},
    "seedward": {SOURCE_SEEDWARD: 1.0},
    "on_policy": {SOURCE_ON_POLICY: 1.0},
    "mixed": {SOURCE_GOLD_ONLY: 0.34, SOURCE_SEEDWARD: 0.33, SOURCE_ON_POLICY: 0.33},
}

SEED_SOURCE = 'root = Stack([], "column")'


class LeakageError(RuntimeError):
    """A train row overlaps a held-out evaluation state (fail-closed)."""


class SnapshotIntegrityError(RuntimeError):
    """A content-addressed snapshot row or manifest failed verification."""


# --------------------------------------------------------------------------- #
# Hashing helpers
# --------------------------------------------------------------------------- #


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _digest(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


def ast_fingerprint(statements: list[Statement]) -> str:
    """Alpha-invariant AST fingerprint (SLM-299 canonical key), sha256."""
    return hashlib.sha256(_canonical_key(statements).encode("utf-8")).hexdigest()


def source_fingerprint(source_text: str) -> str:
    """Fingerprint of a rendered program; raises when it does not parse."""
    statements = parse_statements(source_text)
    if statements is None:
        raise ValueError(f"statements do not parse: {source_text!r}")
    return ast_fingerprint(statements)


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Provenance + row schema
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StateProvenance:
    """Where one training state came from, and who was allowed to see gold."""

    parent_checkpoint: str
    source_commit: str
    rollout_config: dict[str, Any]
    prompt_hash: str
    suite_hash: str
    gold_visibility_policy: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateProvenance":
        return cls(
            parent_checkpoint=str(data["parent_checkpoint"]),
            source_commit=str(data["source_commit"]),
            rollout_config=dict(data.get("rollout_config") or {}),
            prompt_hash=str(data["prompt_hash"]),
            suite_hash=str(data["suite_hash"]),
            gold_visibility_policy=str(data["gold_visibility_policy"]),
        )


def edit_to_dict(edit: Edit) -> dict[str, Any]:
    return {
        "action": edit.action,
        "action_name": ACTION_NAMES[edit.action],
        "stmt": edit.stmt,
        "comp": edit.comp,
        "slot": edit.slot,
        "target": edit.target,
        "payload": edit.payload,
    }


def edit_from_dict(data: dict[str, Any]) -> Edit:
    return Edit(
        int(data["action"]),
        stmt=int(data.get("stmt", 0)),
        comp=int(data.get("comp", 0)),
        slot=int(data.get("slot", 0)),
        target=int(data.get("target", 0)),
        payload=int(data.get("payload", 0)),
    )


@dataclass(frozen=True)
class StateSourceRow:
    """One labeled training state from one source, immutable + addressed."""

    record_id: str
    split: str  # "train" | "held_out" (held_out rows are eval-only)
    source: str  # one of SOURCES
    kind: str  # near_gold | seedward_step | visited | final | verifier_failure_cone
    statements_source: str
    inventory: tuple[str, ...]
    supervised_edit: dict[str, Any] | None  # SLM-305 legal edit toward gold
    child_source: str | None  # apply(supervised_edit); strictly closer when set
    distance_label: dict[str, Any]  # DistanceLabel.to_dict()
    value_target: float | None  # None => exclude from value loss
    provenance: StateProvenance
    schema_version: int = SNAPSHOT_SCHEMA_VERSION

    def content(self) -> dict[str, Any]:
        data = asdict(self)
        data["inventory"] = list(self.inventory)
        data["provenance"] = self.provenance.to_dict()
        return data

    def sha256(self) -> str:
        return _digest(self.content())

    def fingerprint(self) -> str:
        return source_fingerprint(self.statements_source)

    def to_dict(self) -> dict[str, Any]:
        data = self.content()
        data["sha256"] = self.sha256()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateSourceRow":
        row = cls(
            record_id=str(data["record_id"]),
            split=str(data["split"]),
            source=str(data["source"]),
            kind=str(data["kind"]),
            statements_source=str(data["statements_source"]),
            inventory=tuple(str(x) for x in data.get("inventory") or ()),
            supervised_edit=(
                dict(data["supervised_edit"])
                if data.get("supervised_edit") is not None
                else None
            ),
            child_source=data.get("child_source"),
            distance_label=dict(data["distance_label"]),
            value_target=(
                float(data["value_target"])
                if data.get("value_target") is not None
                else None
            ),
            provenance=StateProvenance.from_dict(data["provenance"]),
            schema_version=int(data.get("schema_version", SNAPSHOT_SCHEMA_VERSION)),
        )
        expected = data.get("sha256")
        if expected is not None and expected != row.sha256():
            raise SnapshotIntegrityError(
                f"row {row.record_id}/{row.kind} sha256 mismatch: "
                f"stored {expected}, computed {row.sha256()}"
            )
        return row


def make_provenance(
    record: ExampleRecord,
    *,
    parent_checkpoint: str,
    rollout_config: dict[str, Any],
    suite_id: str,
    gold_visibility_policy: str,
    source_commit: str | None = None,
) -> StateProvenance:
    return StateProvenance(
        parent_checkpoint=parent_checkpoint,
        source_commit=source_commit if source_commit is not None else _git_commit(),
        rollout_config=dict(rollout_config),
        prompt_hash=_text_hash(record.prompt),
        suite_hash=_text_hash(suite_id),
        gold_visibility_policy=gold_visibility_policy,
    )


def _label_row(
    *,
    record: ExampleRecord,
    split: str,
    source: str,
    kind: str,
    statements: list[Statement],
    target: list[Statement],
    inventory: Sequence[str],
    supervised_edit: Edit | None,
    space: TreeEditSpace,
    provenance: StateProvenance,
    witness: int | None,
    child_source: str | None = None,
    verify_edit: bool = True,
) -> StateSourceRow:
    """Oracle-label one state.

    With ``verify_edit`` (default) the supervised edit must be SLM-305 legal
    on this state (fail-closed ``apply``); the resulting child is the
    pairwise-progress target. Gold-corruption rows instead pass the actual
    chain parent as ``child_source`` with ``verify_edit=False`` — the
    historical inverse edit is a supervision target in index space and is
    never re-applied (same semantics as ``training_loss``).
    """
    if supervised_edit is not None and verify_edit:
        child = space.apply(statements, supervised_edit, list(inventory))
        if child is None:
            raise ValueError(
                f"supervised edit {edit_to_dict(supervised_edit)} is not legal "
                f"for {render_statements(statements)!r} (fail-closed apply)"
            )
        child_source = render_statements(child)
    label = distance_to_target(
        statements,
        target,
        space=space,
        inventory=inventory,
        max_depth=ORACLE_MAX_DEPTH,
        node_budget=ORACLE_NODE_BUDGET,
        upper_bound_witness=witness,
    )
    return StateSourceRow(
        record_id=record.id,
        split=split,
        source=source,
        kind=kind,
        statements_source=render_statements(statements),
        inventory=tuple(inventory),
        supervised_edit=(
            edit_to_dict(supervised_edit) if supervised_edit is not None else None
        ),
        child_source=child_source,
        distance_label=label.to_dict(),
        value_target=label.value_target(ORACLE_MAX_DEPTH),
        provenance=provenance,
    )


# --------------------------------------------------------------------------- #
# Leak guard (fail-closed)
# --------------------------------------------------------------------------- #


def drop_leaked_rows(
    rows: Iterable[StateSourceRow],
    held_out_fingerprints: Iterable[str],
) -> tuple[list[StateSourceRow], list[dict[str, str]]]:
    """Remove TRAIN rows whose AST fingerprint collides with a held-out eval
    state, returning (kept, dropped-log). Legitimate cross-record collisions
    (alpha-invariant canonical keys make e.g. a one-leaf near-gold state of a
    train record and a seed-walk state of a held-out record identical) are
    dropped and counted — never silently kept. :func:`assert_no_leakage`
    remains the fail-closed verification on the kept set."""
    held = set(held_out_fingerprints)
    kept: list[StateSourceRow] = []
    dropped: list[dict[str, str]] = []
    for row in rows:
        if row.split == "train" and row.fingerprint() in held:
            dropped.append(
                {
                    "record_id": row.record_id,
                    "source": row.source,
                    "kind": row.kind,
                    "fingerprint": row.fingerprint(),
                    "reason": "held_out_fingerprint_collision",
                }
            )
            continue
        kept.append(row)
    return kept, dropped


def assert_no_leakage(
    rows: Iterable[StateSourceRow],
    held_out_fingerprints: Iterable[str],
) -> None:
    """Raise LeakageError when any TRAIN row's AST fingerprint appears in the
    held-out evaluation fingerprint set. Held-out rows are eval-only and are
    never train supervision, so only ``split == "train"`` rows are checked."""
    held = set(held_out_fingerprints)
    if not held:
        return
    overlaps: list[dict[str, str]] = []
    for row in rows:
        if row.split != "train":
            continue
        fp = row.fingerprint()
        if fp in held:
            overlaps.append(
                {
                    "record_id": row.record_id,
                    "source": row.source,
                    "kind": row.kind,
                    "fingerprint": fp,
                }
            )
    if overlaps:
        raise LeakageError(
            f"{len(overlaps)} train row(s) overlap held-out eval states: "
            + json.dumps(overlaps[:5], sort_keys=True)
        )


# --------------------------------------------------------------------------- #
# Child-edit enumeration (mirrors slm299's extended mode but keeps the Edit)
# --------------------------------------------------------------------------- #


def enumerate_child_edits(
    space: TreeEditSpace,
    statements: list[Statement],
    inventory: Sequence[str],
    *,
    visited: set[str] | None = None,
) -> list[tuple[list[Statement], Edit]]:
    """All one-edit successors under the deployed extended action set,
    applied through ``TreeEditSpace.apply`` (fail-closed validity), keeping
    the :class:`Edit` so it can be used as a supervision target. Mirrors
    ``slm299_edit_reachability._enumerate_children(mode="extended")``."""
    from slm_training.models.tree_edit_diffusion import (
        CONTAINER_COMPONENTS,
        LEAF_COMPONENTS,
    )

    children: list[tuple[list[Statement], Edit]] = []
    n_slots = min(len(inventory), MAX_SLOTS)
    leaf_comp_idxs = [i for i, c in enumerate(space.components) if c in LEAF_COMPONENTS]
    container_comp_idxs = [
        i for i, c in enumerate(space.components) if c in CONTAINER_COMPONENTS
    ]
    pre = None
    if visited is not None:
        pre = lambda working: _canonical_key(working) not in visited  # noqa: E731
    by_name = {s.name: s for s in statements}

    def _replace_subtree_ok(stmt: Statement) -> bool:
        if not stmt.has_list or len(stmt.children) != 1:
            return False
        leaf = by_name.get(stmt.children[0])
        return leaf is not None and not leaf.has_list

    for stmt_idx in range(len(statements)):
        stmt = statements[stmt_idx]
        for comp_idx in range(len(space.components)):
            edit = Edit(ACTION_REPLACE, stmt_idx, comp_idx)
            nxt = space.apply(statements, edit, list(inventory), pre)
            if nxt is not None:
                children.append((nxt, edit))
            for slot_idx in range(n_slots):
                edit = Edit(ACTION_ADD, stmt_idx, comp_idx, slot_idx)
                nxt = space.apply(statements, edit, list(inventory), pre)
                if nxt is not None:
                    children.append((nxt, edit))
        if stmt.has_list:
            for comp_idx in container_comp_idxs:
                for rest_idx in range(len(CONTAINER_RESTS)):
                    edit = Edit(
                        ACTION_ADD_CONTAINER, stmt_idx, comp_idx, target=rest_idx
                    )
                    nxt = space.apply(statements, edit, list(inventory), pre)
                    if nxt is not None:
                        children.append((nxt, edit))
            for comp_idx in container_comp_idxs:
                for slot_idx in range(n_slots):
                    for payload in leaf_comp_idxs:
                        for rest_idx in range(len(CONTAINER_RESTS)):
                            edit = Edit(
                                ACTION_INSERT_SUBTREE,
                                stmt_idx,
                                comp_idx,
                                slot_idx,
                                target=rest_idx,
                                payload=payload,
                            )
                            nxt = space.apply(statements, edit, list(inventory), pre)
                            if nxt is not None:
                                children.append((nxt, edit))
            if _replace_subtree_ok(stmt):
                for slot_idx in range(n_slots):
                    for payload in leaf_comp_idxs:
                        edit = Edit(
                            ACTION_REPLACE_SUBTREE,
                            stmt_idx,
                            slot=slot_idx,
                            payload=payload,
                        )
                        nxt = space.apply(statements, edit, list(inventory), pre)
                        if nxt is not None:
                            children.append((nxt, edit))
        else:
            for slot_idx in range(n_slots):
                edit = Edit(ACTION_BIND_PLACEHOLDER, stmt_idx, slot=slot_idx)
                nxt = space.apply(statements, edit, list(inventory), pre)
                if nxt is not None:
                    children.append((nxt, edit))
        edit = Edit(ACTION_REMOVE_CONTAINER, stmt_idx)
        nxt = space.apply(statements, edit, list(inventory), pre)
        if nxt is not None:
            children.append((nxt, edit))
        edit = Edit(ACTION_REMOVE, stmt_idx)
        nxt = space.apply(statements, edit, list(inventory), pre)
        if nxt is not None:
            children.append((nxt, edit))
        for payload in range(len(V05_TEMPLATES)):
            edit = Edit(ACTION_REPLACE_STATEMENT, stmt_idx, payload=payload)
            nxt = space.apply(statements, edit, list(inventory), pre)
            if nxt is not None:
                children.append((nxt, edit))
    for payload in range(len(V05_TEMPLATES)):
        edit = Edit(ACTION_INSERT_STATEMENT, payload=payload)
        nxt = space.apply(statements, edit, list(inventory), pre)
        if nxt is not None:
            children.append((nxt, edit))
    return children


def oracle_best_child(
    statements: list[Statement],
    target: list[Statement],
    space: TreeEditSpace,
    inventory: Sequence[str],
) -> tuple[Edit, list[Statement], DistanceLabel] | None:
    """Strictly-improving child with the smallest oracle distance, or None.

    Deterministic: children are enumerated in fixed order and ties keep the
    first. Returns the edit (a legal SLM-305 action by construction), the
    child statements, and the child's distance label.
    """
    current = distance_to_target(
        statements,
        target,
        space=space,
        inventory=inventory,
        max_depth=ORACLE_MAX_DEPTH,
        node_budget=ORACLE_NODE_BUDGET,
    )
    d_current = effective_distance(current)
    best: tuple[float, Edit, list[Statement], DistanceLabel] | None = None
    for child, edit in enumerate_child_edits(space, statements, inventory):
        label = distance_to_target(
            child,
            target,
            space=space,
            inventory=inventory,
            max_depth=ORACLE_MAX_DEPTH,
            node_budget=ORACLE_NODE_BUDGET,
        )
        d = effective_distance(label)
        if d is None:
            continue
        if d_current is not None and d >= d_current:
            continue
        if best is None or d < best[0]:
            best = (d, edit, child, label)
    if best is None:
        return None
    return best[1], best[2], best[3]


# --------------------------------------------------------------------------- #
# Acquisition
# --------------------------------------------------------------------------- #


def _inventory_of(record: ExampleRecord) -> list[str]:
    return [p if p.startswith(":") else f":{p}" for p in (record.placeholders or ())]


def acquire_gold_only(
    records: Sequence[ExampleRecord],
    space: TreeEditSpace,
    *,
    split: str = "train",
    k_max: int = 3,
    seed: int = 20260724,
    parent_checkpoint: str = "none",
    suite_id: str = "slm312-fixture",
) -> list[StateSourceRow]:
    """Existing near-gold corruption chain: mutate gold 1..k edits; the
    inverse edit is the supervised repair target (current training path)."""
    rows: list[StateSourceRow] = []
    for idx, record in enumerate(records):
        gold = parse_statements(record.openui or "")
        if gold is None:
            continue
        inventory = _inventory_of(record)
        provenance = make_provenance(
            record,
            parent_checkpoint=parent_checkpoint,
            rollout_config={"acquisition": "sample_mutation", "k_max": k_max},
            suite_id=suite_id,
            gold_visibility_policy=GOLD_VISIBILITY_ACQUISITION_AND_LABELS,
        )
        rng = random.Random(seed + idx)
        for k in (1, 2, 3):
            if k > k_max:
                break
            current = gold
            prev = None
            inverse: Edit | None = None
            applied = 0
            for _ in range(k):
                step = space.sample_mutation(current, inventory, rng)
                if step is None:
                    break
                prev = current
                current, inverse = step
                applied += 1
            if inverse is None:
                continue
            rows.append(
                _label_row(
                    record=record,
                    split=split,
                    source=SOURCE_GOLD_ONLY,
                    kind="near_gold",
                    statements=current,
                    target=gold,
                    inventory=inventory,
                    supervised_edit=inverse,
                    space=space,
                    provenance=provenance,
                    witness=applied,
                    # Pairwise child = the actual chain parent (one inverse
                    # step closer to gold), mirroring training_loss; the
                    # inverse edit itself is an index-space CE target and is
                    # never re-applied there, so it is not re-applied here.
                    child_source=render_statements(prev) if prev is not None else None,
                    verify_edit=False,
                )
            )
    return rows


def acquire_seedward(
    records: Sequence[ExampleRecord],
    space: TreeEditSpace,
    *,
    split: str = "train",
    seed_source: str = SEED_SOURCE,
    max_edits: int = 4,
    parent_checkpoint: str = "none",
    suite_id: str = "slm312-fixture",
) -> list[StateSourceRow]:
    """OFFLINE seedward source: from the minimal decode seed, walk greedy
    oracle-guided edits TOWARD each gold target, emitting every valid
    intermediate (the seed side of the distribution). Every step is a legal
    SLM-305 action and strictly distance-decreasing by construction."""
    rows: list[StateSourceRow] = []
    seed_statements = parse_statements(seed_source)
    if seed_statements is None:
        raise ValueError(f"seed source does not parse: {seed_source!r}")
    for record in records:
        gold = parse_statements(record.openui or "")
        if gold is None:
            continue
        inventory = _inventory_of(record)
        provenance = make_provenance(
            record,
            parent_checkpoint=parent_checkpoint,
            rollout_config={
                "acquisition": "oracle_guided_greedy",
                "seed_source": seed_source,
                "max_edits": max_edits,
            },
            suite_id=suite_id,
            gold_visibility_policy=GOLD_VISIBILITY_ACQUISITION_AND_LABELS,
        )
        current = seed_statements
        chain: list[tuple[list[Statement], Edit | None]] = []  # (state, edit to next)
        for _ in range(max_edits):
            found = oracle_best_child(current, gold, space, inventory)
            if found is None:
                break
            edit, child, _label = found
            chain.append((child, edit))
            current = child
        # Row i's supervised edit is the edit taken NEXT (toward gold); the
        # final chain state gets its own oracle-best continuation, or STOP.
        for i, (state, _took) in enumerate(chain):
            if i + 1 < len(chain):
                supervised: Edit | None = chain[i + 1][1]
            else:
                nxt = oracle_best_child(state, gold, space, inventory)
                supervised = nxt[0] if nxt is not None else None
            rows.append(
                _label_row(
                    record=record,
                    split=split,
                    source=SOURCE_SEEDWARD,
                    kind="seedward_step",
                    statements=state,
                    target=gold,
                    inventory=inventory,
                    supervised_edit=supervised,
                    space=space,
                    provenance=provenance,
                    # Greedy seed-side steps do not prove a path to gold, so
                    # there is no honest upper-bound witness here.
                    witness=None,
                )
            )
    return rows


def acquire_on_policy(
    model: Any,
    records: Sequence[ExampleRecord],
    space: TreeEditSpace,
    *,
    split: str = "train",
    parent_checkpoint: str,
    suite_id: str = "slm312-fixture",
    max_states_per_record: int = 8,
) -> list[StateSourceRow]:
    """Immutable snapshots from ACTUAL seed-originated beam trajectories.

    Runs ``model._decode_one`` per record (gold-blind rollout), then labels
    the visited/final states with the training-time oracle. Verifier-failure
    cone states (states with at least one dead proposal) are tagged so the
    failure neighborhoods are represented and are prioritized under
    ``max_states_per_record`` (oracle-guided relabeling is search-backed, so
    the per-record cap keeps acquisition cost bounded at fixture scale).
    Rows are deduped by fingerprint, keeping the richest kind tag.
    """
    rows: list[StateSourceRow] = []
    seen: set[str] = set()
    rollout_config = {
        "beam_width": model.config.beam_width,
        "expand_per_state": model.config.expand_per_state,
        "max_search_steps": model.config.max_search_steps,
        "stop_slot_accounting": model.config.stop_slot_accounting,
    }
    for record in records:
        gold = parse_statements(record.openui or "")
        if gold is None:
            continue
        inventory = _inventory_of(record)
        provenance = make_provenance(
            record,
            parent_checkpoint=parent_checkpoint,
            rollout_config=rollout_config,
            suite_id=suite_id,
            gold_visibility_policy=GOLD_VISIBILITY_LABELS_ONLY,
        )
        prompt = model._format_context(
            record.prompt, design_md=record.design_md, slot_contract=inventory
        )
        ctx, ctx_pad = model._encode_context([prompt])
        final_text, evidence = model._decode_one(ctx, ctx_pad, inventory)
        if not final_text:
            continue
        # States with at least one dead (verifier-rejected) proposal.
        failure_states = {
            (p["step"], p["beam_row"])
            for p in evidence.get("proposals", [])
            if not p.get("applicable", False)
        }
        candidates: list[tuple[str, str]] = []  # (kind, source)
        for entry in evidence.get("states", []):
            kind = (
                "verifier_failure_cone"
                if (entry["step"], entry["beam_row"]) in failure_states
                else "visited"
            )
            candidates.append((kind, entry["source"]))
        candidates.append(("final", final_text))
        # Prioritize failure-cone + final states under the per-record cap.
        priority = {"verifier_failure_cone": 0, "final": 1, "visited": 2}
        candidates.sort(key=lambda item: priority.get(item[0], 3))
        per_record = 0
        for kind, source in candidates:
            if per_record >= max_states_per_record:
                break
            statements = parse_statements(source)
            if statements is None:
                continue
            fp = ast_fingerprint(statements)
            if fp in seen:
                continue
            seen.add(fp)
            per_record += 1
            found = oracle_best_child(statements, gold, space, inventory)
            edit = found[0] if found is not None else None
            rows.append(
                _label_row(
                    record=record,
                    split=split,
                    source=SOURCE_ON_POLICY,
                    kind=kind,
                    statements=statements,
                    target=gold,
                    inventory=inventory,
                    supervised_edit=edit,
                    space=space,
                    provenance=provenance,
                    witness=None,
                )
            )
    return rows


# --------------------------------------------------------------------------- #
# Immutable content-addressed snapshots
# --------------------------------------------------------------------------- #


@dataclass
class StateSnapshotV1:
    """A frozen set of labeled state rows with a content-address manifest."""

    snapshot_id: str
    config: dict[str, Any]
    rows: list[StateSourceRow] = field(default_factory=list)

    def manifest(self) -> dict[str, Any]:
        row_hashes = [r.sha256() for r in self.rows]
        return {
            "snapshot_schema": f"slm312-state-snapshot/v{SNAPSHOT_SCHEMA_VERSION}",
            "snapshot_id": self.snapshot_id,
            "config": self.config,
            "n_rows": len(self.rows),
            "row_sha256": row_hashes,
            "manifest_sha256": _digest(
                {
                    "snapshot_id": self.snapshot_id,
                    "config": self.config,
                    "row_sha256": row_hashes,
                }
            ),
        }

    def write(self, path: Path) -> Path:
        """Write rows as JSONL plus a ``<path>.manifest.json`` sidecar."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for row in self.rows:
                fh.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        manifest_path.write_text(
            json.dumps(self.manifest(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


def read_snapshot(path: Path) -> StateSnapshotV1:
    """Load a snapshot, verifying every row hash and the manifest (tamper
    detection is fail-closed: any mismatch raises SnapshotIntegrityError)."""
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[StateSourceRow] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(StateSourceRow.from_dict(json.loads(line)))
    if [r.sha256() for r in rows] != list(manifest.get("row_sha256") or []):
        raise SnapshotIntegrityError(
            f"row hash sequence mismatch against manifest {manifest_path}"
        )
    snapshot = StateSnapshotV1(
        snapshot_id=str(manifest["snapshot_id"]),
        config=dict(manifest.get("config") or {}),
        rows=rows,
    )
    if snapshot.manifest()["manifest_sha256"] != manifest.get("manifest_sha256"):
        raise SnapshotIntegrityError(f"manifest sha256 mismatch for {manifest_path}")
    return snapshot


# --------------------------------------------------------------------------- #
# Mixture arms (predeclared weights + caps BEFORE any training)
# --------------------------------------------------------------------------- #


def build_arm_rows(
    rows: Sequence[StateSourceRow],
    arm: str,
    *,
    per_arm_budget: int,
    seed: int = 0,
) -> list[StateSourceRow]:
    """Deterministic per-source sampling under the predeclared arm weights.

    Every arm draws at most ``per_arm_budget`` rows in total; the per-source
    quota is ``round(weight * budget)`` (at least 1 for a positive weight),
    sampled without replacement with a per-arm seed so arms share budgets but
    not rows.
    """
    if arm not in PREREGISTERED_ARMS:
        raise ValueError(
            f"unknown arm {arm!r} (predeclared: {sorted(PREREGISTERED_ARMS)})"
        )
    weights = PREREGISTERED_ARMS[arm]
    train_rows = [r for r in rows if r.split == "train"]
    rng = random.Random(seed)
    selected: list[StateSourceRow] = []
    for source in sorted(weights):
        weight = weights[source]
        if weight <= 0:
            continue
        quota = max(1, round(weight * per_arm_budget))
        pool = [r for r in train_rows if r.source == source]
        # Dedupe by fingerprint within the source pool.
        seen: set[str] = set()
        unique: list[StateSourceRow] = []
        for row in pool:
            fp = row.fingerprint()
            if fp in seen:
                continue
            seen.add(fp)
            unique.append(row)
        n = min(quota, len(unique))
        selected.extend(rng.sample(unique, n) if n < len(unique) else unique)
    return selected[:per_arm_budget]


def source_coverage(rows: Sequence[StateSourceRow]) -> dict[str, Any]:
    """Per-source state-space coverage + AST-fingerprint duplicate rate."""
    per_source: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for row in rows:
        per_source.setdefault(row.source, set()).add(row.fingerprint())
        counts[row.source] = counts.get(row.source, 0) + 1
    total = len(rows)
    unique_total = len({r.fingerprint() for r in rows})
    return {
        "rows": total,
        "unique_fingerprints": unique_total,
        "duplicate_rate": (1.0 - unique_total / total) if total else 0.0,
        "per_source": {
            source: {
                "rows": counts.get(source, 0),
                "unique_fingerprints": len(fps),
            }
            for source, fps in sorted(per_source.items())
        },
    }


# --------------------------------------------------------------------------- #
# Training loss over acquired rows (mirrors training_loss mechanics)
# --------------------------------------------------------------------------- #


def state_supervision_loss(
    model: Any,
    rows: Sequence[StateSourceRow],
    record_by_id: dict[str, ExampleRecord],
) -> Any:
    """Edit-prediction + bounded-distance value loss over acquired state rows.

    Same loss terms as ``TreeEditDiffusionModel.training_loss`` (policy CE on
    the supervised edit, value MSE with UNKNOWN masked out, pairwise progress
    ranking toward the supervised child) but the states come from the arm's
    snapshot rows instead of an internally sampled gold-corruption chain —
    so the four arms share identical code and differ ONLY in state source.
    Rows without a supervised edit (e.g. a state already at/oracle-closest to
    gold) are supervised as STOP when the oracle proves distance 0, else they
    carry value supervision only.
    """
    import torch
    import torch.nn.functional as F

    from slm_training.models.tree_edit_diffusion import (
        ACTION_STOP,
        MAX_STMTS,
        pairwise_progress_loss,
    )

    prompts: list[str] = []
    states: list[str] = []
    targets: list[Edit] = []
    values: list[float] = []
    value_mask: list[bool] = []
    pair_rows: list[tuple[int, str]] = []
    skipped = 0
    n_unknown_excluded = 0
    for row in rows:
        record = record_by_id.get(row.record_id)
        if record is None:
            skipped += 1
            continue
        statements = parse_statements(row.statements_source)
        if statements is None:
            skipped += 1
            continue
        inventory = list(row.inventory)
        prompt = model._format_context(
            record.prompt, design_md=record.design_md, slot_contract=inventory
        )
        prompts.append(prompt)
        states.append(row.statements_source)
        if row.supervised_edit is not None:
            targets.append(edit_from_dict(row.supervised_edit))
        else:
            targets.append(Edit(ACTION_STOP))
        if row.value_target is None:
            values.append(0.0)
            value_mask.append(False)
            n_unknown_excluded += 1
        else:
            values.append(float(row.value_target))
            value_mask.append(True)
        if row.child_source is not None:
            # Pairwise progress: the supervised child is strictly closer by
            # construction (gold inverse edit or oracle-certified improving
            # edit); the oracle labels on the row prove comparability.
            d_parent = effective_distance(
                DistanceLabel(
                    kind=DistanceKind(row.distance_label["kind"]),
                    reason=str(row.distance_label["reason"]),
                    distance=row.distance_label.get("distance"),
                    lo=row.distance_label.get("lo"),
                    hi=row.distance_label.get("hi"),
                )
            )
            if d_parent is not None and d_parent > 0:
                pair_rows.append((len(states) - 1, row.child_source))
    if not states:
        return torch.zeros((), device=model.device_name, requires_grad=True)
    n_main = len(states)
    pair_child_rows = list(range(n_main, n_main + len(pair_rows)))
    states.extend(child for _, child in pair_rows)
    ctx, ctx_pad = model._encode_context(prompts)
    if pair_rows:
        ctx_rows = [row for row, _ in pair_rows]
        ctx = torch.cat([ctx, ctx[ctx_rows]], dim=0)
        ctx_pad = torch.cat([ctx_pad, ctx_pad[ctx_rows]], dim=0)
    out = model.policy(model._state_batch(states), model.tokenizer.pad_id, ctx, ctx_pad)
    device = model.device_name
    action_t = torch.tensor([e.action for e in targets], device=device)
    loss = F.cross_entropy(out["action"][:n_main], action_t)
    losses = {"action": float(loss.detach().cpu())}
    stmt_rows = [i for i, e in enumerate(targets) if e.action != ACTION_STOP]
    if stmt_rows:
        idx = torch.tensor(stmt_rows, device=device)
        stmt_t = torch.tensor(
            [min(targets[i].stmt, MAX_STMTS - 1) for i in stmt_rows], device=device
        )
        stmt_loss = F.cross_entropy(out["stmt"][idx], stmt_t)
        loss = loss + stmt_loss
        losses["stmt"] = float(stmt_loss.detach().cpu())
    comp_rows = [
        i
        for i, e in enumerate(targets)
        if e.action
        in {ACTION_REPLACE, ACTION_ADD, ACTION_ADD_CONTAINER, ACTION_INSERT_SUBTREE}
    ]
    if comp_rows:
        idx = torch.tensor(comp_rows, device=device)
        comp_t = torch.tensor([targets[i].comp for i in comp_rows], device=device)
        comp_loss = F.cross_entropy(out["comp"][idx], comp_t)
        loss = loss + comp_loss
        losses["comp"] = float(comp_loss.detach().cpu())
    slot_rows = [
        i
        for i, e in enumerate(targets)
        if e.action
        in {
            ACTION_ADD,
            ACTION_INSERT_SUBTREE,
            ACTION_REPLACE_SUBTREE,
            ACTION_BIND_PLACEHOLDER,
        }
    ]
    if slot_rows:
        idx = torch.tensor(slot_rows, device=device)
        slot_t = torch.tensor(
            [min(targets[i].slot, MAX_SLOTS - 1) for i in slot_rows], device=device
        )
        slot_loss = F.cross_entropy(out["slot"][idx], slot_t)
        loss = loss + slot_loss
        losses["slot"] = float(slot_loss.detach().cpu())
    value_t = torch.tensor(values, device=device, dtype=out["value"].dtype)
    mask_t = torch.tensor(value_mask, device=device, dtype=torch.bool)
    if bool(mask_t.any()):
        value_loss = F.mse_loss(out["value"][:n_main][mask_t], value_t[mask_t])
        loss = loss + value_loss
        losses["value"] = float(value_loss.detach().cpu())
    if pair_rows:
        pair_loss = pairwise_progress_loss(
            out["value"][torch.tensor([row for row, _ in pair_rows], device=device)],
            out["value"][torch.tensor(pair_child_rows, device=device)],
            margin=model.config.pairwise_progress_margin,
        )
        loss = loss + pair_loss
        losses["pairwise_progress"] = float(pair_loss.detach().cpu())
    losses["value_unknown_excluded"] = float(n_unknown_excluded)
    losses["skipped"] = float(skipped)
    losses["rows"] = float(n_main)
    model.last_training_metrics = losses
    return loss


__all__ = [
    "EXPERIMENT_ID",
    "GOLD_VISIBILITY_ACQUISITION_AND_LABELS",
    "GOLD_VISIBILITY_LABELS_ONLY",
    "LeakageError",
    "PREREGISTERED_ARMS",
    "SEED_SOURCE",
    "SOURCES",
    "SOURCE_GOLD_ONLY",
    "SOURCE_ON_POLICY",
    "SOURCE_SEEDWARD",
    "SNAPSHOT_SCHEMA_VERSION",
    "SnapshotIntegrityError",
    "StateProvenance",
    "StateSnapshotV1",
    "StateSourceRow",
    "acquire_gold_only",
    "acquire_on_policy",
    "acquire_seedward",
    "assert_no_leakage",
    "ast_fingerprint",
    "build_arm_rows",
    "drop_leaked_rows",
    "edit_from_dict",
    "edit_to_dict",
    "enumerate_child_edits",
    "make_provenance",
    "oracle_best_child",
    "read_snapshot",
    "source_coverage",
    "source_fingerprint",
    "state_supervision_loss",
]
