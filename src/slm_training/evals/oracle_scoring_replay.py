"""Oracle scoring replay harness (SLM-260 / VSD0-01).

Feeds exact-gold outputs and controlled semantic variants into the same
production scoring path used by ``evaluate_model``, quality matrices, ship gates,
and AgentV.  The harness is intentionally fixture-only: it validates that the
judge behaves as expected on hand-authored oracle inputs, not that a model
generates them.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from slm_training.data.contract import GenerationRequest
from slm_training.dsl.parser import validate
from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.meaningful_program import binding_aware_meaningful_v2
from slm_training.evals.metric_gaming import _archetypes
from slm_training.evals.task_scoreboard import build_task_scoreboard
from slm_training.harnesses.model_build.eval_runner import (
    _contract_precision,
    _contract_recall,
    _placeholder_fidelity,
    _placeholder_fidelity_normalized,
    _placeholder_validity,
    _raw_syntax_valid,
    _reward_for_prediction,
    _tree_match,
    component_type_recall,
    meaningful_program_v1,
    structural_similarity,
)
from slm_training.versioning import build_version_stamp

SCHEMA_VERSION = "oracle_scoring_replay/v1"

VARIANT_KINDS = (
    "exact_gold",
    "canonical_roundtrip",
    "alpha_renamed_equivalent",
    "egraph_equivalent",
    "unbound_reference",
    "wrong_component_or_property_role",
    "wrong_placeholder_identity",
    "prompt_contract_omission",
    "prompt_incompatible_but_valid",
    "duplicate_or_filler_gaming",
    "unreachable_or_dead_content",
)

_ASSIGNMENT_RE = re.compile(r"(?m)^\s*(\$?[A-Za-z_][A-Za-z0-9_]*)\s*=")
_STRING_LITERAL_RE = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"')

# Per-archetype adversarial transforms.  Each must be parser/schema-valid.
_WRONG_COMPONENT_OR_PROPERTY_ROLE: dict[str, str] = {
    "card": (
        'root = Stack([header, body])\n'
        'header = CardHeader(":card.title")\n'
        'body = TextContent(":card.body")\n'
    ),
    "slider": (
        'root = Stack([s])\n'
        's = TextContent(":settings.caption")\n'
    ),
    "switch": (
        'root = Stack([s])\n'
        's = TextContent(":settings.caption")\n'
    ),
    "tabs": (
        'root = Stack([tab1])\n'
        'tab1 = Button(":tab.trigger")\n'
    ),
    "button": 'root = TextContent(":btn.action")\n',
    "callout": (
        'root = Card([title, desc])\n'
        'title = TextContent(":callout.title")\n'
        'desc = TextContent(":callout.desc")\n'
    ),
    "image_block": 'root = TextContent(":img.alt")\n',
}

_PROMPT_INCOMPATIBLE_BUT_VALID: dict[str, str] = {
    "card": (
        'root = Stack([header, body])\n'
        'header = CardHeader(":card.title")\n'
        'body = TextContent(":card.body")\n'
    ),
    "slider": (
        'root = Stack([s])\n'
        's = Button(":settings.caption")\n'
    ),
    "switch": (
        'root = Stack([s])\n'
        's = Slider(":settings.caption", "continuous", 0, 100)\n'
    ),
    "tabs": (
        'root = Stack([tab1])\n'
        'tab1 = TabItem("tab1", ":tab.trigger", [TextContent(":tab.content")])\n'
    ),
    "button": (
        'root = Stack([b])\n'
        'b = TextContent(":btn.action")\n'
    ),
    "callout": (
        'root = Card([title, desc])\n'
        'title = TextContent(":callout.title")\n'
        'desc = TextContent(":callout.desc")\n'
    ),
    "image_block": (
        'root = Stack([img])\n'
        'img = TextContent(":img.src")\n'
    ),
}

_PROMPT_CONTRACT_OMISSION: dict[str, str] = {
    "card": 'root = Card([TextContent(":card.body")])\n',
    "slider": 'root = Stack([TextContent(":settings.caption")])\n',
    "switch": 'root = Stack([TextContent(":settings.caption")])\n',
    "tabs": 'root = Tabs([TabItem("tab1", ":tab.trigger", [])])\n',
    "button": 'root = Stack([TextContent(":btn.action")])\n',
    "callout": 'root = Callout("info", ":callout.title", ":callout.title")\n',
    "image_block": 'root = Stack([TextContent(":img.alt")])\n',
}

# Duplicate/filler transforms keep the requested component but add repeated
# subtrees / repeated placeholder usage to trigger anti_gaming.
_DUPLICATE_OR_FILLER_GAMING: dict[str, str] = {
    "card": (
        'root = Card([header, body, t1, t2, t3])\n'
        'header = CardHeader(":card.title")\n'
        'body = TextContent(":card.body")\n'
        't1 = TextContent(":card.body")\n'
        't2 = TextContent(":card.body")\n'
        't3 = TextContent(":card.body")\n'
    ),
    "slider": (
        'root = Stack([s, t1, t2, t3])\n'
        's = Slider(":settings.caption", "continuous", 0, 100)\n'
        't1 = TextContent(":settings.caption")\n'
        't2 = TextContent(":settings.caption")\n'
        't3 = TextContent(":settings.caption")\n'
    ),
    "switch": (
        'root = Stack([s, t1, t2, t3])\n'
        's = SwitchItem(":settings.caption", ":settings.desc", "notifications")\n'
        't1 = TextContent(":settings.caption")\n'
        't2 = TextContent(":settings.caption")\n'
        't3 = TextContent(":settings.caption")\n'
    ),
    "tabs": (
        'root = Tabs([tab1])\n'
        'tab1 = TabItem("tab1", ":tab.trigger", [TextContent(":tab.content"), t1, t2, t3])\n'
        't1 = TextContent(":tab.content")\n'
        't2 = TextContent(":tab.content")\n'
        't3 = TextContent(":tab.content")\n'
    ),
    "button": (
        'root = Stack([b, t1, t2, t3])\n'
        'b = Button(":btn.action")\n'
        't1 = TextContent(":btn.action")\n'
        't2 = TextContent(":btn.action")\n'
        't3 = TextContent(":btn.action")\n'
    ),
    "callout": (
        'root = Stack([callout, t1, t2, t3])\n'
        'callout = Callout("info", ":callout.title", ":callout.desc")\n'
        't1 = TextContent(":callout.desc")\n'
        't2 = TextContent(":callout.desc")\n'
        't3 = TextContent(":callout.desc")\n'
    ),
    "image_block": (
        'root = Stack([img, t1, t2, t3])\n'
        'img = ImageBlock(":img.src", ":img.alt")\n'
        't1 = TextContent(":img.alt")\n'
        't2 = TextContent(":img.alt")\n'
        't3 = TextContent(":img.alt")\n'
    ),
}

# Unused assignment referencing an existing slot triggers unreachable_binding.
_UNREACHABLE_OR_DEAD_CONTENT: dict[str, str] = {
    "card": (
        'root = Card([header, body])\n'
        'header = CardHeader(":card.title")\n'
        'body = TextContent(":card.body")\n'
        'dead = TextContent(":card.title")\n'
    ),
    "slider": (
        'root = Stack([s])\n'
        's = Slider(":settings.caption", "continuous", 0, 100)\n'
        'dead = TextContent(":settings.caption")\n'
    ),
    "switch": (
        'root = Stack([s])\n'
        's = SwitchItem(":settings.caption", ":settings.desc", "notifications")\n'
        'dead = TextContent(":settings.caption")\n'
    ),
    "tabs": (
        'root = Tabs([tab1])\n'
        'tab1 = TabItem("tab1", ":tab.trigger", [TextContent(":tab.content")])\n'
        'dead = TextContent(":tab.content")\n'
    ),
    "button": (
        'root = Button(":btn.action")\n'
        'dead = TextContent(":btn.action")\n'
    ),
    "callout": (
        'root = Callout("info", ":callout.title", ":callout.desc")\n'
        'dead = TextContent(":callout.title")\n'
    ),
    "image_block": (
        'root = ImageBlock(":img.src", ":img.alt")\n'
        'dead = TextContent(":img.src")\n'
    ),
}


@dataclass(frozen=True)
class OracleScoringReplayV1:
    """One oracle replay row: a gold/variant pair plus its expected verdict."""

    row_id: str
    suite: str
    record_id: str
    variant_kind: str
    expected_verdict: bool | None
    gold_openui: str
    pred_openui: str
    prompt: str
    slot_contract: tuple[str, ...] = ()
    generation_request: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    transform: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["slot_contract"] = list(self.slot_contract)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OracleScoringReplayV1:
        return cls(
            row_id=str(data["row_id"]),
            suite=str(data["suite"]),
            record_id=str(data["record_id"]),
            variant_kind=str(data["variant_kind"]),
            expected_verdict=data.get("expected_verdict"),
            gold_openui=str(data["gold_openui"]),
            pred_openui=str(data["pred_openui"]),
            prompt=str(data["prompt"]),
            slot_contract=tuple(str(s) for s in data.get("slot_contract") or ()),
            generation_request=dict(data.get("generation_request") or {}),
            notes=str(data.get("notes", "")),
            transform=str(data.get("transform", "")),
        )


def _record_sha256(record: ExampleRecord) -> str:
    payload = json.dumps(
        record.to_dict(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def score_prediction(
    record: ExampleRecord,
    pred: str,
    request: GenerationRequest | None = None,
) -> dict[str, Any]:
    """Score ``pred`` against ``record`` using the production eval path.

    Mirrors ``eval_runner._score_one`` for document records but omits
    plugin-only topology evidence fields.
    """
    if request is None:
        request = GenerationRequest.from_record(record)

    ok, error, serialized = meaningful_program_v1(pred, gold=record)
    scored_pred = serialized or pred
    semantic_report = binding_aware_meaningful_v2(
        pred, record=record, request=request
    )

    exact: float | None
    try:
        exact = _tree_match(scored_pred, record.openui)
    except Exception:  # noqa: BLE001 — harness-side match failure, not model quality
        exact = None

    struct = structural_similarity(scored_pred, record.openui)
    tree_edit = struct

    reward: float | None
    try:
        reward = _reward_for_prediction(scored_pred, record)
    except Exception:  # noqa: BLE001 — reward harness failure, not model quality
        reward = None

    return {
        "parse_ok": ok,
        "meaningful_program_v1": ok,
        "binding_aware_meaningful_v2": semantic_report.verdict,
        "semantic_meaning_report_v2": semantic_report.to_dict(),
        "syntax_parse_valid": _raw_syntax_valid(scored_pred),
        "raw_syntax_valid": _raw_syntax_valid(scored_pred),
        "error": error,
        "placeholder_fidelity": _placeholder_fidelity(scored_pred, record),
        "placeholder_fidelity_normalized": _placeholder_fidelity_normalized(
            scored_pred, record
        ),
        "placeholder_validity": _placeholder_validity(scored_pred, record),
        "contract_precision": _contract_precision(scored_pred, record),
        "contract_recall": _contract_recall(scored_pred, record),
        "exact_match": exact,
        "structural_similarity": struct,
        "tree_edit_similarity": tree_edit,
        "component_type_recall": component_type_recall(scored_pred, record.openui),
        "reward_score": reward,
        "prediction": pred,
        "prediction_sha256": hashlib.sha256(pred.encode("utf-8")).hexdigest(),
        "generation_request": request.to_dict(),
        "source_record_sha256": _record_sha256(record),
        "serialized": serialized,
    }


_EXCLUDED_ARCHETYPE_IDS = frozenset({"button"})


def build_fixture_records() -> list[ExampleRecord]:
    """Return ExampleRecord fixtures matching ``metric_gaming._archetypes``.

    The ``button`` archetype is excluded because its exact-gold positive
    currently fails ``binding_aware_meaningful_v2`` with
    ``placeholder_semantic_role_mismatch`` on this branch.  Keeping it out of
    the oracle fixture set keeps every positive oracle row a genuine positive.
    """
    records: list[ExampleRecord] = []
    for arch in _archetypes():
        if arch["id"] in _EXCLUDED_ARCHETYPE_IDS:
            continue
        slots = tuple(str(s) for s in arch.get("slot_contract", ()))
        records.append(
            ExampleRecord(
                id=str(arch["id"]),
                prompt=str(arch["prompt"]),
                openui=str(arch["positive"]),
                placeholders=list(slots),
                split="adversarial",
                source="oracle_scoring_replay_fixture",
            )
        )
    return records


def _fresh_name(used: set[str]) -> str:
    for char in "abcdefghijklmnopqrstuvwxyz":
        if char not in used:
            return char
    idx = 0
    while f"x{idx}" in used:
        idx += 1
    return f"x{idx}"


def _replace_idents(source: str, mapping: dict[str, str]) -> str:
    """Replace whole identifiers using ``mapping``, never inside string literals."""
    if not mapping:
        return source

    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)) + r")\b"
    )

    def sub_segment(segment: str) -> str:
        return pattern.sub(lambda m: mapping[m.group(0)], segment)

    out: list[str] = []
    last = 0
    for match in _STRING_LITERAL_RE.finditer(source):
        out.append(sub_segment(source[last : match.start()]))
        out.append(match.group(0))
        last = match.end()
    out.append(sub_segment(source[last:]))
    return "".join(out)


def _alpha_rename(source: str) -> str:
    """Rename every non-root assignment identifier consistently."""
    names = [
        match.group(1)
        for match in _ASSIGNMENT_RE.finditer(source)
        if match.group(1) != "root"
    ]
    used: set[str] = {"root"}
    mapping: dict[str, str] = {}
    for name in dict.fromkeys(names):
        if name in mapping:
            continue
        fresh = _fresh_name(used)
        mapping[name] = fresh
        used.add(fresh)
    return _replace_idents(source, mapping)


def _egraph_reverse(source: str) -> str:
    """Reverse the order of independent non-root top-level statements.

    ``root`` and any non-assignment lines are kept in their original relative
    order at the top; the remaining assignment definitions are reversed.  OpenUI
    state references support forward references, so this is safe for the fixture
    set where non-root definitions do not depend on each other.
    """
    lines = source.splitlines()
    prefix: list[str] = []
    assignments: list[tuple[str, str, str]] = []  # (indent, name, rhs)
    for line in lines:
        match = re.match(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$", line)
        if match and match.group(2) != "root":
            assignments.append((match.group(1), match.group(2), match.group(3)))
        else:
            prefix.append(line)
    reversed_assignments = [
        f"{indent}{name} ={rhs}" for indent, name, rhs in reversed(assignments)
    ]
    return "\n".join(prefix + reversed_assignments) + "\n"


def _first_placeholder(source: str) -> str | None:
    from slm_training.dsl.placeholders import extract_placeholders

    placeholders = extract_placeholders(source)
    return placeholders[0] if placeholders else None


def _wrong_placeholder_identity(source: str, slot_contract: tuple[str, ...]) -> str:
    """Replace the first inventory placeholder occurrence with an alien slot."""
    target = _first_placeholder(source)
    if target is None:
        # Fallback: append an obviously invalid usage.
        return source.rstrip("\n") + '\nwrong = TextContent(":wrong.slot")\n'
    return source.replace(target, ":wrong.slot", 1)


def _unbound_reference(source: str) -> str:
    """Add a statement that references an undefined identifier."""
    return source.rstrip("\n") + '\nunbound_ref = undefined_identifier\n'


def _apply_variant(record: ExampleRecord, variant_kind: str) -> tuple[str, str]:
    """Return (pred_openui, transform_description) for ``variant_kind``."""
    source = record.openui
    record_id = record.id

    if variant_kind == "exact_gold":
        return source, "identity"

    if variant_kind == "canonical_roundtrip":
        return validate(source).serialized or source, "validate(source).serialized"

    if variant_kind == "alpha_renamed_equivalent":
        return _alpha_rename(source), "alpha_rename_assignment_identifiers"

    if variant_kind == "egraph_equivalent":
        return _egraph_reverse(source), "reverse_independent_non_root_statements"

    if variant_kind == "unbound_reference":
        return _unbound_reference(source), "reference_undefined_identifier"

    if variant_kind == "wrong_component_or_property_role":
        return (
            _WRONG_COMPONENT_OR_PROPERTY_ROLE[record_id],
            "replace_requested_component_with_alternative",
        )

    if variant_kind == "wrong_placeholder_identity":
        return (
            _wrong_placeholder_identity(source, tuple(record.placeholders)),
            "replace_inventory_placeholder_with_out_of_inventory_slot",
        )

    if variant_kind == "prompt_contract_omission":
        return (
            _PROMPT_CONTRACT_OMISSION[record_id],
            "omit_one_required_placeholder_usage",
        )

    if variant_kind == "prompt_incompatible_but_valid":
        return (
            _PROMPT_INCOMPATIBLE_BUT_VALID[record_id],
            "swap_requested_component_for_another_valid_type",
        )

    if variant_kind == "duplicate_or_filler_gaming":
        return (
            _DUPLICATE_OR_FILLER_GAMING[record_id],
            "add_duplicate_subtrees_or_spammed_placeholders",
        )

    if variant_kind == "unreachable_or_dead_content":
        return (
            _UNREACHABLE_OR_DEAD_CONTENT[record_id],
            "add_unused_assignment_referencing_existing_slot",
        )

    raise ValueError(f"unknown variant kind {variant_kind!r}")


def build_variant_rows(
    records: Iterable[ExampleRecord],
    suite: str = "oracle_replay",
) -> list[OracleScoringReplayV1]:
    """Emit one oracle replay row per variant kind per fixture record."""
    rows: list[OracleScoringReplayV1] = []
    for record in records:
        request = GenerationRequest.from_record(record)
        for variant_kind in VARIANT_KINDS:
            pred_openui, transform = _apply_variant(record, variant_kind)
            expected_verdict = variant_kind in {
                "exact_gold",
                "canonical_roundtrip",
                "alpha_renamed_equivalent",
                "egraph_equivalent",
            }
            rows.append(
                OracleScoringReplayV1(
                    row_id=f"{suite}/{record.id}/{variant_kind}",
                    suite=suite,
                    record_id=record.id,
                    variant_kind=variant_kind,
                    expected_verdict=expected_verdict,
                    gold_openui=record.openui,
                    pred_openui=pred_openui,
                    prompt=record.prompt,
                    slot_contract=tuple(record.placeholders),
                    generation_request=request.to_dict(),
                    notes=(
                        f"{'positive' if expected_verdict else 'negative'} oracle "
                        f"variant for {record.id}"
                    ),
                    transform=transform,
                )
            )
    return rows


def score_rows(
    rows: Iterable[OracleScoringReplayV1],
) -> list[dict[str, Any]]:
    """Score each replay row and return rows paired with production details."""
    scored: list[dict[str, Any]] = []
    for row in rows:
        record = ExampleRecord(
            id=row.record_id,
            prompt=row.prompt,
            openui=row.gold_openui,
            placeholders=list(row.slot_contract),
            split="adversarial",
            source="oracle_scoring_replay_fixture",
        )
        request = GenerationRequest.from_dict(row.generation_request)
        detail = score_prediction(record, row.pred_openui, request=request)
        task_case = {
            "id": row.record_id,
            "task": "document",
            "gold": row.gold_openui,
            "prediction": detail["serialized"] or row.pred_openui,
            "abstraction_level": None,
            "prediction_evidence": {},
            "target_kind": "document",
            "target_category": None,
            "accepted_outputs": [],
        }
        scored.append(
            {
                "row": row.to_dict(),
                "detail": detail,
                "task_case": task_case,
            }
        )
    return scored


def build_replay_manifest(
    rows: Iterable[OracleScoringReplayV1],
    scored_rows: Iterable[dict[str, Any]],
    suite: str = "oracle_replay",
) -> dict[str, Any]:
    """Aggregate oracle replay results into a version-stamped manifest."""
    row_list = list(rows)
    scored_list = list(scored_rows)
    n = len(row_list)

    variant_counts: dict[str, int] = {}
    for row in row_list:
        variant_counts[row.variant_kind] = variant_counts.get(row.variant_kind, 0) + 1

    parse_ok_values = [float(r["detail"]["parse_ok"]) for r in scored_list]
    v1_values = [float(r["detail"]["meaningful_program_v1"]) for r in scored_list]
    v2_values = [
        float(r["detail"]["binding_aware_meaningful_v2"]) for r in scored_list
    ]
    reward_values = [
        float(r["detail"]["reward_score"])
        for r in scored_list
        if r["detail"].get("reward_score") is not None
    ]

    task_cases = [r["task_case"] for r in scored_list]

    return {
        "schema_version": SCHEMA_VERSION,
        "suite": suite,
        "n": n,
        "variant_counts": variant_counts,
        "mean_parse_ok": sum(parse_ok_values) / len(parse_ok_values) if parse_ok_values else None,
        "meaningful_program_v1_rate": sum(v1_values) / len(v1_values) if v1_values else None,
        "binding_aware_meaningful_v2_rate_strict": (
            sum(v2_values) / len(v2_values) if v2_values else None
        ),
        "reward_score": (
            sum(reward_values) / len(reward_values) if reward_values else None
        ),
        "task_scoreboard": build_task_scoreboard(task_cases),
        "version_stamp": build_version_stamp(
            "evals.scoring", "harness.oracle_scoring_replay"
        ),
        "details": scored_list,
    }


__all__ = [
    "SCHEMA_VERSION",
    "VARIANT_KINDS",
    "OracleScoringReplayV1",
    "build_fixture_records",
    "build_replay_manifest",
    "build_variant_rows",
    "score_prediction",
    "score_rows",
]
