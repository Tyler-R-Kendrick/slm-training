#!/usr/bin/env python3
"""CI certificate for reasoning/revmath harness parity (HARN-01 / SLM-520).

Static only — parses source with :mod:`ast` and reads text, no imports. Certifies:

1. ``reasoning/revmath`` is registered under ``harnesses/reasoning/profiles.py``.
2. Default profile stays ``reasoning/g4`` (revmath remains opt-in).
3. Every parity obligation is either ``present`` with a live owner path or an
   explicit ``blocker`` with a non-empty ``blocker_issue`` (no silent gaps).
4. Self-healing ``may_modify`` / ``must_freeze`` are non-empty and disjoint.
5. Absorbed predecessor ``HARN-12`` is recorded; design doc cites the matrix.

Run: ``python -m scripts.verify_revmath_harness_parity``
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PARITY_PATH = "src/slm_training/resources/revmath_harness_parity.json"
OWNER_MAP_PATH = "src/slm_training/resources/revmath_owner_map.json"
SCHEMA = "revmath_harness_parity/v1"
REQUIRED_CATEGORIES = frozenset(
    {
        "commands",
        "schemas",
        "resources",
        "campaign_binding",
        "evidence",
        "replay",
        "bounded_execution",
        "repair",
        "versions",
        "docs",
        "agent_surfaces",
    }
)
ALLOWED_STATUSES = frozenset({"present", "blocker"})


class RevmathHarnessParityError(AssertionError):
    """A revmath harness-parity claim no longer matches the live repository."""


def _read(relative: str) -> str:
    path = ROOT / relative
    if not path.is_file():
        raise RevmathHarnessParityError(f"missing required file: {relative}")
    return path.read_text(encoding="utf-8")


def _load_parity() -> dict[str, Any]:
    return json.loads(_read(PARITY_PATH))


def _defined_names(relative: str) -> set[str]:
    tree = ast.parse(_read(relative), filename=relative)
    names: set[str] = set()

    def visit(nodes: list[ast.stmt]) -> None:
        for node in nodes:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
                if isinstance(node, ast.ClassDef):
                    visit(node.body)
            elif isinstance(node, ast.Assign):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)

    visit(tree.body)
    return names


def _literal_str_assigns(relative: str) -> dict[str, str]:
    """Top-level ``NAME = "..."`` string constants (stdlib AST only)."""
    tree = ast.parse(_read(relative), filename=relative)
    out: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            out[target.id] = node.value.value
    return out


def check_registration(doc: dict[str, Any]) -> dict[str, str]:
    module = doc["registration_module"]
    defined = _defined_names(module)
    for symbol in doc["registration_symbols"]:
        if symbol not in defined:
            raise RevmathHarnessParityError(
                f"registration module {module} missing symbol {symbol!r}"
            )
    literals = _literal_str_assigns(module)
    default_id = doc["default_profile_id"]
    profile_id = doc["profile_id"]
    if literals.get("DEFAULT_PROFILE_ID") != default_id:
        raise RevmathHarnessParityError(
            f"DEFAULT_PROFILE_ID must be {default_id!r}, "
            f"got {literals.get('DEFAULT_PROFILE_ID')!r}"
        )
    if literals.get("REVMATH_PROFILE_ID") != profile_id:
        raise RevmathHarnessParityError(
            f"REVMATH_PROFILE_ID must be {profile_id!r}, "
            f"got {literals.get('REVMATH_PROFILE_ID')!r}"
        )
    if default_id == profile_id:
        raise RevmathHarnessParityError(
            "default_profile_id must differ from revmath profile_id"
        )
    tree = ast.parse(_read(module), filename=module)
    profiles_keys: set[str] = set()
    for node in tree.body:
        value = None
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "PROFILES" for t in node.targets
        ):
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "PROFILES"
        ):
            value = node.value
        if value is None:
            continue
        if not isinstance(value, ast.Dict):
            raise RevmathHarnessParityError("PROFILES must be a dict literal")
        for key in value.keys:
            if isinstance(key, ast.Name) and key.id in literals:
                profiles_keys.add(literals[key.id])
            elif isinstance(key, ast.Constant) and isinstance(key.value, str):
                profiles_keys.add(key.value)
    if default_id not in profiles_keys or profile_id not in profiles_keys:
        raise RevmathHarnessParityError(
            f"PROFILES must register both {default_id!r} and {profile_id!r}; "
            f"found {sorted(profiles_keys)}"
        )
    return {
        "module": module,
        "default_profile_id": default_id,
        "profile_id": profile_id,
    }


def check_obligations(doc: dict[str, Any]) -> dict[str, Any]:
    seen: set[str] = set()
    categories: set[str] = set()
    present: list[str] = []
    blockers: list[str] = []
    for row in doc["obligations"]:
        oid = row["id"]
        if oid in seen:
            raise RevmathHarnessParityError(f"duplicate obligations[].id: {oid!r}")
        seen.add(oid)
        category = row["category"]
        categories.add(category)
        status = row["status"]
        if status not in ALLOWED_STATUSES:
            raise RevmathHarnessParityError(
                f"obligation {oid!r} has unknown status {status!r}; "
                f"expected one of {sorted(ALLOWED_STATUSES)} (no silent fallback)"
            )
        owner = row.get("required_owner")
        blocker = row.get("blocker_issue")
        if status == "present":
            if not owner:
                raise RevmathHarnessParityError(
                    f"obligation {oid!r} is present but required_owner is empty"
                )
            if not (ROOT / owner).exists():
                raise RevmathHarnessParityError(
                    f"obligation {oid!r} required_owner missing: {owner}"
                )
            if blocker:
                raise RevmathHarnessParityError(
                    f"obligation {oid!r} is present but still names blocker_issue"
                )
            present.append(oid)
        else:
            if not blocker or not str(blocker).strip():
                raise RevmathHarnessParityError(
                    f"obligation {oid!r} is a blocker without blocker_issue "
                    "(missing parity must be explicit, never silent)"
                )
            if owner is not None:
                raise RevmathHarnessParityError(
                    f"obligation {oid!r} is a blocker but sets required_owner; "
                    "leave owner null until the blocker lands"
                )
            blockers.append(oid)
    missing_categories = REQUIRED_CATEGORIES - categories
    if missing_categories:
        raise RevmathHarnessParityError(
            "parity matrix missing required categories: "
            + ", ".join(sorted(missing_categories))
        )
    return {
        "present": present,
        "blockers": blockers,
        "categories": sorted(categories),
    }


def check_self_healing(doc: dict[str, Any]) -> dict[str, list[str]]:
    heal = doc["self_healing"]
    may_modify = list(heal["may_modify"])
    must_freeze = list(heal["must_freeze"])
    if not may_modify:
        raise RevmathHarnessParityError("self_healing.may_modify must be non-empty")
    if not must_freeze:
        raise RevmathHarnessParityError("self_healing.must_freeze must be non-empty")
    overlap = set(may_modify) & set(must_freeze)
    if overlap:
        raise RevmathHarnessParityError(
            "self_healing may_modify/must_freeze overlap: " + ", ".join(sorted(overlap))
        )
    if len(may_modify) != len(set(may_modify)) or len(must_freeze) != len(
        set(must_freeze)
    ):
        raise RevmathHarnessParityError(
            "self_healing lists must not contain duplicate entries"
        )
    return {"may_modify": may_modify, "must_freeze": must_freeze}


def check_absorbed(doc: dict[str, Any]) -> list[str]:
    keys = [row["issue_key"] for row in doc["absorbed_predecessors"]]
    if "HARN-12" not in keys:
        raise RevmathHarnessParityError(
            "absorbed_predecessors must record HARN-12 (parity-audit predecessor)"
        )
    design = _read(doc["design_doc"])
    if "HARN-12" not in design:
        raise RevmathHarnessParityError(
            f"{doc['design_doc']} must document absorbed predecessor HARN-12"
        )
    if "revmath_harness_parity" not in design and PARITY_PATH not in design:
        raise RevmathHarnessParityError(
            f"{doc['design_doc']} must cite the harness parity matrix"
        )
    owner_map = json.loads(_read(OWNER_MAP_PATH))
    if owner_map.get("profile_id") != doc["profile_id"]:
        raise RevmathHarnessParityError(
            f"owner map profile_id {owner_map.get('profile_id')!r} != "
            f"parity profile_id {doc['profile_id']!r}"
        )
    return keys


def certify() -> dict[str, Any]:
    doc = _load_parity()
    if doc.get("schema") != SCHEMA:
        raise RevmathHarnessParityError(
            f"{PARITY_PATH} declares schema {doc.get('schema')!r}, expected {SCHEMA!r}"
        )
    return {
        "schema": doc["schema"],
        "linear_id": doc.get("linear_id"),
        "issue_key": doc.get("issue_key"),
        "registration": check_registration(doc),
        "obligations": check_obligations(doc),
        "self_healing": check_self_healing(doc),
        "absorbed_predecessors": check_absorbed(doc),
    }


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        print(json.dumps(certify(), indent=2, sort_keys=True))
    except (RevmathHarnessParityError, KeyError, TypeError) as exc:
        print(f"revmath-harness-parity regression: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
