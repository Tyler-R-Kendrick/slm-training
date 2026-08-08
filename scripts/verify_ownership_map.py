#!/usr/bin/env python3
"""CI certificate for the repository owner/authority map (SGS-001, SLM-435).

Static only -- parses source with :mod:`ast` and reads text, no imports, so it
runs in the dependency-light static CI job. It certifies:

1. Every ``subsystems[]`` entry's ``owner_module`` exists and every listed
   ``owner_symbols`` name is actually a class/def/assignment target in that
   file (catches an owner claim rotting after a rename or delete).
2. Every ``authority_tier`` used is declared in ``authority_tiers``.
3. Every ``duplicate_subsystem_risks[]`` and ``related_overlaps[]`` reference
   resolves to a real path.
4. ``OUTPUT_CONTRACT_VERSION`` in ``dsl/language_contract.py`` matches the
   value cited by ``docs/design/symbol-only-output-contract.md`` -- the exact
   drift class this audit found and fixed (map ``known_drift`` entry
   ``output_contract_version_doc_mismatch``). Regresses loudly if either side
   changes without the other.
5. Every ``downstream_extension_map[]`` row either cites at least one real
   ``subsystems[].id`` as its extension point, or sets
   ``new_owner_justified: true`` with a non-empty ``justification``.

Run: ``python -m scripts.verify_ownership_map``
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = "src/slm_training/resources/ownership_map.json"
LANGUAGE_CONTRACT = "src/slm_training/dsl/language_contract.py"
OUTPUT_CONTRACT_DOC = "docs/design/symbol-only-output-contract.md"


class OwnershipMapError(AssertionError):
    """An ownership-map claim no longer matches the live repository."""


def _read(relative: str) -> str:
    path = ROOT / relative
    if not path.is_file():
        raise OwnershipMapError(f"missing required file: {relative}")
    return path.read_text(encoding="utf-8")


def _load_map() -> dict[str, Any]:
    return json.loads(_read(MAP_PATH))


def _defined_names(relative: str) -> set[str]:
    """Top-level and class-level class/def/assignment names in a module."""
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


def check_subsystems(doc: dict[str, Any]) -> list[str]:
    tiers = set(doc["authority_tiers"])
    checked: list[str] = []
    for entry in doc["subsystems"]:
        sid = entry["id"]
        module = entry["owner_module"]
        if entry["authority_tier"] not in tiers:
            raise OwnershipMapError(
                f"subsystem {sid!r} uses undeclared authority_tier "
                f"{entry['authority_tier']!r}"
            )
        path = ROOT / module
        if not path.is_file():
            raise OwnershipMapError(f"subsystem {sid!r} owner_module missing: {module}")
        if module.endswith(".py"):
            defined = _defined_names(module)
            for symbol in entry["owner_symbols"]:
                if symbol not in defined:
                    raise OwnershipMapError(
                        f"subsystem {sid!r} claims symbol {symbol!r} in {module}, "
                        "but it is not a top-level class/def/assignment there"
                    )
        if not (ROOT / entry["design_doc"]).is_file():
            raise OwnershipMapError(
                f"subsystem {sid!r} design_doc does not exist: {entry['design_doc']}"
            )
        checked.append(sid)
    return checked


def check_duplicate_risks(doc: dict[str, Any]) -> list[str]:
    checked: list[str] = []
    for risk in doc["duplicate_subsystem_risks"]:
        for key in ("canonical_owner", "shadow_owner"):
            module = risk[key].split("::", 1)[0]
            if not (ROOT / module).is_file():
                raise OwnershipMapError(
                    f"duplicate_subsystem_risks {risk['concern']!r} {key} "
                    f"does not exist: {module}"
                )
        checked.append(risk["concern"])
    return checked


def check_related_overlaps(doc: dict[str, Any]) -> list[str]:
    checked: list[str] = []
    for overlap in doc["related_overlaps"]:
        if not (ROOT / overlap["doc"]).is_file():
            raise OwnershipMapError(
                f"related_overlaps {overlap['linear_id']} doc missing: {overlap['doc']}"
            )
        checked.append(overlap["linear_id"])
    return checked


def _output_contract_version_from_code() -> int:
    match = re.search(
        r"^OUTPUT_CONTRACT_VERSION\s*=\s*(\d+)",
        _read(LANGUAGE_CONTRACT),
        re.MULTILINE,
    )
    if not match:
        raise OwnershipMapError(
            f"{LANGUAGE_CONTRACT} no longer defines OUTPUT_CONTRACT_VERSION"
        )
    return int(match.group(1))


def _output_contract_version_from_doc() -> int:
    match = re.search(
        r"OUTPUT_CONTRACT_VERSION\s*=\s*(\d+)", _read(OUTPUT_CONTRACT_DOC)
    )
    if not match:
        raise OwnershipMapError(
            f"{OUTPUT_CONTRACT_DOC} no longer states OUTPUT_CONTRACT_VERSION"
        )
    return int(match.group(1))


def check_output_contract_version() -> int:
    code_version = _output_contract_version_from_code()
    doc_version = _output_contract_version_from_doc()
    if code_version != doc_version:
        raise OwnershipMapError(
            f"OUTPUT_CONTRACT_VERSION drift: {LANGUAGE_CONTRACT} says {code_version}, "
            f"{OUTPUT_CONTRACT_DOC} says {doc_version} -- reconcile them (see SGS-001 / "
            "known_drift.output_contract_version_doc_mismatch in ownership_map.json)"
        )
    return code_version


def check_downstream_extension_map(doc: dict[str, Any]) -> list[str]:
    subsystem_ids = {s["id"] for s in doc["subsystems"]}
    checked: list[str] = []
    for row in doc["downstream_extension_map"]:
        issue = row["issue"]
        points = row["extension_points"]
        unknown = [p for p in points if p not in subsystem_ids]
        if unknown:
            raise OwnershipMapError(
                f"downstream row {issue} cites unknown extension_points: {unknown}"
            )
        if not points and not row.get("new_owner_justified"):
            raise OwnershipMapError(
                f"downstream row {issue} names no extension_points and does not set "
                "new_owner_justified"
            )
        if row.get("new_owner_justified") and not row.get("justification", "").strip():
            raise OwnershipMapError(
                f"downstream row {issue} sets new_owner_justified but has no justification"
            )
        checked.append(issue)
    return checked


def certify() -> dict[str, Any]:
    doc = _load_map()
    return {
        "map_schema": doc["schema"],
        "subsystems": check_subsystems(doc),
        "duplicate_subsystem_risks": check_duplicate_risks(doc),
        "related_overlaps": check_related_overlaps(doc),
        "output_contract_version": check_output_contract_version(),
        "downstream_extension_map": check_downstream_extension_map(doc),
    }


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        print(json.dumps(certify(), indent=2, sort_keys=True))
    except OwnershipMapError as exc:
        print(f"ownership-map regression: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
