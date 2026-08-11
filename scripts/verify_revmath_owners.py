#!/usr/bin/env python3
"""CI certificate for reverse-mathematics / formal evidence owner map (EVID-01).

Static only — parses source with :mod:`ast` and reads text, no imports. Certifies:

1. Every ``surfaces[]`` entry's ``owner_module`` exists and listed ``owner_symbols``
   are defined there (JSON resources may omit symbols).
2. ``design_doc`` and ``adr`` paths exist; both cite ``base_sha``.
3. Each ``surfaces[].id`` is unique (exactly one canonical owner per surface).
4. ``ownership_map_subsystem`` references resolve against ``ownership_map.json``.
5. Prohibited shadow patterns (parallel revmath campaign/evidence/proof stacks)
   do not appear as new module filenames under ``src/`` or ``scripts/``.
6. ``profile_id`` is documented in the ADR and design doc.
7. INTEG-09 discoverability: skill + examples use current schemas/commands;
   design marks external solvers as experiments; no retired write-path ads.

Run: ``python -m scripts.verify_revmath_owners``
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = "src/slm_training/resources/revmath_owner_map.json"
GLOBAL_MAP_PATH = "src/slm_training/resources/ownership_map.json"
DESIGN_DOC = "docs/design/reverse-mathematics-computability.md"
ADR_DOC = "docs/design/adr-revmath-reasoning-profile.md"
SCHEMA = "revmath_owner_map/v1"
SCAN_ROOTS = ("src", "scripts")
SKILL_PATH = ".agents/skills/revmath/SKILL.md"
EXAMPLES_README = "src/slm_training/resources/revmath/examples/README.md"
AGENTS_MD = "AGENTS.md"
ALLOWED_TASK_SCHEMAS = frozenset({"revmath_task/v1"})
REQUIRED_DESIGN_MARKERS = (
    "Trust boundaries",
    "Practical vs genuine RM labels",
    "Exact-refutation authority",
    "Cost models",
    "Runtime refinement scope",
    "Self-healing invariants",
    "Experiment promotion rules",
    "Optional external tools (experiments, not prerequisites)",
)


class RevmathOwnerError(AssertionError):
    """A revmath owner-map claim no longer matches the live repository."""


def _read(relative: str) -> str:
    path = ROOT / relative
    if not path.is_file():
        raise RevmathOwnerError(f"missing required file: {relative}")
    return path.read_text(encoding="utf-8")


def _load_map() -> dict[str, Any]:
    return json.loads(_read(MAP_PATH))


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


def check_surfaces(doc: dict[str, Any]) -> list[str]:
    checked: list[str] = []
    seen: set[str] = set()
    for entry in doc["surfaces"]:
        sid = entry["id"]
        if sid in seen:
            raise RevmathOwnerError(f"duplicate surfaces[].id: {sid!r}")
        seen.add(sid)
        module = entry["owner_module"]
        path = ROOT / module
        if not path.is_file():
            raise RevmathOwnerError(f"surface {sid!r} owner_module missing: {module}")
        symbols = entry.get("owner_symbols") or []
        if module.endswith(".py") and symbols:
            defined = _defined_names(module)
            for symbol in symbols:
                if symbol not in defined:
                    raise RevmathOwnerError(
                        f"surface {sid!r} claims symbol {symbol!r} in {module}, "
                        "but it is not a top-level class/def/assignment there"
                    )
        design = entry.get("design_doc") or doc["design_doc"]
        if not (ROOT / design).is_file():
            raise RevmathOwnerError(f"surface {sid!r} design_doc missing: {design}")
        checked.append(sid)
    return checked


def check_global_subsystems(doc: dict[str, Any]) -> list[str]:
    global_doc = json.loads(_read(GLOBAL_MAP_PATH))
    subsystem_ids = {s["id"] for s in global_doc["subsystems"]}
    checked: list[str] = []
    for entry in doc["surfaces"]:
        ref = entry.get("ownership_map_subsystem")
        if ref is None:
            continue
        if ref not in subsystem_ids:
            raise RevmathOwnerError(
                f"surface {entry['id']!r} cites unknown ownership_map_subsystem: {ref!r}"
            )
        checked.append(ref)
    return checked


def check_docs(doc: dict[str, Any]) -> dict[str, str]:
    base_sha = doc["base_sha"]
    for relative in (doc["design_doc"], doc["adr"], DESIGN_DOC, ADR_DOC):
        text = _read(relative)
        if base_sha not in text:
            raise RevmathOwnerError(
                f"{relative} must cite base_sha {base_sha!r} for reproducible audit"
            )
    profile = doc["profile_id"]
    for relative in (doc["design_doc"], doc["adr"]):
        if profile not in _read(relative):
            raise RevmathOwnerError(f"{relative} must document profile_id {profile!r}")
    return {"base_sha": base_sha, "profile_id": profile}


def check_prohibited_shadows(doc: dict[str, Any]) -> list[str]:
    patterns = [
        re.compile(row["shadow_pattern"], re.IGNORECASE)
        for row in doc["prohibited_duplicate_risks"]
    ]
    hits: list[str] = []
    for root_name in SCAN_ROOTS:
        root = ROOT / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            rel = path.relative_to(ROOT).as_posix()
            stem = path.stem
            for pattern in patterns:
                if pattern.search(stem) or pattern.search(rel):
                    hits.append(rel)
                    break
    if hits:
        raise RevmathOwnerError(
            "prohibited revmath shadow module(s) detected: "
            + ", ".join(sorted(set(hits)))
        )
    return []



def check_integ09_discoverability(doc: dict[str, Any]) -> dict[str, Any]:
    """Skill/examples/docs parity for INTEG-09 (no duplicate registry, no stale writes)."""
    skill = _read(SKILL_PATH)
    for marker in (
        "reasoning/revmath",
        "verify_revmath_owners",
        "reverse-mathematics-computability.md",
        "practical_computability_only",
        "experiments, not prerequisites",
        "run_revmath_task",
        "run_revmath_profile",
    ):
        if marker not in skill:
            raise RevmathOwnerError(f"{SKILL_PATH} missing required marker {marker!r}")
    agents = _read(AGENTS_MD)
    for marker in ("`revmath`", "verify_revmath_owners", "reverse-mathematics-computability.md"):
        if marker not in agents:
            raise RevmathOwnerError(f"{AGENTS_MD} missing revmath discoverability marker {marker!r}")
    examples = _read(EXAMPLES_README)
    for marker in ("revmath_task/v1", "run_revmath_task", "run_revmath_profile", "experiments"):
        if marker not in examples:
            raise RevmathOwnerError(f"{EXAMPLES_README} missing marker {marker!r}")
    design = _read(doc["design_doc"])
    for marker in REQUIRED_DESIGN_MARKERS:
        if marker not in design:
            raise RevmathOwnerError(
                f"{doc['design_doc']} missing INTEG-09 section marker {marker!r}"
            )
    # Positive write ads only — mentioning a marker as retired/prohibited is fine.
    positive_ads = (
        'schema_version": "revmath_task/v0"',
        "schema_version': 'revmath_task/v0'",
        "RevMathCampaignStore",
        "write_revmath_evidence(",
    )
    for relative in (SKILL_PATH, EXAMPLES_README, doc["design_doc"], doc["adr"]):
        body = _read(relative)
        for banned in positive_ads:
            if banned in body:
                raise RevmathOwnerError(
                    f"{relative} advertises retired write marker {banned!r}"
                )
    fixtures = ROOT / "src/slm_training/resources/revmath/fixtures"
    checked = 0
    if fixtures.is_dir():
        for path in sorted(fixtures.glob("*.task.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            schema = payload.get("schema_version")
            if schema not in ALLOWED_TASK_SCHEMAS:
                raise RevmathOwnerError(
                    f"{path.relative_to(ROOT).as_posix()} uses schema_version "
                    f"{schema!r}; expected one of {sorted(ALLOWED_TASK_SCHEMAS)}"
                )
            checked += 1
    if checked < 1:
        raise RevmathOwnerError("no revmath *.task.json fixtures found to parity-check")
    return {
        "skill": SKILL_PATH,
        "examples": EXAMPLES_README,
        "fixtures_checked": checked,
    }


def check_extension_seams(doc: dict[str, Any]) -> list[str]:
    surface_ids = {s["id"] for s in doc["surfaces"]}
    checked: list[str] = []
    for row in doc["extension_seams"]:
        target = row["extends"]
        if target not in surface_ids:
            raise RevmathOwnerError(
                f"extension_seams row {row['issue']!r} cites unknown surface: {target!r}"
            )
        if row.get("new_owner_justified") and not row.get("justification", "").strip():
            raise RevmathOwnerError(
                f"extension_seams row {row['issue']!r} sets new_owner_justified "
                "without justification"
            )
        checked.append(row["issue"])
    return checked


def certify() -> dict[str, Any]:
    doc = _load_map()
    if doc.get("schema") != SCHEMA:
        raise RevmathOwnerError(
            f"{MAP_PATH} declares schema {doc.get('schema')!r}, expected {SCHEMA!r}"
        )
    return {
        "map_schema": doc["schema"],
        "linear_id": doc.get("linear_id"),
        "issue_key": doc.get("issue_key"),
        "docs": check_docs(doc),
        "surfaces": check_surfaces(doc),
        "ownership_map_subsystems": check_global_subsystems(doc),
        "extension_seams": check_extension_seams(doc),
        "prohibited_shadows": check_prohibited_shadows(doc),
        "integ09": check_integ09_discoverability(doc),
    }


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        print(json.dumps(certify(), indent=2, sort_keys=True))
    except (RevmathOwnerError, KeyError, TypeError) as exc:
        print(f"revmath-owner regression: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
