#!/usr/bin/env python3
"""Regenerate or verify docs/design/repository-ownership-map.md from ownership_map.json.

The markdown tables and notes are generated from
``src/slm_training/resources/ownership_map.json``; prose headers and the
contract-version policy block stay in this script as templates. Never hand-edit
the generated tables in the markdown file — edit the JSON and re-run:

    python -m scripts.render_repository_ownership_map
    python -m scripts.render_repository_ownership_map --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "src/slm_training/resources/ownership_map.json"
OUT_PATH = ROOT / "docs/design/repository-ownership-map.md"


def _load_map() -> dict[str, Any]:
    return json.loads(MAP_PATH.read_text(encoding="utf-8"))


def _design_link(design_doc: str) -> str:
    if design_doc.startswith("docs/"):
        name = Path(design_doc).name
        if design_doc.startswith("docs/design/"):
            return f"[{name}]({name})"
        return f"[{name}](../{Path(design_doc).parent.name}/{name})"
    return design_doc


def _render_authority_tiers(doc: dict[str, Any]) -> str:
    lines = [
        "## Authority tiers",
        "",
        'Every subsystem below is tagged with one of these tiers (acceptance criterion: "Authority table distinguishes compiler-hard, verifier-hard, advisory learned, oracle diagnostic, and evaluation-only evidence").',
        "",
        "| Tier | Meaning |",
        "| --- | --- |",
    ]
    for tier, meaning in doc["authority_tiers"].items():
        lines.append(f"| `{tier}` | {meaning} |")
    lines.append("")
    return "\n".join(lines)


def _render_subsystems(doc: dict[str, Any]) -> str:
    lines = [
        "## Subsystem owners",
        "",
        "| Subsystem | Owner module | Key symbols | Tier | Design doc |",
        "| --- | --- | --- | --- | --- |",
    ]
    for sub in doc["subsystems"]:
        symbols = ", ".join(f"`{sym}`" for sym in sub.get("owner_symbols", [])) or "—"
        lines.append(
            f"| `{sub['id']}` — {sub['name']} | `{sub['owner_module']}` | "
            f"{symbols} | `{sub['authority_tier']}` | {_design_link(sub['design_doc'])} |"
        )
    lines.extend(["", "Notes, one per subsystem:", ""])
    for sub in doc["subsystems"]:
        note = sub.get("notes", "").strip()
        if note:
            lines.append(f"- **`{sub['id']}`**: {note}")
    lines.append("")
    return "\n".join(lines)


def _render_serialized_contracts() -> str:
    from scripts.verify_ownership_map import SERIALIZED_CONTRACTS

    covered = ", ".join(f"`{c.contract_id}`" for c in SERIALIZED_CONTRACTS)
    return (
        "**Serialized-contract compatibility (SGS-010/SLM-445):** "
        "`scripts/verify_ownership_map.py` `SERIALIZED_CONTRACTS` additionally certifies, per "
        "persisted contract, that the contract declares a version symbol and that its "
        "class-scoped reader consults that symbol and raises on mismatch — so no payload is "
        f"silently reinterpreted. Covered: {covered}. Runtime reader behavior "
        "(round-trip, future-version rejection, missing-version rejection, registry parity) is "
        "in `tests/test_data/test_synthesis_contract_versions.py`. Policy: "
        "[sgs-010-schema-versioning-compatibility.md](sgs-010-schema-versioning-compatibility.md)."
    )


def _render_contract_governance(doc: dict[str, Any]) -> str:
    gov = doc.get("contract_version_governance", {})
    checked = ", ".join(f"`{item}`" for item in gov.get("checked_identities", []))
    load = gov.get("checkpoint_load_enforcement", {})
    return "\n".join(
        [
            "## Contract-version consistency (SGS-002)",
            "",
            f"**Mechanism:** {gov.get('mechanism', 'scripts/verify_ownership_map.py CONTRACT_VERSION_CHECKS')}.",
            "",
            f"- **Currently checked identities:** {checked or '—'}",
            "",
            _render_serialized_contracts(),
            "",
            "**Historical artifacts are never silently reinterpreted (acceptance criterion):** "
            "OUTPUT_CONTRACT_VERSION is enforced a second, independent time at model-checkpoint load: "
            "`require_current_output_contract()` raises OutputContractError on any mismatch.",
            "",
            f"- Enforced by: `{load.get('enforced_by_symbol', 'require_current_output_contract')}` "
            f"in `{load.get('enforced_by_module', 'src/slm_training/dsl/language_contract.py')}`",
            f"- Called from: {', '.join(f'`{p}`' for p in load.get('called_from', []))}",
            f"- Regression tests: {', '.join(f'`{p}`' for p in load.get('regression_tests', []))}",
            "",
        ]
    )


def _render_duplicate_risks(doc: dict[str, Any]) -> str:
    lines = [
        "## Duplicate-subsystem risks",
        "",
        'Concerns with two owners registered against the same concept. Acceptance criterion: "No new registry/search/verifier/telemetry/pack/version owner duplicates an existing canonical owner" — these are the two the audit found on the *existing* stack; new work must not add a third.',
        "",
    ]
    for risk in doc.get("duplicate_subsystem_risks", []):
        lines.extend(
            [
                f"### {risk['concern']}",
                "",
                f"- **Canonical owner:** `{risk['canonical_owner']}`",
                f"- **Shadow owner:** `{risk['shadow_owner']}`",
                f"- **Status:** `{risk['status']}`",
                f"- **Evidence:** {risk['evidence']}",
                f"- **Recommendation:** {risk['recommendation']}",
                "",
            ]
        )
    return "\n".join(lines)


def _render_known_drift(doc: dict[str, Any]) -> str:
    lines = ["## Known drift (reconciled)", ""]
    for item in doc.get("known_drift", []):
        lines.extend(
            [
                f"### `{item['id']}`",
                "",
                item["description"],
                "",
                f"**Status:** `{item['status']}`. **Fix:** {item['fix']}",
                "",
            ]
        )
    return "\n".join(lines)


def _render_related_overlaps(doc: dict[str, Any]) -> str:
    lines = [
        "## Related overlaps",
        "",
        "Existing Linear issues/docs that downstream SGS-001 work must coordinate with rather than "
        "re-derive (per SGS-001's scope: \"Record overlaps with VSS, LDI, Semantic Planning, "
        "SLM-106, SLM-131, SLM-159, and SLM-160\").",
        "",
        "| Issue | Title | Doc | Relation |",
        "| --- | --- | --- | --- |",
    ]
    for row in doc.get("related_overlaps", []):
        doc_link = _design_link(row["doc"]) if row.get("doc", "").startswith("docs/") else row["doc"]
        lines.append(
            f"| {row['linear_id']} | {row['title']} | {doc_link} | {row['relation']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_downstream_map(doc: dict[str, Any]) -> str:
    lines = [
        "## Downstream extension map",
        "",
        "Every SGS/VCE/PCT/SRP/SIE/RSP/PGS backlog item mapped to the existing subsystem(s) it "
        "extends, or an explicit justification for a new owner (acceptance criterion: \"Map every "
        "downstream item to an existing extension point or explicitly justify a new owner\").",
        "",
        "| Issue | Linear | Extension point(s) | New owner? | Why |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in doc.get("downstream_extension_map", []):
        points = ", ".join(f"`{p}`" for p in row.get("extension_points", [])) or "—"
        new_owner = "yes" if row.get("new_owner_justified") else "no"
        lines.append(
            f"| **{row['issue']}** — {row['title']} | {row['linear_id']} | {points} | "
            f"{new_owner} | {row.get('justification', '').strip()} |"
        )
    lines.append("")
    return "\n".join(lines)


def render(doc: dict[str, Any] | None = None) -> str:
    doc = doc if doc is not None else _load_map()
    header = "\n".join(
        [
            "# Repository ownership map",
            "",
            "> **Goal law:** this document is bound by [decode-invariants.md](decode-invariants.md) — "
            "a rejected approach never closes a goal, and every agent surface carries the law "
            "(I7/I14/I15).",
            "",
            'SGS-001 (SLM-435, milestone "Compatibility and ownership gate"). Commits a machine-readable '
            "owner/authority map for the live OpenUI synthesis stack (`SemanticPlanV1`, the G0-G12 "
            "verifier gates, semantic-failure taxonomy, repair, counterfactual replay, the experiment "
            "registry, telemetry, certified completion artifacts, DSL packs, search/lattice state, "
            "versioning, and generated docs) so the downstream SGS/VCE/PCT/SRP/SIE/RSP/PGS backlog "
            "extends an existing owner instead of duplicating one. SGS-002 (SLM-436) generalizes the "
            "OUTPUT_CONTRACT_VERSION doc/code guard this map introduced into a table-driven, "
            'CI-enforced contract-version consistency mechanism (see "Contract-version consistency" below).',
            "",
            "**Machine-readable source of truth:** "
            "[`src/slm_training/resources/ownership_map.json`]"
            "(../../src/slm_training/resources/ownership_map.json) "
            f"(`schema: {doc.get('schema', 'ownership_map/v1')}`). "
            "**Verified by:** `python -m scripts.verify_ownership_map` (static AST/text check, no "
            "imports — every `owner_module` must exist and every `owner_symbols` entry must actually "
            "be defined there; every downstream row must cite a real extension point or justify a "
            "new owner; every registered contract-version pair must agree between code and its "
            "canonical docs). Wired into `.github/workflows/ci.yml` (`python-static` job) and the "
            "changed-file pre-commit hook (`scripts/check_changed.py`, `OWNERSHIP_MAP_FILES`). "
            "Regression coverage: `tests/test_scripts/test_verify_ownership_map.py`. "
            "Regenerate this page: `python -m scripts.render_repository_ownership_map`.",
            "",
            "*This page is mechanically generated from the JSON above — do not hand-edit the tables "
            "below; edit the JSON and regenerate.*",
            "",
        ]
    )
    footer = "\n".join(
        [
            "## Validation",
            "",
            "```bash",
            "python -m scripts.render_repository_ownership_map --check",
            "python -m scripts.verify_ownership_map",
            "python -m pytest tests/test_scripts/test_verify_ownership_map.py -q",
            "python -m scripts.verify_agent_surfaces",
            "python -m scripts.verify_version_stamps --check",
            "python -m scripts.verify_decode_invariants",
            "```",
            "",
            "`verify_ownership_map` checks every owner claim by parsing the owner module's AST and "
            "confirming each claimed symbol is defined there as a class, function, or assignment "
            "target. It reads only the JSON, never the prose in this file — importer counts in the "
            "notes above come from a manual audit and are not themselves certified.",
            "",
            "## Non-goals",
            "",
            "No new semantic/search behavior and no historical-result rewriting. The `dsl/packs/` "
            "shadow pack registry and the `semantic_repair.py` → `counterfactual_probe.py` "
            "supersession are recorded here as known state, not resolved by this audit — resolving "
            "them is downstream work (see the DSL pack registry risk above and VCE-002/VCE-004).",
            "",
        ]
    )
    parts = [
        header,
        _render_authority_tiers(doc),
        _render_subsystems(doc),
        _render_contract_governance(doc),
        _render_duplicate_risks(doc),
        _render_known_drift(doc),
        _render_related_overlaps(doc),
        _render_downstream_map(doc),
        footer,
    ]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    rendered = render()
    if not rendered.endswith("\n"):
        rendered += "\n"

    if args.check:
        if not OUT_PATH.is_file() or OUT_PATH.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(
                "repository-ownership-map.md is stale; run: python -m scripts.render_repository_ownership_map"
            )
        print(f"repository-ownership-map.md is current: {OUT_PATH}")
        return 0

    OUT_PATH.write_text(rendered, encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
