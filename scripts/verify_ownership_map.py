#!/usr/bin/env python3
"""CI certificate for the repository owner/authority map and its watched
contract-version pairs (SGS-001/SLM-435, SGS-002/SLM-436).

Static only -- parses source with :mod:`ast` and reads text, no imports, so it
runs in the dependency-light static CI job (`.github/workflows/ci.yml`
`python-static` job) and the changed-file pre-commit hook
(`scripts/check_changed.py`). It certifies:

1. Every ``subsystems[]`` entry's ``owner_module`` exists and every listed
   ``owner_symbols`` name is actually a class/def/assignment target in that
   file (catches an owner claim rotting after a rename or delete).
2. Every ``authority_tier`` used is declared in ``authority_tiers``.
3. Every ``duplicate_subsystem_risks[]`` and ``related_overlaps[]`` reference
   resolves to a real path.
4. Every registered ``CONTRACT_VERSION_CHECKS`` entry: a code constant
   (``OUTPUT_CONTRACT_VERSION``, ``DSL_TOKENIZER_VERSION``, ...) matches every
   version number its canonical docs cite for it in prose -- the exact drift
   class SGS-001 found (map ``known_drift`` entry
   ``output_contract_version_doc_mismatch``). Regresses loudly if either side
   changes without the other; add an entry here for any new contract/tokenizer/
   pack/artifact/schema/evaluator version identity that gains a doc claim.
5. Every ``SERIALIZED_CONTRACTS`` entry (SGS-010/SLM-445): the contract
   declares an explicit version symbol and its reader consults that symbol and
   raises on mismatch, so no persisted payload is ever silently reinterpreted.
6. Every ``downstream_extension_map[]`` row either cites at least one real
   ``subsystems[].id`` as its extension point, or sets
   ``new_owner_justified: true`` with a non-empty ``justification``.

Run: ``python -m scripts.verify_ownership_map``
"""

from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import dataclass
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
            if not entry["owner_symbols"]:
                raise OwnershipMapError(
                    f"subsystem {sid!r} owns Python module {module} but lists no "
                    "owner_symbols -- an empty list certifies only that the file "
                    "exists, not that the claimed ownership is real"
                )
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


def check_contract_version_governance(doc: dict[str, Any]) -> str:
    """The doc's claim that OUTPUT_CONTRACT_VERSION is independently
    re-enforced at checkpoint load (SGS-002 acceptance criterion: historical
    artifacts are never silently reinterpreted) must point at real code and
    real regression tests, not just prose."""
    governance = doc["contract_version_governance"]
    enforcement = governance["checkpoint_load_enforcement"]
    module = enforcement["enforced_by_module"]
    symbol = enforcement["enforced_by_symbol"]
    if not (ROOT / module).is_file():
        raise OwnershipMapError(
            f"contract_version_governance.checkpoint_load_enforcement "
            f"enforced_by_module missing: {module}"
        )
    if symbol not in _defined_names(module):
        raise OwnershipMapError(
            f"contract_version_governance.checkpoint_load_enforcement claims "
            f"{symbol!r} in {module}, but it is not defined there"
        )
    for relative in (*enforcement["called_from"], *enforcement["regression_tests"]):
        if not (ROOT / relative).is_file():
            raise OwnershipMapError(
                f"contract_version_governance.checkpoint_load_enforcement "
                f"references a missing file: {relative}"
            )
    return governance["mechanism"]


def check_related_overlaps(doc: dict[str, Any]) -> list[str]:
    checked: list[str] = []
    for overlap in doc["related_overlaps"]:
        if not (ROOT / overlap["doc"]).is_file():
            raise OwnershipMapError(
                f"related_overlaps {overlap['linear_id']} doc missing: {overlap['doc']}"
            )
        checked.append(overlap["linear_id"])
    return checked


class ContractVersionCheck:
    """One `<CONSTANT> = N` in code that must never silently disagree with
    the docs that cite it in prose (SGS-002, SLM-436). ``doc_patterns`` are
    tried in order against each doc in ``doc_paths``; every match across
    every pattern/doc becomes a "mention" that must fall inside
    ``{current, current - 1}`` (the doc may legitimately describe both the
    live version and the one it superseded)."""

    def __init__(
        self,
        name: str,
        code_path: str,
        code_pattern: str,
        doc_paths: tuple[str, ...],
        doc_patterns: tuple[str, ...],
    ) -> None:
        self.name = name
        self.code_path = code_path
        self.code_pattern = code_pattern
        self.doc_paths = doc_paths
        self.doc_patterns = doc_patterns

    def code_version(self) -> int:
        match = re.search(self.code_pattern, _read(self.code_path), re.MULTILINE)
        if not match:
            raise OwnershipMapError(f"{self.code_path} no longer defines {self.name}")
        return int(match.group(1))

    def doc_mentions(self) -> set[int]:
        mentions: set[int] = set()
        for doc_path in self.doc_paths:
            text = _read(doc_path)
            for pattern in self.doc_patterns:
                mentions |= {int(n) for n in re.findall(pattern, text)}
        return mentions

    def check(self) -> int:
        code_version = self.code_version()
        doc_versions = self.doc_mentions()
        docs = ", ".join(self.doc_paths)
        if not doc_versions:
            raise OwnershipMapError(f"{docs} no longer state {self.name}")
        # The docs legitimately mention the current version and the
        # immediately prior one (e.g. "v2 is checkpoint-incompatible with v1").
        allowed = {code_version, code_version - 1}
        stray = sorted(doc_versions - allowed)
        if stray:
            raise OwnershipMapError(
                f"{self.name} drift: {self.code_path} says {code_version}, but "
                f"{docs} also mention version(s) {stray} that are neither the "
                f"current ({code_version}) nor prior ({code_version - 1}) "
                "version -- reconcile them (SGS-002 contract-version consistency)"
            )
        if code_version not in doc_versions:
            raise OwnershipMapError(
                f"{self.name} drift: {self.code_path} says {code_version}, but "
                f"{docs} never mention it (only {sorted(doc_versions)}) -- "
                "reconcile them (SGS-002 contract-version consistency)"
            )
        return code_version


# `OUTPUT_CONTRACT_VERSION`'s doc is single-topic (the whole file is about
# this one contract), so every bare `vN`/`Version N` mention in it belongs to
# this check -- that's how SGS-001 caught "output contract v4" prose sitting
# next to a correctly-spelled "= 2". A multi-topic doc would need a
# name-scoped pattern instead (see DSL_TOKENIZER_VERSION below) or every
# unrelated version number in the doc would trip this check.
CONTRACT_VERSION_CHECKS: tuple[ContractVersionCheck, ...] = (
    ContractVersionCheck(
        name="OUTPUT_CONTRACT_VERSION",
        code_path=LANGUAGE_CONTRACT,
        code_pattern=r"^OUTPUT_CONTRACT_VERSION\s*=\s*(\d+)",
        doc_paths=(OUTPUT_CONTRACT_DOC,),
        doc_patterns=(
            r"OUTPUT_CONTRACT_VERSION\s*=\s*(\d+)",
            r"\bv(\d+)\b",
            r"\bVersion (\d+)\b",
        ),
    ),
    ContractVersionCheck(
        name="DSL_TOKENIZER_VERSION",
        code_path="src/slm_training/models/dsl_tokenizer.py",
        code_pattern=r"^DSL_TOKENIZER_VERSION\s*=\s*(\d+)",
        doc_paths=("docs/design/agent-harness-parity-audit.md",),
        doc_patterns=(r"DSL_TOKENIZER_VERSION\D{0,20}(\d+)",),
    ),
)


def check_contract_versions() -> dict[str, int]:
    return {check.name: check.check() for check in CONTRACT_VERSION_CHECKS}


@dataclass(frozen=True)
class SerializedContract:
    """One persisted contract that must version itself and fail closed on drift.

    SGS-010 (SLM-445): every serialized contract this initiative introduced has
    to (a) declare an explicit version constant and (b) reject a payload whose
    version does not match, rather than silently reinterpreting it. Checked
    statically so it runs in the dependency-light CI job; the matching runtime
    reader behavior is covered by
    ``tests/test_data/test_synthesis_contract_versions.py``.
    """

    contract_id: str
    module: str
    version_symbol: str
    reader: str

    def check(self) -> str:
        source = _read(self.module)
        tree = ast.parse(source, filename=self.module)
        defined = _defined_names(self.module)
        if self.version_symbol not in defined:
            raise OwnershipMapError(
                f"serialized contract {self.contract_id!r} does not define its "
                f"version symbol {self.version_symbol!r} in {self.module}"
            )
        owner: ast.ClassDef | None = next(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef) and node.name == self.contract_id
            ),
            None,
        )
        if owner is None:
            raise OwnershipMapError(
                f"serialized contract {self.contract_id!r} is not defined in "
                f"{self.module}"
            )
        # Class-scoped: sibling contracts in the same module each own their
        # reader, so a module-wide walk would grade the wrong function.
        reader_node: ast.FunctionDef | None = next(
            (
                node
                for node in owner.body
                if isinstance(node, ast.FunctionDef) and node.name == self.reader
            ),
            None,
        )
        if reader_node is None:
            raise OwnershipMapError(
                f"serialized contract {self.contract_id!r} has no {self.reader!r} "
                f"reader in {self.module}"
            )
        reader_src = ast.get_source_segment(source, reader_node) or ""
        if self.version_symbol not in reader_src:
            raise OwnershipMapError(
                f"{self.contract_id!r} reader {self.reader!r} never consults "
                f"{self.version_symbol!r}: an older/newer payload would be "
                "silently reinterpreted"
            )
        if not any(isinstance(n, ast.Raise) for n in ast.walk(reader_node)):
            raise OwnershipMapError(
                f"{self.contract_id!r} reader {self.reader!r} does not raise on a "
                "version mismatch (must fail closed, never coerce)"
            )
        return self.contract_id


SERIALIZED_CONTRACTS: tuple[SerializedContract, ...] = (
    SerializedContract(
        contract_id="PromptSemanticRequirementsV1",
        module="src/slm_training/data/progspec/prompt_requirements.py",
        version_symbol="REQUIREMENTS_SCHEMA_VERSION",
        reader="from_dict",
    ),
    SerializedContract(
        contract_id="VerifiedSynthesisProblemV1",
        module="src/slm_training/data/progspec/synthesis_problem.py",
        version_symbol="SYNTHESIS_PROBLEM_SCHEMA_VERSION",
        reader="from_dict",
    ),
    SerializedContract(
        contract_id="VerifierWitnessV1",
        module="src/slm_training/evals/semantic_failure.py",
        version_symbol="WITNESS_SCHEMA_VERSION",
        reader="from_dict",
    ),
    SerializedContract(
        contract_id="DecodeStatsRecordV1",
        module="src/slm_training/models/decode_stats.py",
        version_symbol="DECODE_STATS_RECORD_SCHEMA",
        reader="from_dict",
    ),
    SerializedContract(
        contract_id="ComputeStateFeaturesV1",
        module="src/slm_training/harnesses/experiments/compute_state_controller.py",
        version_symbol="COMPUTE_STATE_FEATURES_SCHEMA",
        reader="from_dict",
    ),
)


def check_serialized_contracts() -> list[str]:
    return [contract.check() for contract in SERIALIZED_CONTRACTS]


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


SCHEMA = "ownership_map/v1"


def _check_no_duplicate_ids(doc: dict[str, Any]) -> None:
    ids = [s["id"] for s in doc["subsystems"]]
    duplicates = sorted({sid for sid in ids if ids.count(sid) > 1})
    if duplicates:
        raise OwnershipMapError(f"duplicate subsystems[].id values: {duplicates}")


def certify() -> dict[str, Any]:
    doc = _load_map()
    if doc.get("schema") != SCHEMA:
        raise OwnershipMapError(
            f"{MAP_PATH} declares schema {doc.get('schema')!r}, expected {SCHEMA!r}"
        )
    _check_no_duplicate_ids(doc)
    return {
        "map_schema": doc["schema"],
        "subsystems": check_subsystems(doc),
        "duplicate_subsystem_risks": check_duplicate_risks(doc),
        "related_overlaps": check_related_overlaps(doc),
        "contract_versions": check_contract_versions(),
        "serialized_contracts": check_serialized_contracts(),
        "contract_version_governance": check_contract_version_governance(doc),
        "downstream_extension_map": check_downstream_extension_map(doc),
    }


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        print(json.dumps(certify(), indent=2, sort_keys=True))
    except (OwnershipMapError, KeyError, TypeError) as exc:
        print(f"ownership-map regression: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
