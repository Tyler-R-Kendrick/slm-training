"""The ownership-map certificate must fail on real regressions.

SGS-001 (SLM-435) commits ``src/slm_training/resources/ownership_map.json`` as
the checked-in owner/authority map for the live synthesis stack. SGS-002
(SLM-436) generalizes its OUTPUT_CONTRACT_VERSION doc/code guard into a
table-driven ``CONTRACT_VERSION_CHECKS`` mechanism and wires the whole
certificate into CI (`.github/workflows/ci.yml`) and the changed-file
pre-commit hook (`scripts/check_changed.py`). A map nobody watches drift is
worse than no map: it reads as current while lying. Each test below
reproduces one way the map's claims can rot -- an owner file deleted, a
symbol renamed, a contract-version doc/code mismatch -- and asserts the
certificate catches it.
"""

from __future__ import annotations

import json

import pytest

from scripts import verify_ownership_map as verifier


def _minimal_map() -> dict:
    return {
        "schema": "ownership_map/v1",
        "authority_tiers": {
            "compiler-hard": "deterministic",
            "evaluation-only": "observability",
        },
        "subsystems": [
            {
                "id": "widget",
                "name": "Widget contract",
                "owner_module": "src/slm_training/widget.py",
                "owner_symbols": ["Widget", "build_widget"],
                "authority_tier": "compiler-hard",
                "design_doc": "docs/design/widget.md",
                "notes": "test fixture",
            }
        ],
        "duplicate_subsystem_risks": [
            {
                "concern": "test risk",
                "canonical_owner": "src/slm_training/widget.py",
                "shadow_owner": "src/slm_training/widget.py",
                "status": "reconciled-by-docstring",
                "evidence": "n/a",
                "recommendation": "n/a",
            }
        ],
        "known_drift": [],
        "contract_version_governance": {
            "mechanism": "test fixture",
            "checked_identities": ["OUTPUT_CONTRACT_VERSION", "DSL_TOKENIZER_VERSION"],
            "checkpoint_load_enforcement": {
                "description": "test fixture",
                "enforced_by_module": "src/slm_training/widget.py",
                "enforced_by_symbol": "build_widget",
                "called_from": ["src/slm_training/widget.py"],
                "regression_tests": ["docs/design/widget.md"],
            },
        },
        "related_overlaps": [
            {
                "linear_id": "SLM-1",
                "title": "n/a",
                "doc": "docs/design/widget.md",
                "relation": "n/a",
            }
        ],
        "downstream_extension_map": [
            {
                "issue": "TST-001",
                "linear_id": "SLM-2",
                "title": "n/a",
                "extension_points": ["widget"],
                "new_owner_justified": False,
                "justification": "",
            },
            {
                "issue": "TST-002",
                "linear_id": "SLM-3",
                "title": "n/a",
                "extension_points": [],
                "new_owner_justified": True,
                "justification": "no existing owner fits",
            },
        ],
    }


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A minimal repo tree whose ownership_map.json passes, so each test can
    break exactly one claim. Every ``CONTRACT_VERSION_CHECKS`` entry's code
    and doc files must exist here too, since ``check_contract_versions``
    checks all of them unconditionally against ``verifier.ROOT``."""
    (tmp_path / "src/slm_training/resources").mkdir(parents=True)
    (tmp_path / "src/slm_training/dsl").mkdir(parents=True)
    (tmp_path / "src/slm_training/models").mkdir(parents=True)
    (tmp_path / "docs/design").mkdir(parents=True)

    def write(relative: str, text: str) -> None:
        (tmp_path / relative).write_text(text, encoding="utf-8")

    write(
        "src/slm_training/widget.py",
        "class Widget:\n    pass\n\n\ndef build_widget():\n    return Widget()\n",
    )
    write("docs/design/widget.md", "# Widget\n")
    write(
        "src/slm_training/dsl/language_contract.py",
        "OUTPUT_CONTRACT_VERSION = 2\n",
    )
    write(
        "docs/design/symbol-only-output-contract.md",
        "The canonical contract is `OUTPUT_CONTRACT_VERSION = 2` here.\n",
    )
    write(
        "src/slm_training/models/dsl_tokenizer.py",
        "DSL_TOKENIZER_VERSION = 5\n",
    )
    write(
        "docs/design/agent-harness-parity-audit.md",
        "DSL_TOKENIZER_VERSION stays 5 in both cases.\n",
    )
    write(
        "src/slm_training/resources/ownership_map.json",
        json.dumps(_minimal_map(), indent=2),
    )

    # SGS-010: copy the real serialized-contract modules rather than restating
    # them, so a new SERIALIZED_CONTRACTS row needs no fixture edit.
    for contract in verifier.SERIALIZED_CONTRACTS:
        target = tmp_path / contract.module
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            (verifier.ROOT / contract.module).read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    return tmp_path


def test_passes_on_a_consistent_repo(sandbox):
    result = verifier.certify()
    assert result["subsystems"] == ["widget"]
    assert result["contract_versions"] == {
        "OUTPUT_CONTRACT_VERSION": 2,
        "DSL_TOKENIZER_VERSION": 5,
    }
    assert "TST-001" in result["downstream_extension_map"]


def test_passes_on_the_real_repository():
    """The live ownership_map.json must certify against the live source tree."""
    result = verifier.certify()
    assert result["contract_versions"]["OUTPUT_CONTRACT_VERSION"] == 2
    assert result["contract_versions"]["DSL_TOKENIZER_VERSION"] == 5
    assert "CONTRACT_VERSION_CHECKS" in result["contract_version_governance"]
    assert len(result["subsystems"]) >= 20
    assert len(result["downstream_extension_map"]) >= 50


def test_catches_a_fabricated_checkpoint_enforcement_symbol(sandbox):
    path = sandbox / "src/slm_training/resources/ownership_map.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    governance = doc["contract_version_governance"]["checkpoint_load_enforcement"]
    governance["enforced_by_symbol"] = "does_not_exist"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(verifier.OwnershipMapError, match="not defined there"):
        verifier.certify()


def test_catches_a_missing_checkpoint_enforcement_regression_test(sandbox):
    path = sandbox / "src/slm_training/resources/ownership_map.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    governance = doc["contract_version_governance"]["checkpoint_load_enforcement"]
    governance["regression_tests"] = ["tests/test_scripts/test_does_not_exist.py"]
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(verifier.OwnershipMapError, match="references a missing file"):
        verifier.certify()


def test_catches_deleted_owner_module(sandbox):
    (sandbox / "src/slm_training/widget.py").unlink()
    with pytest.raises(verifier.OwnershipMapError, match="owner_module missing"):
        verifier.certify()


def test_catches_renamed_owner_symbol(sandbox):
    (sandbox / "src/slm_training/widget.py").write_text(
        "class RenamedWidget:\n    pass\n", encoding="utf-8"
    )
    with pytest.raises(verifier.OwnershipMapError, match="claims symbol 'Widget'"):
        verifier.certify()


def test_catches_undeclared_authority_tier(sandbox):
    path = sandbox / "src/slm_training/resources/ownership_map.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["subsystems"][0]["authority_tier"] = "vibes-based"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(verifier.OwnershipMapError, match="undeclared authority_tier"):
        verifier.certify()


def test_catches_output_contract_version_doc_drift(sandbox):
    """The exact SGS-001 finding: code and doc disagree on OUTPUT_CONTRACT_VERSION."""
    (sandbox / "docs/design/symbol-only-output-contract.md").write_text(
        "The canonical contract is `OUTPUT_CONTRACT_VERSION = 4` here.\n",
        encoding="utf-8",
    )
    with pytest.raises(
        verifier.OwnershipMapError, match="OUTPUT_CONTRACT_VERSION drift"
    ):
        verifier.certify()


def test_catches_bare_prose_version_drift(sandbox):
    """A second SGS-001 finding, missed by the first fix: the correctly-spelled
    ``OUTPUT_CONTRACT_VERSION = 2`` can coexist with stray bare-prose mentions
    ("output contract v4", "Version 4") elsewhere in the same doc."""
    (sandbox / "docs/design/symbol-only-output-contract.md").write_text(
        "The canonical contract is `OUTPUT_CONTRACT_VERSION = 2` here.\n"
        "Checkpoint loading requires output contract v4 exactly. Version 4 "
        "requires rewriting markers.\n",
        encoding="utf-8",
    )
    with pytest.raises(
        verifier.OwnershipMapError, match="OUTPUT_CONTRACT_VERSION drift"
    ):
        verifier.certify()


def test_catches_dsl_tokenizer_version_doc_drift(sandbox):
    (sandbox / "docs/design/agent-harness-parity-audit.md").write_text(
        "DSL_TOKENIZER_VERSION stays 3 in both cases.\n", encoding="utf-8"
    )
    with pytest.raises(verifier.OwnershipMapError, match="DSL_TOKENIZER_VERSION drift"):
        verifier.certify()


def test_dsl_tokenizer_version_check_ignores_unrelated_numbers_in_the_doc(sandbox):
    """agent-harness-parity-audit.md is a multi-topic doc -- the check must be
    scoped to numbers near the constant name, not every bare vN/number in the
    file, or unrelated version mentions elsewhere would false-positive."""
    (sandbox / "docs/design/agent-harness-parity-audit.md").write_text(
        "The gate stack is at v9 and the tokenizer vocab is 605 wide.\n"
        "DSL_TOKENIZER_VERSION stays 5 in both cases.\n",
        encoding="utf-8",
    )
    result = verifier.certify()
    assert result["contract_versions"]["DSL_TOKENIZER_VERSION"] == 5


def test_catches_empty_owner_symbols_on_a_python_module(sandbox):
    path = sandbox / "src/slm_training/resources/ownership_map.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["subsystems"][0]["owner_symbols"] = []
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(verifier.OwnershipMapError, match="lists no owner_symbols"):
        verifier.certify()


def test_catches_duplicate_subsystem_ids(sandbox):
    path = sandbox / "src/slm_training/resources/ownership_map.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["subsystems"].append(dict(doc["subsystems"][0]))
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(verifier.OwnershipMapError, match="duplicate subsystems"):
        verifier.certify()


def test_catches_unexpected_schema(sandbox):
    path = sandbox / "src/slm_training/resources/ownership_map.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["schema"] = "ownership_map/v2"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(verifier.OwnershipMapError, match="declares schema"):
        verifier.certify()


def test_catches_downstream_row_with_no_extension_point_and_no_justification(sandbox):
    path = sandbox / "src/slm_training/resources/ownership_map.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["downstream_extension_map"][0]["extension_points"] = []
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(verifier.OwnershipMapError, match="names no extension_points"):
        verifier.certify()


def test_catches_downstream_row_citing_unknown_subsystem(sandbox):
    path = sandbox / "src/slm_training/resources/ownership_map.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["downstream_extension_map"][0]["extension_points"] = ["nonexistent"]
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(verifier.OwnershipMapError, match="unknown extension_points"):
        verifier.certify()


def test_catches_serialized_contract_reader_that_ignores_its_version(sandbox):
    # SGS-010: a reader that drops the version comparison would silently
    # reinterpret an older/newer payload -- must fail loudly, not pass.
    path = sandbox / "src/slm_training/data/progspec/synthesis_problem.py"
    source = path.read_text(encoding="utf-8")
    path.write_text(
        source.replace(
            'if str(value.get("schema_version")) != SYNTHESIS_PROBLEM_SCHEMA_VERSION:',
            "if False:",
        ),
        encoding="utf-8",
    )
    with pytest.raises(verifier.OwnershipMapError, match="never consults"):
        verifier.certify()
