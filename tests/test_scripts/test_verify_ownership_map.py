"""The ownership-map certificate must fail on real regressions.

SGS-001 (SLM-435) commits ``src/slm_training/resources/ownership_map.json`` as
the checked-in owner/authority map for the live synthesis stack. A map nobody
watches drift is worse than no map: it reads as current while lying. Each
test below reproduces one way the map's claims can rot -- an owner file
deleted, a symbol renamed, the exact OUTPUT_CONTRACT_VERSION doc/code
mismatch this audit found and fixed -- and asserts the certificate catches it.
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
    break exactly one claim."""
    (tmp_path / "src/slm_training/resources").mkdir(parents=True)
    (tmp_path / "src/slm_training/dsl").mkdir(parents=True)
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
        "src/slm_training/resources/ownership_map.json",
        json.dumps(_minimal_map(), indent=2),
    )

    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    return tmp_path


def test_passes_on_a_consistent_repo(sandbox):
    result = verifier.certify()
    assert result["subsystems"] == ["widget"]
    assert result["output_contract_version"] == 2
    assert "TST-001" in result["downstream_extension_map"]


def test_passes_on_the_real_repository():
    """The live ownership_map.json must certify against the live source tree."""
    result = verifier.certify()
    assert result["output_contract_version"] == 2
    assert len(result["subsystems"]) >= 20
    assert len(result["downstream_extension_map"]) >= 50


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
