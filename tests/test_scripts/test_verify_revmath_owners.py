"""Revmath / formal-evidence owner map must fail on real regressions (EVID-01)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import verify_revmath_owners as verifier

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_passes_on_the_real_repository():
    result = verifier.certify()
    assert result["map_schema"] == "revmath_owner_map/v1"
    assert result["issue_key"] == "EVID-01"
    assert result["linear_id"] == "SLM-515"
    assert len(result["surfaces"]) >= 15
    assert result["docs"]["profile_id"] == "reasoning/revmath"
    assert result["integ09"]["skill"] == ".agents/skills/revmath/SKILL.md"
    assert result["integ09"]["fixtures_checked"] >= 1


def test_catches_renamed_owner_symbol(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    map_path = tmp_path / verifier.MAP_PATH
    map_path.parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads((REPO_ROOT / verifier.MAP_PATH).read_text(encoding="utf-8"))
    widget = tmp_path / "src/slm_training/widget.py"
    widget.parent.mkdir(parents=True, exist_ok=True)
    widget.write_text("class Renamed:\n    pass\n", encoding="utf-8")
    doc["surfaces"] = [
        {
            "id": "widget",
            "kind": "schema",
            "owner_module": "src/slm_training/widget.py",
            "owner_symbols": ["Widget"],
            "design_doc": verifier.DESIGN_DOC,
            "ownership_map_subsystem": None,
            "notes": "fixture",
        }
    ]
    for relative in (verifier.DESIGN_DOC, verifier.ADR_DOC):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"base {doc['base_sha']} profile {doc['profile_id']}\n",
            encoding="utf-8",
        )
    (tmp_path / verifier.GLOBAL_MAP_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / verifier.GLOBAL_MAP_PATH).write_text(
        json.dumps({"subsystems": []}), encoding="utf-8"
    )
    map_path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(verifier.RevmathOwnerError, match="claims symbol 'Widget'"):
        verifier.certify()


def test_catches_prohibited_shadow_module(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    doc = json.loads((REPO_ROOT / verifier.MAP_PATH).read_text(encoding="utf-8"))
    shadow = tmp_path / "src/slm_training/revmath_campaign.py"
    shadow.parent.mkdir(parents=True, exist_ok=True)
    shadow.write_text("# shadow owner\n", encoding="utf-8")
    with pytest.raises(verifier.RevmathOwnerError, match="prohibited revmath shadow"):
        verifier.check_prohibited_shadows(doc)


def test_integ09_requires_skill_markers(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    doc = json.loads((REPO_ROOT / verifier.MAP_PATH).read_text(encoding="utf-8"))
    # Minimal tree mirroring certify prerequisites for discoverability only.
    for relative in (
        verifier.MAP_PATH,
        verifier.DESIGN_DOC,
        verifier.ADR_DOC,
        verifier.GLOBAL_MAP_PATH,
        verifier.SKILL_PATH,
        verifier.EXAMPLES_README,
        verifier.AGENTS_MD,
    ):
        src = REPO_ROOT / relative
        dst = tmp_path / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if relative == verifier.SKILL_PATH:
            dst.write_text("incomplete skill\n", encoding="utf-8")
        elif relative == verifier.GLOBAL_MAP_PATH:
            dst.write_text(json.dumps({"subsystems": [{"id": "agent_surface_parity"}]}), encoding="utf-8")
        elif src.is_file():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(verifier.RevmathOwnerError, match="missing required marker"):
        verifier.check_integ09_discoverability(doc)
