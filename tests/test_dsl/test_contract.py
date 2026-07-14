"""Tests for the reproducible OpenUI ``contract_id`` (F0)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from slm_training.bridge_utils import repo_root
from slm_training.dsl import contract as contract_mod
from slm_training.dsl import contract_components, contract_fingerprint, contract_id
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.train_data.pipeline import _normalize_record

_EXPECTED_KEYS = {
    "lang_spec_version",
    "parser_commit",
    "component_schema",
    "tool_schema",
    "canonicalizer",
    "renderer",
    "tokenizer_version",
}


def test_contract_id_is_deterministic() -> None:
    assert contract_id() == contract_id()


def test_contract_id_format() -> None:
    cid = contract_id()
    assert cid.startswith("oc-")
    hexpart = cid[len("oc-") :]
    assert len(hexpart) == 16
    int(hexpart, 16)  # valid hex


def test_all_seven_components_present() -> None:
    assert set(contract_components()) == _EXPECTED_KEYS
    assert all(isinstance(v, str) and v for v in contract_components().values())


def test_contract_id_matches_hash_of_components() -> None:
    payload = json.dumps(
        contract_components(), sort_keys=True, separators=(",", ":")
    )
    expected = "oc-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    assert contract_id() == expected


def test_lang_spec_version_tracks_bridge_package() -> None:
    pkg = json.loads(
        (repo_root() / "tools" / "openui_bridge" / "package.json").read_text(
            encoding="utf-8"
        )
    )
    declared = pkg["dependencies"]["@openuidev/lang-core"].lstrip("^~>=< ")
    assert contract_components()["lang_spec_version"] == declared


def test_tool_schema_reflects_0_2_x_scope() -> None:
    # 0.2.x layout subset has no tool constructs; v0.5 upgrade is pending.
    assert contract_components()["tool_schema"] == "none@0.2.x"


def test_fingerprint_shape() -> None:
    fp = contract_fingerprint()
    assert fp["contract_id"] == contract_id()
    assert fp["components"] == contract_components()


def test_stamp_injects_and_preserves_meta() -> None:
    stamped = contract_mod.stamp({"parser": "x", "structure_only": True})
    assert stamped["contract_id"] == contract_id()
    assert stamped["parser"] == "x"
    assert stamped["structure_only"] is True
    # non-destructive
    original: dict = {}
    contract_mod.stamp(original)
    assert original == {}


def test_contract_id_changes_when_an_input_changes(monkeypatch) -> None:
    base = contract_id()
    # Any change to a contract input must change the id (new dataset version).
    monkeypatch.setattr(contract_mod, "CANONICALIZER_VERSION", 999_999)
    contract_mod.contract_components.cache_clear()
    contract_mod.contract_id.cache_clear()
    try:
        assert contract_id() != base
    finally:
        monkeypatch.undo()
        contract_mod.contract_components.cache_clear()
        contract_mod.contract_id.cache_clear()
    assert contract_id() == base


def test_read_int_const_reads_real_tokenizer_version() -> None:
    tok = repo_root() / "src" / "slm_training" / "models" / "dsl_tokenizer.py"
    version = contract_mod._read_int_const(tok, "DSL_TOKENIZER_VERSION", -1)
    assert version >= 1
    assert contract_components()["tokenizer_version"] == f"dsl-tok@v{version}"


def test_read_int_const_default_on_missing() -> None:
    missing = Path("/nonexistent/does_not_exist.py")
    assert contract_mod._read_int_const(missing, "ANYTHING", 7) == 7


def test_pipeline_stamps_contract_id_on_records() -> None:
    record = ExampleRecord(
        id="contract-stamp-1",
        prompt="a hero section",
        openui='root = Stack([hero])\nhero = TextContent(":hero.title")',
        split="train",
        source="fixture",
    )
    normalized = _normalize_record(record)
    assert normalized.meta["contract_id"] == contract_id()
