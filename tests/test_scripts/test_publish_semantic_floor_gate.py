from __future__ import annotations

import json
from pathlib import Path

from scripts.publish_semantic_floor_gate import main
from slm_training.harnesses.experiments.semantic_floor_gate import (
    SemanticFloorGateV1,
    render_markdown,
)


def test_publish_then_check_is_idempotent(tmp_path: Path) -> None:
    json_path = tmp_path / "gate.json"
    md_path = tmp_path / "gate.md"
    args = ["--json", str(json_path), "--markdown", str(md_path)]
    assert main(args) == 0
    assert main([*args, "--check"]) == 0
    gate = SemanticFloorGateV1.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
    assert gate.verdict == "inconclusive"
    assert md_path.read_text(encoding="utf-8") == render_markdown(gate)


def test_committed_gate_and_narrative_are_current() -> None:
    assert main(["--check"]) == 0
