import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_vercel_function_excludes_nested_agentv_evidence() -> None:
    config = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    function = config["functions"]["src/slm_training/web/vercel.py"]
    assert "docs/design/*.json" in function["includeFiles"]
    assert "docs/design/**" not in function["includeFiles"]
    assert "docs/design/*-agentv-*/**" in function["excludeFiles"]
    for path in (
        "src/slm_training/flow/samplers.py",
        "src/slm_training/flow/targets.py",
        "src/slm_training/harnesses/experiments/slm199_legal_edit_flow.py",
        "src/slm_training/models/legal_edit_flow.py",
    ):
        assert path in function["excludeFiles"]
