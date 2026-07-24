import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_vercel_function_excludes_nested_agentv_evidence() -> None:
    config = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    function = config["functions"]["src/slm_training/web/vercel.py"]
    assert "docs/design/*.json" in function["includeFiles"]
    assert "docs/design/**" not in function["includeFiles"]
    assert "docs/design/{*-agentv-*/**,iter-slm200-*.json}" in function["excludeFiles"]
    assert len(function["excludeFiles"]) <= 256
    ignored = (ROOT / ".vercelignore").read_text(encoding="utf-8")
    assert "src/slm_training/resources/data/train/slm230_symbol_only_v1/**" in ignored
    assert "docs/design/*-agentv-*/**" in ignored
    assert "scripts/run_slm243_recursive_update_gate.py" in ignored
    assert (
        "src/slm_training/harnesses/experiments/slm243_recursive_update_gate.py"
        in ignored
    )
    assert "scripts/run_slm233_recursive_campaign.py" in ignored
    assert (
        "src/slm_training/harnesses/experiments/slm233_recursive_campaign.py"
        in ignored
    )
    for fragment in (
        "flow/{samplers,targets}",
        "harnesses/experiments/slm{199,200}_*",
        "models/legal_edit_flow",
    ):
        assert fragment in function["excludeFiles"]
