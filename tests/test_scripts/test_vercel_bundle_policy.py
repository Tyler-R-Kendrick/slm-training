import json
from pathlib import Path
import tomllib


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
    package_data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "resources/data/train/slm230_symbol_only_v1/**" in package_data["tool"][
        "setuptools"
    ]["exclude-package-data"]["slm_training"]
    assert "docs/design/*-agentv-*/**" in ignored
    assert "src/slm_training/resources/data/**/records.jsonl" in ignored
    assert "src/slm_training/web/static/**/*.map" in ignored
    assert "src/slm_training/resources/checkpoints/**" in ignored
    for artifact in (
        "/outputs/",
        "/.venv/",
        "/node_modules/",
        "/build/",
        "/session/",
        "/_uv/",
        "/tests/",
    ):
        assert artifact in ignored
    requirements = (ROOT / "src/slm_training/web/requirements-vercel.txt").read_text(
        encoding="utf-8"
    )
    req_deps = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert not any("onnxruntime" in dep for dep in req_deps)
    assert not any("numpy" in dep for dep in req_deps)
    assert any("fastapi" in dep for dep in req_deps)
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
            "harnesses/experiments/**",
        "models/legal_edit_flow",
    ):
        assert fragment in function["excludeFiles"]
