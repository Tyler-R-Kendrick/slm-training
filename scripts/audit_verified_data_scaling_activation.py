"""Write SLM-268's fail-closed local activation disposition."""
from __future__ import annotations
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slm_training.harnesses.experiments.slm268_verified_data_activation import build_activation_report, render_markdown

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = build_activation_report(root)
    base = root / "docs/design/iter-slm268-verified-data-scaling-activation-20260725"
    base.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    base.with_suffix(".md").write_text(render_markdown(report))
    print(report["activation_verdict"])
if __name__ == "__main__":
    main()
