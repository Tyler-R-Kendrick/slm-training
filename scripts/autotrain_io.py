"""Generic file reads shared by the loop's extracted modules.

Deliberately tiny. ``read_json`` is needed by almost every other module here;
parking it in whichever one uses it most would force the rest to depend on that
module for a reason unrelated to its purpose. A shared kernel keeps those
dependencies honest (Common Reuse Principle).

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}
