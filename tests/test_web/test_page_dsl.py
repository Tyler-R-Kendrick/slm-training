"""The dashboard's OpenUI page programs stay valid and in sync.

Guards the OpenUI rendering contract: every dashboard route has a
committed ``static/openui/<slug>.openui`` program, each one references only real
components (``library.tsx`` / stock ``@openuidev``) and real tool-provider queries
(``toolProvider.ts``), and ``MANIFEST.json`` is regenerated. See
``scripts/validate_page_dsl.py`` and ``.agents/skills/dashboard-openui-parity``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
VALIDATOR = REPO / "scripts" / "validate_page_dsl.py"
OPENUI_DIR = REPO / "src" / "slm_training" / "web" / "static" / "openui"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_page_dsl", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


vp = _load_validator()


def test_every_route_has_a_page_program() -> None:
    slugs = vp.route_slugs() - vp.COMPILED_ONLY_SLUGS
    assert slugs, "no dashboard routes discovered in main.tsx"
    have = {p.stem for p in OPENUI_DIR.glob("*.openui")}
    missing = sorted(slugs - have)
    assert not missing, f"routes without a .openui program: {missing}"


def test_page_programs_reference_only_real_components_and_queries() -> None:
    _manifest, errors = vp.build_manifest()
    assert not errors, "OpenUI page DSL validation errors:\n" + "\n".join(errors)


def test_manifest_is_in_sync() -> None:
    manifest, errors = vp.build_manifest()
    assert not errors
    rendered = vp._dumps(manifest)
    current = vp.MANIFEST.read_text(encoding="utf-8") if vp.MANIFEST.exists() else ""
    assert current == rendered, (
        "static/openui/MANIFEST.json is stale — run `python scripts/validate_page_dsl.py`"
    )


@pytest.mark.parametrize("slug", ["overview", "data", "experiments", "smoke", "checkpoints", "playground"])
def test_program_has_root(slug: str) -> None:
    prog = OPENUI_DIR / f"{slug}.openui"
    assert prog.exists(), f"missing {prog.name}"
    assert "root =" in prog.read_text(encoding="utf-8"), f"{prog.name} has no root assignment"
