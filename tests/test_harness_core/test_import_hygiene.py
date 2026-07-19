"""Enforce the harness-core contract: DSL-agnostic and import-light.

``slm_training.harness_core`` holds frozen machinery shared by every harness
and must support any DSL/AST. Modules here may import stdlib, third-party
packages, and ``slm_training.harness_core.*`` — nothing else from
``slm_training`` at module level (docs/design/harness-core.md). DSL- or
metric-specific behavior enters via parameters and callbacks instead.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = REPO_ROOT / "src" / "slm_training" / "harness_core"

ALLOWED_PREFIX = "slm_training.harness_core"

# Grandfathered function-level (lazy) imports of other slm_training layers.
# Do not extend this list without updating docs/design/harness-core.md.
LAZY_ALLOWLIST = {
    ("lineage/data_cycle.py", "slm_training.data.store"),
}


def _collect_imports(tree: ast.Module) -> tuple[list[str], list[str]]:
    """Return (module_level, function_level) absolute imported module names."""
    module_level: list[str] = []
    function_level: list[str] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.depth = 0

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1

        visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

        def visit_Import(self, node: ast.Import) -> None:
            bucket = function_level if self.depth else module_level
            bucket.extend(alias.name for alias in node.names)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.module and node.level == 0:
                bucket = function_level if self.depth else module_level
                bucket.append(node.module)

    Visitor().visit(tree)
    return module_level, function_level


def _core_modules() -> list[Path]:
    paths = sorted(CORE_ROOT.rglob("*.py"))
    assert paths, f"no modules found under {CORE_ROOT}"
    return paths


def test_no_dsl_coupled_imports_at_module_level() -> None:
    violations: list[str] = []
    for path in _core_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module_level, _ = _collect_imports(tree)
        for name in module_level:
            if name.startswith("slm_training") and not name.startswith(
                ALLOWED_PREFIX
            ):
                violations.append(f"{path.relative_to(CORE_ROOT)}: {name}")
    assert not violations, (
        "harness_core must stay DSL-agnostic; module-level slm_training imports "
        f"outside {ALLOWED_PREFIX}: {violations}"
    )


def test_lazy_imports_match_allowlist() -> None:
    found: set[tuple[str, str]] = set()
    for path in _core_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        _, function_level = _collect_imports(tree)
        for name in function_level:
            if name.startswith("slm_training") and not name.startswith(
                ALLOWED_PREFIX
            ):
                found.add((str(path.relative_to(CORE_ROOT)), name))
    assert found == LAZY_ALLOWLIST, (
        "lazy slm_training imports in harness_core drifted from the "
        f"grandfathered allowlist: unexpected={sorted(found - LAZY_ALLOWLIST)} "
        f"missing={sorted(LAZY_ALLOWLIST - found)}"
    )


def test_core_import_is_torch_free() -> None:
    code = (
        "import sys; import slm_training.harness_core; "
        "assert 'torch' not in sys.modules, 'harness_core import pulled torch'"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
