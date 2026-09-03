"""Component dependency graph, built from ``ast`` import statements.

A "component" here is a Python package directory (``slm_training.harnesses``,
``slm_training.dsl.grammar``, ...) -- the smallest unit this repository could
release or reuse independently, which is the unit Robert C. Martin's package
principles are stated over.

Static only -- nothing is imported, so a module with heavy or failing imports is
still analysable. See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

#: The package tree these metrics are computed over.
SOURCE_ROOT = "src"
ANALYSED_ROOT = "slm_training"


@dataclass(frozen=True)
class ModuleInfo:
    """One module: its dotted name, component, and outbound dotted imports."""

    dotted: str
    component: str
    path: str
    imports: frozenset[str]
    classes: int
    abstract_classes: int


def _dotted_name(relative: Path) -> str:
    parts = list(relative.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _component_of(dotted: str, *, is_package_init: bool) -> str:
    """The package a module belongs to.

    An ``__init__`` module *is* its package; any other module belongs to its
    parent package.
    """

    if is_package_init:
        return dotted
    return dotted.rpartition(".")[0] or dotted


def _is_abstract(node: ast.ClassDef) -> bool:
    """True for ABCs, Protocols, and classes with an abstract method."""

    for base in node.bases:
        name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
        if name in {"ABC", "Protocol", "ABCMeta"}:
            return True
    for keyword in node.keywords:
        value = keyword.value
        name = (
            value.attr
            if isinstance(value, ast.Attribute)
            else getattr(value, "id", "")
        )
        if keyword.arg == "metaclass" and name == "ABCMeta":
            return True
    for child in ast.walk(node):
        if not isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for decorator in child.decorator_list:
            label = (
                decorator.attr
                if isinstance(decorator, ast.Attribute)
                else getattr(decorator, "id", "")
            )
            if label in {"abstractmethod", "abstractproperty"}:
                return True
    return False


def _imported_dotted_names(tree: ast.Module, *, package: str) -> set[str]:
    """Every dotted target imported by a module, with relative forms resolved."""

    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                trimmed = base[: len(base) - node.level + 1]
                prefix = ".".join([*trimmed, node.module] if node.module else trimmed)
            else:
                prefix = node.module or ""
            if not prefix:
                continue
            targets.add(prefix)
            targets.update(f"{prefix}.{alias.name}" for alias in node.names)
    return targets


def parse_modules(*, root: Path = ROOT) -> list[ModuleInfo]:
    """Every module under the analysed package root."""

    base = root / SOURCE_ROOT / ANALYSED_ROOT
    modules: list[ModuleInfo] = []
    for path in sorted(base.rglob("*.py")):
        relative = path.relative_to(root / SOURCE_ROOT)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        dotted = _dotted_name(relative)
        if not dotted:
            continue
        is_init = path.name == "__init__.py"
        component = _component_of(dotted, is_package_init=is_init)
        package = dotted if is_init else dotted.rpartition(".")[0]
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        modules.append(
            ModuleInfo(
                dotted=dotted,
                component=component,
                path=path.relative_to(root).as_posix(),
                imports=frozenset(_imported_dotted_names(tree, package=package)),
                classes=len(classes),
                abstract_classes=sum(1 for n in classes if _is_abstract(n)),
            )
        )
    return modules


def resolve_edges(modules: list[ModuleInfo]) -> dict[str, set[str]]:
    """Component -> the other components it depends on.

    A dotted import target is attributed to the longest known module prefix, so
    ``from slm_training.dsl.grammar import Rule`` counts against the module
    ``slm_training.dsl.grammar`` even though ``Rule`` is a symbol inside it.
    """

    known = {module.dotted: module.component for module in modules}
    edges: dict[str, set[str]] = defaultdict(set)
    for module in modules:
        edges.setdefault(module.component, set())
        for target in module.imports:
            if not target.startswith(f"{ANALYSED_ROOT}."):
                continue
            candidate = target
            while candidate and candidate not in known:
                candidate = candidate.rpartition(".")[0]
            if not candidate:
                continue
            component = known[candidate]
            if component != module.component:
                edges[module.component].add(component)
    return dict(edges)
