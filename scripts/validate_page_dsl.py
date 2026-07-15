#!/usr/bin/env python3
"""Validate the dashboard's per-page OpenUI DSL programs and keep a manifest.

The dashboard can render each page two ways (see the compiled ↔ interpreted toggle):

* **compiled**  — the hand-written React pages in ``src/apps/dashboard/src/pages/``
* **interpreted** — the committed OpenUI Lang programs in
  ``src/slm_training/web/static/openui/*.openui`` run live through the official
  ``@openuidev`` ``<Renderer>`` with the dashboard's hybrid component library
  (``src/apps/dashboard/src/interpret/library.tsx``) and ``/api`` tool provider
  (``src/apps/dashboard/src/interpret/toolProvider.ts``).

Those ``.openui`` programs are a **different dialect** from the placeholder-only
training DSL that ``src/apps/openui_bridge`` validates: they use the full OpenUI Lang
(``Query`` / ``@Each`` / ``$state``) plus the dashboard's own custom components. So
this validator does **not** go through the training bridge. Instead it structurally
checks each program against the two source-of-truth files that back interpreted mode:

* every ``Query("name", …)`` resolves to a key in ``toolProvider.ts``
* every ``Component(…)`` call resolves to a stock ``@openuidev`` component or a
  ``defineComponent`` in ``library.tsx`` (or a known builtin)
* brackets balance and a ``root =`` assignment exists
* every dashboard route has a matching ``.openui`` file

It is intentionally dependency-light (pure text analysis, standard library only) so
it runs in CI without installing the dashboard's ``node_modules``. On success it
rewrites ``src/slm_training/web/static/openui/MANIFEST.json`` deterministically.

Usage::

    python scripts/validate_page_dsl.py            # validate all + rewrite manifest
    python scripts/validate_page_dsl.py --check     # validate, fail if manifest stale
    python scripts/validate_page_dsl.py --changed    # PostToolUse hook mode (reads stdin)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OPENUI_DIR = REPO / "src" / "slm_training" / "web" / "static" / "openui"
LIBRARY_TSX = REPO / "src" / "apps" / "dashboard" / "src" / "interpret" / "library.tsx"
TOOLPROVIDER_TS = (
    REPO / "src" / "apps" / "dashboard" / "src" / "interpret" / "toolProvider.ts"
)
MAIN_TSX = REPO / "src" / "apps" / "dashboard" / "src" / "main.tsx"
MANIFEST = OPENUI_DIR / "MANIFEST.json"

# Stock @openuidev/react-ui component names (openuiLibrary.components) — embedded so
# the validator stays dependency-free. Refresh with:
#   node -e 'import("@openuidev/react-ui").then(m=>console.log(JSON.stringify(Object.keys(m.openuiLibrary.components).sort())))'
STOCK_COMPONENTS = {
    "Accordion", "AccordionItem", "AreaChart", "BarChart", "Button", "Buttons",
    "Callout", "Card", "CardHeader", "Carousel", "CheckBoxGroup", "CheckBoxItem",
    "CodeBlock", "Col", "DatePicker", "Form", "FormControl", "HorizontalBarChart",
    "Image", "ImageBlock", "ImageGallery", "Input", "Label", "LineChart",
    "MarkDownRenderer", "Modal", "PieChart", "Point", "RadarChart", "RadialChart",
    "RadioGroup", "RadioItem", "ScatterChart", "ScatterSeries", "Select",
    "SelectItem", "Separator", "Series", "SingleStackedBarChart", "Slice", "Slider",
    "Stack", "Steps", "StepsItem", "SwitchGroup", "SwitchItem", "TabItem", "Table",
    "Tabs", "Tag", "TagBlock", "TextArea", "TextCallout", "TextContent",
}

# Language builtins that appear as Capitalized calls in OpenUI Lang programs.
BUILTINS = {"Query", "Mutation", "Action"}

# Routes that are intentionally compiled-only (dynamic / not expressible as a static
# page program) and therefore exempt from the "every route has a .openui" check.
COMPILED_ONLY_SLUGS = {"runs"}

_STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'')
_CALL_RE = re.compile(r"(?<!@)(?<![\w.])([A-Z][A-Za-z0-9_]*)\s*\(")
_QUERY_RE = re.compile(r'Query\(\s*"([^"]+)"')
_DEFINE_NAME_RE = re.compile(r'name:\s*"([A-Z][A-Za-z0-9_]*)"')
_PROVIDER_KEY_RE = re.compile(r"^\s{2}([A-Za-z_][A-Za-z0-9_]*):\s*async", re.MULTILINE)
_ROUTE_PATH_RE = re.compile(r'path:\s*"([^"]*)"')


def _strip_strings(text: str) -> str:
    """Blank out string-literal contents so their words aren't read as calls."""
    return _STRING_RE.sub('""', text)


def library_components() -> set[str]:
    if not LIBRARY_TSX.exists():
        return set()
    return set(_DEFINE_NAME_RE.findall(LIBRARY_TSX.read_text(encoding="utf-8")))


def provider_queries() -> set[str]:
    if not TOOLPROVIDER_TS.exists():
        return set()
    return set(_PROVIDER_KEY_RE.findall(TOOLPROVIDER_TS.read_text(encoding="utf-8")))


def route_slugs() -> set[str]:
    if not MAIN_TSX.exists():
        return set()
    slugs: set[str] = set()
    for path in _ROUTE_PATH_RE.findall(MAIN_TSX.read_text(encoding="utf-8")):
        slug = "overview" if path == "/" else path.strip("/").split("/")[0]
        if slug:
            slugs.add(slug)
    return slugs


def _balanced(text: str) -> str | None:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for ch in _strip_strings(text):
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return f"unbalanced '{ch}'"
    if stack:
        return f"unclosed '{stack[-1]}'"
    return None


def validate_file(path: Path, known_components: set[str], known_queries: set[str]) -> tuple[dict, list[str]]:
    errors: list[str] = []
    src = path.read_text(encoding="utf-8")
    code = _strip_strings(src)

    bal = _balanced(src)
    if bal:
        errors.append(f"{path.name}: {bal}")

    if not re.search(r"(^|\n)\s*root\s*=", code):
        errors.append(f"{path.name}: no `root =` assignment")

    calls = sorted(set(_CALL_RE.findall(code)))
    components = [c for c in calls if c not in BUILTINS]
    for c in components:
        if c not in known_components:
            errors.append(f"{path.name}: unknown component `{c}` (not in library.tsx or stock openui)")

    queries = sorted(set(_QUERY_RE.findall(src)))
    for q in queries:
        if q not in known_queries:
            errors.append(f"{path.name}: Query(\"{q}\") has no matching key in toolProvider.ts")

    entry = {
        "file": path.name,
        "sha256": hashlib.sha256(src.encode("utf-8")).hexdigest(),
        "components": components,
        "queries": queries,
        "status": "ok" if not errors else "error",
    }
    return entry, errors


def build_manifest() -> tuple[dict, list[str]]:
    known_components = library_components() | STOCK_COMPONENTS
    known_queries = provider_queries()
    slugs = route_slugs()

    files = sorted(OPENUI_DIR.glob("*.openui"))
    pages: dict[str, dict] = {}
    errors: list[str] = []
    for f in files:
        entry, errs = validate_file(f, known_components, known_queries)
        pages[f.stem] = entry
        errors.extend(errs)

    have = {f.stem for f in files}
    for slug in sorted(slugs - COMPILED_ONLY_SLUGS):
        if slug not in have:
            errors.append(f"route '/{'' if slug == 'overview' else slug}' has no {slug}.openui page program")

    manifest = {
        "note": "Generated by scripts/validate_page_dsl.py — do not edit by hand.",
        "pages": pages,
    }
    return manifest, errors


def _dumps(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _changed_paths_from_stdin() -> list[str]:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return []
    ti = payload.get("tool_input", {}) if isinstance(payload, dict) else {}
    paths = [ti.get("file_path")]
    for key in ("edits", "changes"):
        for item in ti.get(key, []) or []:
            if isinstance(item, dict):
                paths.append(item.get("file_path"))
    return [p for p in paths if p]


def _relevant(paths: list[str]) -> bool:
    watched = ("static/openui/", "interpret/library.tsx", "interpret/toolProvider.ts", "dashboard/src/main.tsx")
    return any(any(w in p for w in watched) for p in paths)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail if MANIFEST.json is stale instead of rewriting it")
    ap.add_argument("--changed", action="store_true", help="PostToolUse hook mode: read stdin, act only on relevant edits")
    args = ap.parse_args()

    if args.changed and not _relevant(_changed_paths_from_stdin()):
        return 0

    if not OPENUI_DIR.exists():
        print(f"[validate_page_dsl] no openui dir at {OPENUI_DIR}", file=sys.stderr)
        return 0

    manifest, errors = build_manifest()
    rendered = _dumps(manifest)

    if errors:
        print("OpenUI page DSL validation FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  ✗ {e}", file=sys.stderr)
        return 1

    if args.check:
        current = MANIFEST.read_text(encoding="utf-8") if MANIFEST.exists() else ""
        if current != rendered:
            print(f"[validate_page_dsl] {MANIFEST.relative_to(REPO)} is stale — run `python scripts/validate_page_dsl.py`", file=sys.stderr)
            return 1
    else:
        MANIFEST.write_text(rendered, encoding="utf-8")

    n = len(manifest["pages"])
    print(f"[validate_page_dsl] {n} page program(s) valid; manifest {'checked' if args.check else 'written'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
