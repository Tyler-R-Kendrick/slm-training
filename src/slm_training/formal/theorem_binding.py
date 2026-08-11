"""EVID-07 exact Lean theorem binding (proposition + environment + axioms).

A sealed binding proves the named declaration has the expected elaborated
proposition in a specific source/toolchain/dependency environment — not merely
that a same-named declaration exists and the project builds. Project-wide
forbidden-proof / axiom-purity audit remains an *additional* check.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LEAN_THEOREM_BINDING_SCHEMA = "lean.theorem_binding/v1"

_DECL_START_RE = re.compile(
    r"(?m)^(?P<indent>\s*)(?:theorem|lemma)\s+(?P<name>[A-Za-z_][A-Za-z0-9_'!]*)\b"
)
_AXIOM_LIST_RE = re.compile(
    r"depends on axioms:\s*\[(?P<body>[^\]]*)\]",
    re.IGNORECASE,
)
_AXIOM_NONE_RE = re.compile(
    r"does not depend on any axioms",
    re.IGNORECASE,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_proposition(text: str) -> str:
    """Collapse whitespace for portable proposition fingerprints."""

    return " ".join(text.split())


def proposition_fingerprint(text: str) -> str:
    return _sha256(normalize_proposition(text))


def axiom_footprint_digest(axioms: Sequence[str]) -> str:
    return _sha256(_canonical_json(sorted({str(item) for item in axioms})))


def parse_axiom_print(output: str, *, fq_name: str) -> tuple[str, ...]:
    """Parse ``#print axioms`` stdout for one declaration into a sorted tuple."""

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    focused = [line for line in lines if fq_name in line] or lines
    text = "\n".join(focused)
    if _AXIOM_NONE_RE.search(text):
        return ()
    match = _AXIOM_LIST_RE.search(text)
    if match is None:
        # Fail closed: unparseable axiom prints are not an empty footprint.
        raise ValueError(f"unparseable #print axioms output for {fq_name!r}")
    body = match.group("body").strip()
    if not body:
        return ()
    parts = [part.strip() for part in body.split(",") if part.strip()]
    return tuple(sorted(parts))


def extract_declaration_type(source: str, *, local_name: str) -> str:
    """Extract the binder/type ascription between ``theorem name`` and ``:=``."""

    for match in _DECL_START_RE.finditer(source):
        if match.group("name") != local_name:
            continue
        start = match.end()
        depth_paren = depth_brack = depth_brace = 0
        i = start
        while i < len(source) - 1:
            ch = source[i]
            nxt = source[i + 1]
            if ch == "-" and nxt == "-":
                while i < len(source) and source[i] not in "\n\r":
                    i += 1
                continue
            if ch == "/" and nxt == "-":
                i += 2
                while i < len(source) - 1 and not (
                    source[i] == "-" and source[i + 1] == "/"
                ):
                    i += 1
                i += 2
                continue
            if ch == "(":
                depth_paren += 1
            elif ch == ")":
                depth_paren = max(0, depth_paren - 1)
            elif ch == "[":
                depth_brack += 1
            elif ch == "]":
                depth_brack = max(0, depth_brack - 1)
            elif ch == "{":
                depth_brace += 1
            elif ch == "}":
                depth_brace = max(0, depth_brace - 1)
            elif (
                ch == ":"
                and nxt == "="
                and depth_paren == depth_brack == depth_brace == 0
            ):
                return source[start:i].strip()
            i += 1
        raise ValueError(f"declaration {local_name!r} has no top-level ':='")
    raise ValueError(f"declaration {local_name!r} not found in source")


def ascription_to_expected_proposition(ascription: str) -> str:
    """Turn ``(a : A) : B`` / ``: B`` into a Pi/∀ proposition for ``example``."""

    text = ascription.strip()
    if not text:
        raise ValueError("empty declaration ascription")
    if text.startswith(":"):
        return text[1:].strip()
    # Split final top-level ``: Ret`` from binders.
    depth_paren = depth_brack = depth_brace = 0
    colon_at: int | None = None
    for i, ch in enumerate(text):
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(0, depth_paren - 1)
        elif ch == "[":
            depth_brack += 1
        elif ch == "]":
            depth_brack = max(0, depth_brack - 1)
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace = max(0, depth_brace - 1)
        elif ch == ":" and depth_paren == depth_brack == depth_brace == 0:
            colon_at = i
    if colon_at is None:
        return text
    binders = text[:colon_at].strip()
    ret = text[colon_at + 1 :].strip()
    if not binders:
        return ret
    return f"∀ {binders}, {ret}"


def read_lean_version(lean_root: Path) -> str:
    toolchain = lean_root / "lean-toolchain"
    if toolchain.is_file():
        return toolchain.read_text(encoding="utf-8").strip()
    return "unknown"


def lake_manifest_digest(lean_root: Path) -> str:
    """Digest lake lock + toolchain + lakefile (dependency/toolchain identity)."""

    parts: list[bytes] = []
    for name in ("lean-toolchain", "lakefile.toml", "lakefile.lean", "lake-manifest.json"):
        path = lean_root / name
        if path.is_file():
            parts.append(name.encode("utf-8"))
            parts.append(b"\0")
            parts.append(path.read_bytes())
            parts.append(b"\0")
    if not parts:
        raise FileNotFoundError(f"no lake/toolchain files under {lean_root}")
    return _sha256_bytes(b"".join(parts))


def source_tree_digest(lean_root: Path, *, module: str | None = None) -> str:
    """Digest Lean sources under the package (optionally one module first)."""

    files: list[Path] = []
    if module:
        rel = Path(*module.split(".")).with_suffix(".lean")
        module_path = lean_root / rel
        if module_path.is_file():
            files.append(module_path)
    lean_dirs = [
        lean_root / "LeverProofLean",
        lean_root / "Test",
    ]
    for directory in lean_dirs:
        if not directory.is_dir():
            continue
        files.extend(sorted(directory.rglob("*.lean")))
    # Dedup while preserving order.
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(path)
    if not ordered:
        raise FileNotFoundError(f"no lean sources under {lean_root}")
    h = hashlib.sha256()
    for path in ordered:
        rel = path.relative_to(lean_root).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def render_bridge_lean(
    *,
    module: str,
    fq_name: str,
    expected_proposition: str,
) -> str:
    """Generate a bridge file that type-binds ``fq_name`` to ``ExpectedProposition``."""

    return (
        "-- Auto-generated by EVID-07 theorem binding. Do not edit.\n"
        f"import {module}\n"
        f"#check {fq_name}\n"
        f"def ExpectedProposition := {expected_proposition}\n"
        f"example : ExpectedProposition := {fq_name}\n"
        f"#print axioms {fq_name}\n"
    )


def bridge_digest(bridge_source: str) -> str:
    return _sha256(bridge_source)


@dataclass(frozen=True)
class LeanTheoremBindingV1:
    """Exact proposition + environment seal for one Lean declaration."""

    fq_name: str
    module: str
    expected_proposition: str
    expected_proposition_fingerprint: str
    lean_version: str
    lake_manifest_digest: str
    source_tree_digest: str
    axiom_footprint: tuple[str, ...]
    axiom_footprint_digest: str
    bridge_digest: str
    schema: str = LEAN_THEOREM_BINDING_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "fq_name": self.fq_name,
            "module": self.module,
            "expected_proposition": self.expected_proposition,
            "expected_proposition_fingerprint": self.expected_proposition_fingerprint,
            "lean_version": self.lean_version,
            "lake_manifest_digest": self.lake_manifest_digest,
            "source_tree_digest": self.source_tree_digest,
            "axiom_footprint": list(self.axiom_footprint),
            "axiom_footprint_digest": self.axiom_footprint_digest,
            "bridge_digest": self.bridge_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LeanTheoremBindingV1:
        if data.get("schema") != LEAN_THEOREM_BINDING_SCHEMA:
            raise ValueError(
                f"expected schema {LEAN_THEOREM_BINDING_SCHEMA!r}, "
                f"got {data.get('schema')!r}"
            )
        axioms = tuple(str(item) for item in data.get("axiom_footprint", ()))
        expected = str(data["expected_proposition"])
        prop_fp = str(data["expected_proposition_fingerprint"])
        if proposition_fingerprint(expected) != prop_fp:
            raise ValueError("expected_proposition fingerprint mismatch")
        ax_fp = str(data["axiom_footprint_digest"])
        if axiom_footprint_digest(axioms) != ax_fp:
            raise ValueError("axiom_footprint digest mismatch")
        bridge = render_bridge_lean(
            module=str(data["module"]),
            fq_name=str(data["fq_name"]),
            expected_proposition=expected,
        )
        declared_bridge = str(data["bridge_digest"])
        if bridge_digest(bridge) != declared_bridge:
            raise ValueError("bridge_digest mismatch with rendered bridge")
        return cls(
            fq_name=str(data["fq_name"]),
            module=str(data["module"]),
            expected_proposition=expected,
            expected_proposition_fingerprint=prop_fp,
            lean_version=str(data["lean_version"]),
            lake_manifest_digest=str(data["lake_manifest_digest"]),
            source_tree_digest=str(data["source_tree_digest"]),
            axiom_footprint=axioms,
            axiom_footprint_digest=ax_fp,
            bridge_digest=declared_bridge,
            schema=str(data["schema"]),
        )

    def source_digest_map(self) -> dict[str, str]:
        """sha256-only map for FormalAuthorityV2.source_digests."""

        return {
            "theorem_binding.expected_proposition": self.expected_proposition_fingerprint,
            "theorem_binding.lake_manifest": self.lake_manifest_digest,
            "theorem_binding.source_tree": self.source_tree_digest,
            "theorem_binding.axiom_footprint": self.axiom_footprint_digest,
            "theorem_binding.bridge": self.bridge_digest,
        }


def seal_theorem_binding(
    *,
    fq_name: str,
    module: str,
    lean_root: Path,
    axiom_footprint: Sequence[str] | None = None,
    lean_version: str | None = None,
) -> LeanTheoremBindingV1:
    """Seal the live source/toolchain proposition environment into a binding."""

    local_name = fq_name.rsplit(".", 1)[-1]
    rel = Path(*module.split(".")).with_suffix(".lean")
    source_path = lean_root / rel
    if not source_path.is_file():
        raise FileNotFoundError(f"lean module source missing: {rel}")
    source = source_path.read_text(encoding="utf-8")
    ascription = extract_declaration_type(source, local_name=local_name)
    expected = ascription_to_expected_proposition(ascription)
    axioms = tuple(sorted({str(item) for item in (axiom_footprint or ())}))
    bridge = render_bridge_lean(
        module=module, fq_name=fq_name, expected_proposition=expected
    )
    return LeanTheoremBindingV1(
        fq_name=fq_name,
        module=module,
        expected_proposition=expected,
        expected_proposition_fingerprint=proposition_fingerprint(expected),
        lean_version=lean_version if lean_version is not None else read_lean_version(lean_root),
        lake_manifest_digest=lake_manifest_digest(lean_root),
        source_tree_digest=source_tree_digest(lean_root, module=module),
        axiom_footprint=axioms,
        axiom_footprint_digest=axiom_footprint_digest(axioms),
        bridge_digest=bridge_digest(bridge),
    )


def verify_theorem_binding(
    binding: LeanTheoremBindingV1,
    *,
    lean_root: Path,
    catalog_module: str | None = None,
    live_axiom_footprint: Sequence[str] | None = None,
    require_axioms: bool = True,
) -> list[str]:
    """Return violation strings (empty means the sealed binding still holds)."""

    violations: list[str] = []
    if catalog_module is not None and catalog_module != binding.module:
        violations.append(
            f"catalog redirection: catalog module {catalog_module!r} != "
            f"binding module {binding.module!r}"
        )

    try:
        live_version = read_lean_version(lean_root)
    except OSError as exc:
        return [f"lean toolchain unreadable: {exc}"]
    if live_version != binding.lean_version:
        violations.append(
            f"toolchain drift: lean-toolchain {live_version!r} != "
            f"sealed {binding.lean_version!r}"
        )

    try:
        live_manifest = lake_manifest_digest(lean_root)
    except (OSError, FileNotFoundError) as exc:
        violations.append(f"lake manifest digest failed: {exc}")
        live_manifest = None
    if live_manifest is not None and live_manifest != binding.lake_manifest_digest:
        violations.append("lake manifest/lock digest drift")

    try:
        live_tree = source_tree_digest(lean_root, module=binding.module)
    except (OSError, FileNotFoundError) as exc:
        violations.append(f"source-tree digest failed: {exc}")
        live_tree = None
    if live_tree is not None and live_tree != binding.source_tree_digest:
        violations.append("source-tree digest drift")

    rel = Path(*binding.module.split(".")).with_suffix(".lean")
    source_path = lean_root / rel
    if not source_path.is_file():
        violations.append(f"lean module source missing: {rel}")
        return violations
    try:
        source = source_path.read_text(encoding="utf-8")
        ascription = extract_declaration_type(
            source, local_name=binding.fq_name.rsplit(".", 1)[-1]
        )
        live_expected = ascription_to_expected_proposition(ascription)
    except (OSError, ValueError) as exc:
        violations.append(f"proposition extraction failed: {exc}")
        return violations

    live_fp = proposition_fingerprint(live_expected)
    if live_fp != binding.expected_proposition_fingerprint:
        violations.append(
            "proposition mutation: live type fingerprint != sealed expected"
        )
    if normalize_proposition(live_expected) != normalize_proposition(
        binding.expected_proposition
    ):
        violations.append(
            "proposition mutation: live ascription != sealed expected_proposition"
        )

    rendered = render_bridge_lean(
        module=binding.module,
        fq_name=binding.fq_name,
        expected_proposition=binding.expected_proposition,
    )
    if bridge_digest(rendered) != binding.bridge_digest:
        violations.append("stale bridge digest")

    if require_axioms or live_axiom_footprint is not None:
        if live_axiom_footprint is None:
            violations.append("missing live axiom footprint for exact declaration")
        else:
            live_axioms = tuple(sorted({str(item) for item in live_axiom_footprint}))
            if live_axioms != binding.axiom_footprint:
                violations.append(
                    "unexpected axiom footprint: "
                    f"live={list(live_axioms)} sealed={list(binding.axiom_footprint)}"
                )
            if axiom_footprint_digest(live_axioms) != binding.axiom_footprint_digest:
                violations.append("axiom footprint digest drift")

    return violations


__all__ = [
    "LEAN_THEOREM_BINDING_SCHEMA",
    "LeanTheoremBindingV1",
    "ascription_to_expected_proposition",
    "axiom_footprint_digest",
    "bridge_digest",
    "extract_declaration_type",
    "lake_manifest_digest",
    "normalize_proposition",
    "parse_axiom_print",
    "proposition_fingerprint",
    "read_lean_version",
    "render_bridge_lean",
    "seal_theorem_binding",
    "source_tree_digest",
    "verify_theorem_binding",
]
