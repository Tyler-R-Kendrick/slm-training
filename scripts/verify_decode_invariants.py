#!/usr/bin/env python3
"""CI certificate for the decode invariants (docs/design/decode-invariants.md).

Static only — this parses source with :mod:`ast` and reads text, so it runs in
the dependency-light static CI job and cannot be defeated by an import-time
side effect. It certifies six things:

1. Canonical config defaults hold the fail-closed value of every registered
   constraint-weakening lever (I6/I2).
2. Named strict evaluation policies pin those safe values explicitly (I6).
3. Every registered serving backend fails closed rather than returning
   uncertified text (I6).
4. Every registered decode backend has a deterministic-bypass test (I2).
5. Every agent surface carries the invariants (I7), and every doc that must
   link the canonical statement does (I8/I15).
6. The reserved operator-token channel has not drifted from default-off
   without a documented decision (I13).

Run: ``python -m scripts.verify_decode_invariants``
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DOC = "docs/design/decode-invariants.md"

# (module, dataclass) pairs whose field defaults are the repository's answer to
# "what happens when nobody configures anything".
CANONICAL_CONFIGS: tuple[tuple[str, str], ...] = (
    ("src/slm_training/models/twotower.py", "TwoTowerConfig"),
    ("src/slm_training/harnesses/model_build/config.py", "ModelBuildConfig"),
)

# Strict policies are ship-gated paths: they must state the safe value rather
# than inherit it, so a future default change cannot silently unpin them.
STRICT_POLICIES: tuple[tuple[str, str], ...] = (
    ("src/slm_training/harnesses/model_build/eval_policy.py", "STRICT_EVALUATION_POLICY"),
)

# A serving backend hands output to a real consumer. Each must contain the
# named fail-closed marker; returning uncertified text is the regression this
# check exists to catch.
SERVING_FAIL_CLOSED: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "src/slm_training/models/onnx_inference.py",
        ("raise GrammarCertificationError",),
    ),
    (
        "src/slm_training/web/service.py",
        ("raise GenerationExhausted", "require_constrained_production_config"),
    ),
)

# I2: a decode backend without a bypass test is a backend that will regress.
BYPASS_TESTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "torch twotower",
        ("tests/test_models/test_inference_speed.py", "forwards_count == 0"),
    ),
    (
        "onnx serving",
        (
            "tests/test_web/test_onnx_inference.py",
            "last_forced_tokens_without_forward",
        ),
    ),
)

# I7: every agent surface must carry the invariants, not merely exist.
AGENT_SURFACES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("AGENTS.md", ("Non-negotiable architecture invariants", CANONICAL_DOC)),
    ("CLAUDE.md", ("Non-negotiable architecture invariants", CANONICAL_DOC)),
    ("GEMINI.md", ("Non-negotiable architecture invariants", CANONICAL_DOC)),
    (
        ".github/copilot-instructions.md",
        ("Non-negotiable architecture invariants", "AGENTS.md"),
    ),
    (".cursor/rules/decode-invariants.mdc", ("alwaysApply: true", CANONICAL_DOC)),
    (".agents/skills/autotrain/SKILL.md", ("decode-invariants.md",)),
    (".agents/skills/honest-ship-eval/SKILL.md", ("decode-invariants.md",)),
    (".agents/skills/improve-openui-harnesses/SKILL.md", ("decode-invariants.md",)),
    (".agents/skills/running-experiment-matrices/SKILL.md", ("decode-invariants.md",)),
)

# I8/I15: the canonical doc is only canonical if the docs that describe these
# subsystems point at it.
LINKING_DOCS: tuple[str, ...] = (
    "README.md",
    "docs/MODEL_CARD.md",
    "docs/design/grammar-fastpath.md",
    "docs/design/symbol-only-output-contract.md",
    "docs/design/dsl-native-tokenizer.md",
    "docs/design/speculative-denoising.md",
    "docs/design/dsh3-08-conversation-state-graph-20260723.md",
    "docs/openwiki/INSTRUCTIONS.md",
)


class DecodeInvariantError(AssertionError):
    """A decode invariant regressed."""


def _read(relative: str) -> str:
    path = ROOT / relative
    if not path.is_file():
        raise DecodeInvariantError(f"missing required file: {relative}")
    return path.read_text(encoding="utf-8")


def _module(relative: str) -> ast.Module:
    return ast.parse(_read(relative), filename=relative)


def _weakening_levers() -> dict[str, Any]:
    """Read the registry statically so the check needs no torch import."""
    tree = _module("src/slm_training/levers.py")
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign) and not isinstance(node, ast.Assign):
            continue
        targets = (
            [node.target] if isinstance(node, ast.AnnAssign) else list(node.targets)
        )
        names = {t.id for t in targets if isinstance(t, ast.Name)}
        if "CONSTRAINT_WEAKENING_LEVERS" not in names or node.value is None:
            continue
        registry = ast.literal_eval(node.value)
        if not registry:
            raise DecodeInvariantError("CONSTRAINT_WEAKENING_LEVERS is empty")
        return registry
    raise DecodeInvariantError(
        "levers.py no longer defines CONSTRAINT_WEAKENING_LEVERS"
    )


def _dataclass_defaults(relative: str, class_name: str) -> dict[str, Any]:
    tree = _module(relative)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            defaults: dict[str, Any] = {}
            for statement in node.body:
                if not isinstance(statement, ast.AnnAssign) or statement.value is None:
                    continue
                if not isinstance(statement.target, ast.Name):
                    continue
                try:
                    defaults[statement.target.id] = ast.literal_eval(statement.value)
                except ValueError:
                    continue
            return defaults
    raise DecodeInvariantError(f"{relative} no longer defines {class_name}")


def _dict_literal(relative: str, name: str) -> dict[str, Any]:
    tree = _module(relative)
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        elif isinstance(node, ast.Assign):
            targets = list(node.targets)
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == name for t in targets):
            continue
        value = node.value
        literal: dict[str, Any] = {}
        if isinstance(value, ast.Dict):
            for key, item in zip(value.keys, value.values, strict=True):
                if key is None:
                    continue
                try:
                    literal[ast.literal_eval(key)] = ast.literal_eval(item)
                except ValueError:
                    continue
        return literal
    raise DecodeInvariantError(f"{relative} no longer defines {name}")


def check_canonical_defaults(levers: dict[str, Any]) -> list[str]:
    """I6/I2: unconfigured is fail-closed."""
    checked: list[str] = []
    for relative, class_name in CANONICAL_CONFIGS:
        defaults = _dataclass_defaults(relative, class_name)
        for lever, spec in levers.items():
            if lever not in defaults:
                continue
            safe = spec["safe_value"]
            actual = defaults[lever]
            if bool(actual) != bool(safe):
                raise DecodeInvariantError(
                    f"{class_name}.{lever} defaults to {actual!r}; decode invariant "
                    f"{spec['invariant']} requires {safe!r} "
                    f"(see {CANONICAL_DOC})"
                )
            checked.append(f"{class_name}.{lever}")
    if not checked:
        raise DecodeInvariantError(
            "no canonical config exposes a registered weakening lever; "
            "the registry and the configs have drifted apart"
        )
    return checked


def check_strict_policies(levers: dict[str, Any]) -> list[str]:
    """I6: a ship-gated policy pins the safe values, it does not inherit them."""
    checked: list[str] = []
    for relative, name in STRICT_POLICIES:
        policy = _dict_literal(relative, name)
        for lever, spec in levers.items():
            if lever not in policy:
                raise DecodeInvariantError(
                    f"{name} does not pin {lever}; a ship-gated policy must state "
                    f"{spec['safe_value']!r} explicitly (see {CANONICAL_DOC})"
                )
            if bool(policy[lever]) != bool(spec["safe_value"]):
                raise DecodeInvariantError(
                    f"{name}.{lever} is {policy[lever]!r}; decode invariant "
                    f"{spec['invariant']} requires {spec['safe_value']!r}"
                )
            checked.append(f"{name}.{lever}")
    return checked


def check_serving_fails_closed() -> list[str]:
    """I6: a serving backend raises rather than returning uncertified output."""
    checked: list[str] = []
    for relative, markers in SERVING_FAIL_CLOSED:
        text = _read(relative)
        for marker in markers:
            if marker not in text:
                raise DecodeInvariantError(
                    f"{relative} lost its fail-closed marker {marker!r}; a serving "
                    f"path may not return uncertified output (see {CANONICAL_DOC})"
                )
            checked.append(f"{relative}:{marker}")
    return checked


def check_bypass_tests() -> list[str]:
    """I2: every decode backend proves its singletons cost no forward."""
    checked: list[str] = []
    for backend, (relative, marker) in BYPASS_TESTS:
        text = _read(relative)
        if marker not in text:
            raise DecodeInvariantError(
                f"{backend} lost its deterministic-bypass assertion "
                f"({marker!r} in {relative}); a decode path without one may not "
                f"merge (see {CANONICAL_DOC})"
            )
        checked.append(backend)
    return checked


def check_agent_surfaces() -> list[str]:
    """I7: every agent surface carries the invariants."""
    checked: list[str] = []
    for relative, markers in AGENT_SURFACES:
        text = _read(relative)
        for marker in markers:
            if marker not in text:
                raise DecodeInvariantError(
                    f"{relative} does not carry {marker!r}; every agent surface "
                    f"must enforce the invariants (see {CANONICAL_DOC})"
                )
        checked.append(relative)
    return checked


def check_docs_link_canonical() -> list[str]:
    """I8/I15: the canonical doc exists and is reachable from every anchor."""
    doc = _read(CANONICAL_DOC)
    for required in ("I2", "I3", "I4", "I6", "I11", "I12", "I13", "I14"):
        if f"### {required} " not in doc and f"### {required}b " not in doc:
            raise DecodeInvariantError(
                f"{CANONICAL_DOC} no longer documents invariant {required}"
            )
    checked: list[str] = []
    for relative in LINKING_DOCS:
        if "decode-invariants.md" not in _read(relative):
            raise DecodeInvariantError(
                f"{relative} does not link {CANONICAL_DOC}; the canonical "
                "statement must stay reachable"
            )
        checked.append(relative)
    return checked


def check_reserved_ops_default_off() -> str:
    """I13: the reserved op-token channel stays default-off until certified."""
    defaults = _dataclass_defaults(
        "src/slm_training/dsl/operators/reserved_tokens.py",
        "ReservedOperatorTokenConfigV1",
    )
    if defaults.get("enabled") is not False:
        raise DecodeInvariantError(
            "ReservedOperatorTokenConfigV1.enabled is no longer default-off; "
            "e803 rejected decoder-target op tokens, and re-enabling them needs "
            f"a preregistered campaign (see {CANONICAL_DOC})"
        )
    return "reserved_operator_tokens=default_off"


def certify() -> dict[str, Any]:
    levers = _weakening_levers()
    return {
        "canonical_doc": CANONICAL_DOC,
        "weakening_levers": sorted(levers),
        "canonical_defaults": check_canonical_defaults(levers),
        "strict_policies": check_strict_policies(levers),
        "serving_fail_closed": check_serving_fails_closed(),
        "bypass_tests": check_bypass_tests(),
        "agent_surfaces": check_agent_surfaces(),
        "linking_docs": check_docs_link_canonical(),
        "reserved_ops": check_reserved_ops_default_off(),
    }


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        print(json.dumps(certify(), indent=2, sort_keys=True))
    except DecodeInvariantError as exc:
        print(f"decode-invariant regression: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
