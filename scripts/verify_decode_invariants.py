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
   link the canonical statement does (I15).
6. The reserved operator-token channel has not drifted from default-off
   without a documented decision, and the shared encoder/decoder ops
   vocabulary matches its committed fingerprint (I13).
7. Every registered model variant's contract (VAR0-01) matches its live
   sources, and no variant falsely claims pack-derived component inventory.

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

# The mandatory generation floor applies to every evaluation policy, including
# checkpoint-declared. It must state each safe value rather than inherit it, so
# a future default change cannot silently unpin a ship-gated path.
STRICT_POLICIES: tuple[tuple[str, str], ...] = (
    (
        "src/slm_training/harnesses/model_build/eval_policy.py",
        "MANDATORY_GENERATION_POLICY",
    ),
)

# A serving backend hands output to a real consumer. Each must contain the
# named fail-closed marker; returning uncertified text is the regression this
# check exists to catch.
SERVING_FAIL_CLOSED: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "src/slm_training/models/onnx_inference.py",
        ("require_constrained_generation", "fallback_used"),
    ),
    (
        "src/slm_training/web/service.py",
        (
            "raise GenerationExhausted",
            "require_constrained_production_config",
            "_raise_on_substituted_generation",
        ),
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
        ("tests/test_web/test_onnx_inference.py", "singleton_bypasses"),
    ),
)

# I7: every agent surface must carry the invariants, not merely exist. The
# matrix itself lives in scripts.verify_agent_surfaces, which owns the same
# check for every other repository law; keeping two copies is what let the
# other laws drift in the first place.
AGENT_SURFACE_OBLIGATION = "decode.invariants"

# I15: the canonical doc is only canonical if the docs that describe these
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


def _lever_registry(registry_name: str) -> dict[str, Any]:
    """Read a lever registry statically so the check needs no torch import."""
    tree = _module("src/slm_training/levers.py")
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign) and not isinstance(node, ast.Assign):
            continue
        targets = (
            [node.target] if isinstance(node, ast.AnnAssign) else list(node.targets)
        )
        names = {t.id for t in targets if isinstance(t, ast.Name)}
        if registry_name not in names or node.value is None:
            continue
        registry = ast.literal_eval(node.value)
        if not registry:
            raise DecodeInvariantError(f"{registry_name} is empty")
        return registry
    raise DecodeInvariantError(f"levers.py no longer defines {registry_name}")


def _weakening_levers() -> dict[str, Any]:
    return _lever_registry("CONSTRAINT_WEAKENING_LEVERS")


def _capacity_levers() -> dict[str, Any]:
    """Invariant VI: every capacity knob stays declared with its baseline.

    Deleting or emptying this registry would make model growth invisible to
    size-matching and to EG_params, which is exactly how a scaled-up model
    gets credited with a capability win.
    """
    registry = _lever_registry("CAPACITY_SCALING_LEVERS")
    missing = [
        name
        for name, spec in sorted(registry.items())
        if "baseline_value" not in spec or not spec.get("axis")
    ]
    if missing:
        raise DecodeInvariantError(
            "CAPACITY_SCALING_LEVERS entries missing baseline_value/axis: "
            + ", ".join(missing)
        )
    return registry


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
    from scripts.verify_agent_surfaces import AgentSurfaceError
    from scripts.verify_agent_surfaces import check as check_surfaces

    try:
        checked = check_surfaces(obligation_id=AGENT_SURFACE_OBLIGATION)
    except AgentSurfaceError as exc:
        raise DecodeInvariantError(f"{exc} (see {CANONICAL_DOC})") from exc
    return checked[AGENT_SURFACE_OBLIGATION]


def check_docs_link_canonical() -> list[str]:
    """I15: the canonical doc exists and is reachable from every anchor."""
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


def check_ops_vocab() -> dict[str, Any]:
    """I13: the shared ops vocabulary is derived, pinned, and layered.

    This one check imports, because the vocabulary's whole point is that it is
    derived from the live operator registries rather than authored. A pure text
    check could not tell a real derivation from a stale copy of one.
    """
    try:
        from slm_training.dsl.ops_vocab import (
            OPS_NAMESPACE,
            assert_layering,
            fingerprint,
            load_registry,
            manifest,
        )
        from slm_training.dsl.openui_tokens import (
            TOKEN_ID_NAMESPACE_RANGES,
            logical_token_id,
        )
    except Exception as exc:  # noqa: BLE001
        raise DecodeInvariantError(f"ops vocabulary is unimportable: {exc}") from exc

    if OPS_NAMESPACE not in TOKEN_ID_NAMESPACE_RANGES:
        raise DecodeInvariantError(
            f"token-id namespace {OPS_NAMESPACE!r} is gone; the reserved ops "
            f"vocabulary has no home (see {CANONICAL_DOC})"
        )
    try:
        committed = load_registry()
    except Exception as exc:  # noqa: BLE001
        raise DecodeInvariantError(
            f"committed ops-vocab registry is unreadable: {exc}"
        ) from exc
    live = manifest()
    if committed.get("fingerprint") != fingerprint():
        raise DecodeInvariantError(
            "ops vocabulary drifted from its committed registry; an operator "
            "was added, removed, or reclassified. Rebuild with "
            "`python -c 'from slm_training.dsl.ops_vocab import write_registry;"
            " write_registry()'` and bump `ops.vocab` in versions.json "
            f"(see {CANONICAL_DOC})"
        )
    if committed != live:
        raise DecodeInvariantError("committed ops-vocab registry is stale")
    if not live["count"]:
        raise DecodeInvariantError("reserved ops vocabulary is empty")

    # Grammar symbols must layer above ops, never inside the reserved range.
    grammar_ids = {
        f"openui:{index}": logical_token_id("openui", index) for index in range(64)
    }
    assert_layering(grammar_ids)
    return {"fingerprint": live["fingerprint"], "count": live["count"]}


def check_variant_contracts() -> dict[str, Any]:
    """VAR0-01: every registered model variant matches its live sources.

    This one check imports, like ``check_ops_vocab``, because the point is
    comparing a committed registry against values derived from each
    variant's live sources (module-level action constants, the live operator
    registry, the pack registry) — a pure text check could not tell an
    honest classification from a stale or false one. Building
    ``slm_training.dsl.variants`` itself stays torch-free, so this adds no
    heavy dependency to the static CI job.
    """
    try:
        from slm_training.dsl.variants import fingerprint, load_registry, manifest
    except Exception as exc:  # noqa: BLE001
        raise DecodeInvariantError(f"variant registry is unimportable: {exc}") from exc

    try:
        committed = load_registry()
    except Exception as exc:  # noqa: BLE001
        raise DecodeInvariantError(
            f"committed variant registry is unreadable: {exc}"
        ) from exc
    live = manifest()
    if committed.get("fingerprint") != fingerprint():
        raise DecodeInvariantError(
            "variant registry drifted from its committed contracts; a variant "
            "was added, removed, or reclassified. Rebuild with "
            "`python -c 'from slm_training.dsl.variants import write_registry;"
            " write_registry()'` and bump `dsl.variants` in versions.json "
            f"(see {CANONICAL_DOC})"
        )
    if committed != live:
        raise DecodeInvariantError("committed variant registry is stale")
    if not live["count"]:
        raise DecodeInvariantError("variant registry is empty")
    return {"fingerprint": live["fingerprint"], "count": live["count"]}


def certify() -> dict[str, Any]:
    levers = _weakening_levers()
    return {
        "canonical_doc": CANONICAL_DOC,
        "weakening_levers": sorted(levers),
        "capacity_levers": sorted(_capacity_levers()),
        "canonical_defaults": check_canonical_defaults(levers),
        "strict_policies": check_strict_policies(levers),
        "serving_fail_closed": check_serving_fails_closed(),
        "bypass_tests": check_bypass_tests(),
        "agent_surfaces": check_agent_surfaces(),
        "linking_docs": check_docs_link_canonical(),
        "reserved_ops": check_reserved_ops_default_off(),
        "ops_vocab": check_ops_vocab(),
        "variant_contracts": check_variant_contracts(),
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
