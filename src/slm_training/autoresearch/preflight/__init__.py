"""Discovery-based preflight gate seam for the continuous autotrain loop.

Motivation (RC3, ``docs/design/harness-evolution-architecture-review-20260809.md``):
the continuous loop rediscovered and rejected the identical 0.3267→0.3828 delta
in nine separate loops because nothing consulted the durable evidence record
before spending a screening cycle. This package is the integration seam that
closes that gap: the driver builds a **candidate dict** for the arm it is about
to run, calls :func:`run_preflight`, and skips any candidate that draws a
``"block"`` verdict.

Contract
--------

Every module in this package that exposes a module-level ``CHECK`` object
satisfying :class:`PreflightCheck` is discovered automatically via
``pkgutil.iter_modules`` — sibling plugins (e.g. ``power_check.py``,
``concluded_family.py``) attach by merely existing in this directory; nothing
registers them by name and their absence is never an error.

Candidate dict shape (all keys optional — checks must tolerate missing keys)::

    {
        "hypothesis_text": str,        # arm hypothesis sentence
        "lever_keys": list[str],       # sorted lever/config keys being moved
        "config_fingerprint": str,     # canonical config identity hash
        "n_seeds": int,                # seeds this attempt would spend
        "steps": int,                  # train steps per arm
        "minimum_effect": float,       # policy minimum effect for the endpoint
        "endpoint_metric": str,        # e.g. "smoke.structural_similarity"
    }

Extra keys are allowed (the driver also passes ``slug`` and ``levers`` so
checks can fall back to slug identity / recompute fingerprints); checks must
ignore keys they do not understand.

Fail-soft law: a crashing check yields a ``"warn"`` verdict carrying the
traceback summary — a preflight bug must never take down the loop, and a
crash must never silently *block* a candidate either.
"""

from __future__ import annotations

import importlib
import pkgutil
import traceback
from typing import Iterable, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

__all__ = [
    "PreflightCheck",
    "PreflightVerdict",
    "has_block",
    "run_preflight",
]


class PreflightVerdict(BaseModel):
    """Typed outcome of one preflight check against one candidate."""

    check_id: str
    verdict: Literal["pass", "warn", "block"]
    reasons: list[str]
    data: dict = Field(default_factory=dict)


@runtime_checkable
class PreflightCheck(Protocol):
    """A pluggable preflight check: module-level ``CHECK`` objects satisfy this."""

    check_id: str

    def run(self, candidate: dict) -> PreflightVerdict: ...


def _crash_summary(exc: BaseException) -> str:
    """One-line traceback summary safe to embed in a verdict reason."""
    frames = traceback.extract_tb(exc.__traceback__)
    location = ""
    if frames:
        last = frames[-1]
        location = f" at {last.filename.rsplit('/', 1)[-1]}:{last.lineno}"
    return f"{type(exc).__name__}: {exc}{location}"


def _discover_module_names() -> list[str]:
    return sorted(
        info.name
        for info in pkgutil.iter_modules(__path__)
        if not info.name.startswith("_")
    )


def run_preflight(candidate: dict) -> list[PreflightVerdict]:
    """Run every discovered preflight check against ``candidate``.

    Discovers every module in this package exposing a module-level ``CHECK``
    object, runs all of them, and returns the verdicts sorted by ``check_id``.
    A single check's crash (import failure, ``run`` raising, or a malformed
    verdict) becomes a ``"warn"`` verdict with the traceback summary in
    ``reasons`` — never an exception and never an implicit block.
    """
    verdicts: list[PreflightVerdict] = []
    for name in _discover_module_names():
        try:
            module = importlib.import_module(f"{__name__}.{name}")
        except Exception as exc:  # noqa: BLE001 — fail-soft per package law
            verdicts.append(
                PreflightVerdict(
                    check_id=name,
                    verdict="warn",
                    reasons=[f"plugin import failed: {_crash_summary(exc)}"],
                )
            )
            continue
        check = getattr(module, "CHECK", None)
        if check is None:
            continue
        check_id = str(getattr(check, "check_id", name))
        try:
            verdict = check.run(dict(candidate))
            if not isinstance(verdict, PreflightVerdict):
                verdict = PreflightVerdict.model_validate(verdict)
        except Exception as exc:  # noqa: BLE001 — fail-soft per package law
            verdicts.append(
                PreflightVerdict(
                    check_id=check_id,
                    verdict="warn",
                    reasons=[f"check crashed: {_crash_summary(exc)}"],
                )
            )
            continue
        verdicts.append(verdict)
    return sorted(verdicts, key=lambda item: item.check_id)


def has_block(verdicts: Iterable[PreflightVerdict]) -> bool:
    """True when any verdict blocks — the candidate must not be run."""
    return any(item.verdict == "block" for item in verdicts)
