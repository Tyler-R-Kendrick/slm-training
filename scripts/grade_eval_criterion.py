#!/usr/bin/env python3
"""Deterministic AgentEvals grader for one raw gate criterion."""

from __future__ import annotations

import json
import math
import sys
from typing import Any


def _finite_real(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _integral_count(value: Any) -> bool:
    return _finite_real(value) and float(value).is_integer() and float(value) >= 0


def _passes(actual: Any, operator: str, expected: Any, criterion_id: str) -> bool:
    if operator == "present":
        return actual is not None
    if operator == "eq":
        if criterion_id.endswith(":certified_fallback"):
            return _integral_count(actual) and int(actual) == expected
        return actual is not None and actual == expected
    if operator == "gte":
        valid = (
            _integral_count(actual)
            if criterion_id.endswith(":insufficient_n")
            else _finite_real(actual)
        )
        return valid and _finite_real(expected) and float(actual) >= float(expected)
    raise ValueError(f"unsupported criterion operator: {operator!r}")


def main() -> int:
    payload = json.load(sys.stdin)
    criterion = payload.get("config") or {}
    criterion_id = str(criterion.get("id", "unnamed"))
    actual = criterion.get("actual")
    operator = str(criterion.get("operator", ""))
    expected = criterion.get("expected")
    passed = _passes(actual, operator, expected, criterion_id)
    json.dump(
        {
            "score": 1 if passed else 0,
            "assertions": [
                {
                    "text": criterion_id,
                    "passed": passed,
                    "evidence": (
                        f"actual={actual!r} operator={operator} expected={expected!r}"
                    ),
                }
            ],
            "details": {
                "criterion_id": criterion_id,
                "actual": actual,
                "operator": operator,
                "expected": expected,
            },
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
