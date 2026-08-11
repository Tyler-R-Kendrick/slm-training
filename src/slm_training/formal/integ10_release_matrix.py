"""INTEG-10 / SLM-576: bounded infrastructure release matrix.

Codifies required vs optional/research gates as a machine-readable matrix,
runs or identity-certifies existing verify_* / pytest owners under finite
process caps, and emits a content-addressed release-evidence summary.

Skip semantics (non-negotiable):
- optional tool missing → ``skipped_optional`` (not ok, not a refutation)
- optional tool timeout → ``skipped_optional_timeout`` (same)
- required timeout → ``timeout`` (failure; never evidence)
- research/default-off → ``research_default_off`` (visible; not required)
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from slm_training.versioning import build_version_stamp


def _workflow_jobs(text: str) -> list[tuple[str, int, int]]:
    """Minimal CI job span parser (mirrors scripts.repo_policy._workflow_jobs)."""

    import re

    lines = text.splitlines(keepends=True)
    jobs_line = next(
        (i for i, line in enumerate(lines) if line.rstrip() == "jobs:"), None
    )
    if jobs_line is None:
        return []
    end = next(
        (
            i
            for i in range(jobs_line + 1, len(lines))
            if lines[i].strip() and not lines[i][0].isspace()
        ),
        len(lines),
    )
    job_re = re.compile(r"^  ([A-Za-z0-9_-]+):\s*$")
    starts = [
        (match.group(1), i)
        for i in range(jobs_line + 1, end)
        if (match := job_re.match(lines[i].rstrip()))
    ]
    return [
        (name, start, starts[index + 1][1] if index + 1 < len(starts) else end)
        for index, (name, start) in enumerate(starts)
    ]

MATRIX_SCHEMA = "integ10_bounded_release_matrix/v1"
REPORT_SCHEMA = "integ10_release_evidence_summary/v1"
MATRIX_RELPATH = (
    "src/slm_training/resources/formal/integ10_release_matrix.v1.json"
)
DEFAULT_RESULTS_RELPATH = "docs/design/integ-10-bounded-release-matrix-results.json"

_REPO = Path(__file__).resolve().parents[3]
CAS_SCHEME = "cas://sha256/"

# Statuses that mean "required gate satisfied" for release.passed.
_REQUIRED_OK = frozenset({"ok", "delegated"})
# Statuses that must never be treated as success for an optional probe.
_OPTIONAL_NON_SUCCESS = frozenset(
    {
        "skipped_optional",
        "skipped_optional_timeout",
        "failed",
        "timeout",
    }
)


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def content_address(payload: Any) -> str:
    """Content-address a JSON-shaped payload (cas://sha256/<digest>)."""

    return f"{CAS_SCHEME}{_sha256_bytes(_canonical_json(payload))}"


def load_matrix(*, root: Path | None = None) -> dict[str, Any]:
    base = root or _REPO
    path = base / MATRIX_RELPATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != MATRIX_SCHEMA:
        raise ValueError(f"bad matrix schema: {payload.get('schema')!r}")
    return payload


def matrix_content_digest(matrix: Mapping[str, Any] | None = None) -> str:
    payload = matrix or load_matrix()
    return _sha256_bytes(_canonical_json(payload))


def _owner_texts(*, root: Path) -> dict[str, str]:
    """Load owner surfaces used for delegate identity certification."""

    ci = root / ".github/workflows/ci.yml"
    ci_text = ci.read_text(encoding="utf-8") if ci.is_file() else ""
    jobs: dict[str, str] = {}
    if ci_text:
        lines = ci_text.splitlines(keepends=True)
        for name, start, end in _workflow_jobs(ci_text):
            jobs[name] = "".join(lines[start:end])
    merge_path = root / "scripts/verify_merge_ready.py"
    merge_text = merge_path.read_text(encoding="utf-8") if merge_path.is_file() else ""
    return {
        "ci.yml": ci_text,
        "ci.python-static": jobs.get("python-static", ""),
        "ci.lean-formal": jobs.get("lean-formal", ""),
        "merge_ready": merge_text,
    }


def _markers_present(text: str, markers: Sequence[str]) -> list[str]:
    return [marker for marker in markers if marker not in text]


def _build_cmd(gate: Mapping[str, Any], *, python: str) -> list[str] | None:
    module = gate.get("module")
    if not module:
        return None
    argv = list(gate.get("argv") or [])
    if module == "pytest":
        return [python, "-m", "pytest", *argv]
    return [python, "-m", module, *argv]


def _display_cmd(cmd: Sequence[str]) -> str:
    """Repo-portable command string (never commit machine-absolute python)."""

    display = list(cmd)
    if display:
        base = Path(display[0]).name
        if base.startswith("python"):
            display[0] = "python"
    return shlex.join(display)


def _run_subprocess(
    cmd: Sequence[str],
    *,
    root: Path,
    budget_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        env = dict(os.environ)
        src = str(root / "src")
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = src if not prev else f"{src}:{prev}"
        result = subprocess.run(
            list(cmd),
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=budget_seconds,
            env=env,
        )
        status = "ok" if result.returncode == 0 else "failed"
        output = (result.stdout or "") + (result.stderr or "")
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        output = "".join(
            part.decode(errors="replace")
            if isinstance(part, bytes)
            else (part or "")
            for part in (exc.stdout, exc.stderr)
        )
    record: dict[str, Any] = {
        "status": status,
        "seconds": round(time.monotonic() - started, 2),
        "cmd": _display_cmd(cmd),
    }
    if status != "ok":
        # Keep tails portable: strip accidental absolute worktree prefixes.
        root_s = str(root)
        record["output_tail"] = output[-4000:].replace(root_s, ".")
    return record


def _probe_optional(
    gate: Mapping[str, Any],
    *,
    root: Path,
    python: str,
    execute: bool,
) -> dict[str, Any]:
    probe = gate.get("probe") or {}
    which = list(probe.get("which") or [])
    missing = [name for name in which if shutil.which(name) is None]
    if missing:
        return {
            "status": "skipped_optional",
            "seconds": 0.0,
            "cmd": None,
            "skip_reason": f"optional tools missing: {', '.join(missing)}",
        }
    if not execute:
        return {
            "status": "skipped_optional",
            "seconds": 0.0,
            "cmd": None,
            "skip_reason": (
                "optional tools present but check mode does not invoke "
                "heavy optional probes; use --execute"
            ),
        }
    cmd = _build_cmd(gate, python=python)
    if not cmd:
        return {
            "status": "skipped_optional",
            "seconds": 0.0,
            "cmd": None,
            "skip_reason": "no module for optional probe",
        }
    result = _run_subprocess(
        cmd,
        root=root,
        budget_seconds=float(gate.get("budget_seconds") or 60),
    )
    if result["status"] == "timeout":
        result["status"] = "skipped_optional_timeout"
        result["skip_reason"] = "optional probe timed out; not a refutation"
    elif result["status"] == "failed":
        # Optional probe failure is reported but does not refute infrastructure.
        result["skip_reason"] = (
            "optional probe failed; not counted as infrastructure refutation"
        )
    return result


def _certify_delegate(
    gate: Mapping[str, Any],
    *,
    owners: Mapping[str, str],
) -> dict[str, Any]:
    owner = str(gate.get("owner") or "")
    markers = list(gate.get("owner_markers") or [])
    # Always check merge_ready for python-static-owned gates (local gate parity).
    surfaces = [owners.get(owner, "")]
    if owner == "ci.python-static":
        surfaces.append(owners.get("merge_ready", ""))
    if owner == "ci.lean-formal":
        surfaces.append(owners.get("ci.yml", ""))
    blob = "\n".join(surfaces)
    missing = _markers_present(blob, markers) if markers else []
    if missing:
        return {
            "status": "failed",
            "seconds": 0.0,
            "cmd": None,
            "output_tail": (
                f"delegate identity missing markers {missing!r} in owner {owner!r}"
            ),
        }
    return {
        "status": "delegated",
        "seconds": 0.0,
        "cmd": None,
        "delegated_owner": owner,
        "markers_verified": markers,
    }


def run_gate(
    gate: Mapping[str, Any],
    *,
    root: Path,
    python: str,
    execute: bool,
    owners: Mapping[str, str],
) -> dict[str, Any]:
    """Run or certify one matrix gate; return a machine-readable row."""

    mode = str(gate.get("check_mode") or "invoke")
    category = str(gate.get("category") or "required")
    base = {
        "gate_id": gate["id"],
        "category": category,
        "check_mode": mode,
        "owner": gate.get("owner"),
        "module": gate.get("module"),
        "identities": gate.get("identities") or {},
        "description": gate.get("description"),
    }

    if mode == "status_only" or category == "research_default_off":
        payload = dict(gate.get("status_payload") or {})
        return {
            **base,
            "status": "research_default_off",
            "seconds": 0.0,
            "cmd": None,
            "research_status": payload,
        }

    if mode == "probe_optional" or category == "optional_tool":
        result = _probe_optional(
            gate, root=root, python=python, execute=execute
        )
        return {**base, **result}

    if mode == "delegate" and not execute:
        result = _certify_delegate(gate, owners=owners)
        return {**base, **result}

    # invoke (or execute=True for delegated gates)
    cmd = _build_cmd(gate, python=python)
    if not cmd:
        return {
            **base,
            "status": "failed",
            "seconds": 0.0,
            "cmd": None,
            "output_tail": "missing module for required gate",
        }
    result = _run_subprocess(
        cmd,
        root=root,
        budget_seconds=float(gate.get("budget_seconds") or 60),
    )
    if mode == "delegate" and result["status"] == "ok":
        # Still record owner identity alongside the live invoke.
        cert = _certify_delegate(gate, owners=owners)
        result["delegated_owner"] = cert.get("delegated_owner")
        result["markers_verified"] = cert.get("markers_verified")
        if cert["status"] == "failed":
            result = {**result, **cert}
    return {**base, **result}


def run_matrix(
    *,
    root: Path | None = None,
    execute: bool = False,
    python: str | None = None,
) -> dict[str, Any]:
    """Execute/certify the bounded release matrix and build the evidence summary."""

    base = root or _REPO
    matrix = load_matrix(root=base)
    owners = _owner_texts(root=base)
    py = python or sys.executable
    rows = [
        run_gate(
            gate,
            root=base,
            python=py,
            execute=execute,
            owners=owners,
        )
        for gate in matrix["gates"]
    ]

    required = [row for row in rows if row["category"] == "required"]
    optional = [row for row in rows if row["category"] == "optional_tool"]
    research = [row for row in rows if row["category"] == "research_default_off"]

    required_failed = [
        row
        for row in required
        if row["status"] not in _REQUIRED_OK
    ]
    # Optional non-success must not flip release.passed, and must not look like ok.
    optional_reported = [
        {
            "gate_id": row["gate_id"],
            "status": row["status"],
            "skip_reason": row.get("skip_reason"),
            "counted_as_success": False,
            "counted_as_refutation": False,
        }
        for row in optional
        if row["status"] in _OPTIONAL_NON_SUCCESS or row["status"] == "ok"
    ]
    for item in optional_reported:
        if item["status"] == "ok":
            item["counted_as_success"] = False  # still optional; informational only

    passed = not required_failed
    gate_identity_table = [
        {
            "gate_id": row["gate_id"],
            "category": row["category"],
            "status": row["status"],
            "owner": row.get("owner"),
            "module": row.get("module"),
            "identities": row.get("identities") or {},
            "content_address": content_address(
                {
                    "gate_id": row["gate_id"],
                    "identities": row.get("identities") or {},
                    "module": row.get("module"),
                    "owner": row.get("owner"),
                }
            ),
        }
        for row in rows
    ]

    evidence_body = {
        "schema": REPORT_SCHEMA,
        "integ": "INTEG-10",
        "issue": "SLM-576",
        "passed": passed,
        "execute": execute,
        "matrix_schema": MATRIX_SCHEMA,
        "matrix_relpath": MATRIX_RELPATH,
        "matrix_sha256": matrix_content_digest(matrix),
        "matrix_content_address": f"{CAS_SCHEME}{matrix_content_digest(matrix)}",
        "required_count": len(required),
        "required_ok_count": len(required) - len(required_failed),
        "required_failed": [row["gate_id"] for row in required_failed],
        "optional_status": optional_reported,
        "research_default_off": [
            {
                "gate_id": row["gate_id"],
                "status": row["status"],
                "payload": row.get("research_status") or {},
                "required_for_release": False,
            }
            for row in research
        ],
        "gates": rows,
        "gate_identity_table": gate_identity_table,
        "empirical_performance_separate": True,
        "no_optional_skip_is_success": True,
        "no_optional_skip_is_refutation": True,
        "infrastructure_release_needs_research_pilot": False,
    }
    evidence_body["evidence_sha256"] = _sha256_bytes(_canonical_json(evidence_body))
    evidence_body["evidence_content_address"] = (
        f"{CAS_SCHEME}{evidence_body['evidence_sha256']}"
    )
    evidence_body["version_stamp"] = build_version_stamp("formal.objects")
    return evidence_body


def write_report(report: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def stable_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Stamp-free fields for --check drift detection."""

    return {
        "schema": report.get("schema"),
        "integ": report.get("integ"),
        "issue": report.get("issue"),
        "passed": report.get("passed"),
        "execute": report.get("execute"),
        "matrix_schema": report.get("matrix_schema"),
        "matrix_relpath": report.get("matrix_relpath"),
        "matrix_sha256": report.get("matrix_sha256"),
        "required_count": report.get("required_count"),
        "required_ok_count": report.get("required_ok_count"),
        "required_failed": report.get("required_failed"),
        "optional_status": [
            {
                "gate_id": item["gate_id"],
                "status": item["status"],
                "counted_as_success": item.get("counted_as_success"),
                "counted_as_refutation": item.get("counted_as_refutation"),
            }
            for item in report.get("optional_status") or []
        ],
        "research_default_off": [
            {
                "gate_id": item["gate_id"],
                "status": item["status"],
                "required_for_release": item.get("required_for_release"),
                "payload": {
                    "required_for_release": (item.get("payload") or {}).get(
                        "required_for_release"
                    ),
                    "infrastructure_release_needs_pilot": (
                        item.get("payload") or {}
                    ).get("infrastructure_release_needs_pilot"),
                },
            }
            for item in report.get("research_default_off") or []
        ],
        "gate_identity_table": [
            {
                "gate_id": row["gate_id"],
                "category": row["category"],
                "status": row["status"],
                "owner": row.get("owner"),
                "module": row.get("module"),
                "identities": row.get("identities") or {},
                "content_address": row.get("content_address"),
            }
            for row in report.get("gate_identity_table") or []
        ],
        "empirical_performance_separate": report.get(
            "empirical_performance_separate"
        ),
        "no_optional_skip_is_success": report.get("no_optional_skip_is_success"),
        "no_optional_skip_is_refutation": report.get(
            "no_optional_skip_is_refutation"
        ),
        "infrastructure_release_needs_research_pilot": report.get(
            "infrastructure_release_needs_research_pilot"
        ),
        "gates": [
            {
                "gate_id": row["gate_id"],
                "category": row["category"],
                "check_mode": row.get("check_mode"),
                "owner": row.get("owner"),
                "module": row.get("module"),
                "status": row["status"],
                "delegated_owner": row.get("delegated_owner"),
                "markers_verified": row.get("markers_verified"),
                "research_status": row.get("research_status"),
            }
            for row in report.get("gates") or []
        ],
    }


def assert_suite_passed(report: Mapping[str, Any]) -> None:
    if report.get("passed"):
        return
    raise AssertionError(
        "INTEG-10 release matrix failed: "
        f"required_failed={report.get('required_failed')!r}"
    )


__all__ = [
    "CAS_SCHEME",
    "DEFAULT_RESULTS_RELPATH",
    "MATRIX_RELPATH",
    "MATRIX_SCHEMA",
    "REPORT_SCHEMA",
    "assert_suite_passed",
    "content_address",
    "load_matrix",
    "matrix_content_digest",
    "run_matrix",
    "stable_report",
    "write_report",
]
