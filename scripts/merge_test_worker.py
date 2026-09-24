"""Untrusted pytest workload protocol for the local merge verifier.

The controller copies this file into each private invocation directory. This
output is workload evidence, never an independently issued release receipt.
"""

from __future__ import annotations

import json
import faulthandler
import os
import sys
from pathlib import Path


class TestEvidence:
    def __init__(self) -> None:
        self.nodes: list[str] = []
        self.markers: dict[str, list[str]] = {}
        self.reports: list[dict] = []
        self.deselected: list[str] = []
        self.deselected_markers: dict[str, list[str]] = {}
        self.collection_errors: list[str] = []

    def pytest_sessionstart(self, session) -> None:
        # CLI options have been parsed. The protocol, not thousands of printed
        # node IDs, owns collection evidence; execution keeps normal diagnostics.
        if session.config.option.collectonly:
            reporter = session.config.pluginmanager.get_plugin("terminalreporter")
            if reporter is not None:
                session.config.pluginmanager.unregister(reporter)

    def pytest_collection_finish(self, session) -> None:
        self.nodes = [item.nodeid for item in session.items]
        self.markers = {
            item.nodeid: sorted({marker.name for marker in item.iter_markers()})
            for item in session.items
        }

    def pytest_deselected(self, items) -> None:
        for item in items:
            self.deselected.append(item.nodeid)
            self.deselected_markers[item.nodeid] = sorted(
                {marker.name for marker in item.iter_markers()}
            )

    def pytest_collectreport(self, report) -> None:
        if report.failed:
            self.collection_errors.append(report.nodeid)
            print(str(report.longrepr)[-4000:], file=sys.stderr)

    def pytest_runtest_logreport(self, report) -> None:
        self.reports.append(
            {
                "nodeid": report.nodeid,
                "when": report.when,
                "outcome": report.outcome,
                "duration_seconds": report.duration,
                "wasxfail": bool(getattr(report, "wasxfail", False)),
            }
        )


def main() -> int:
    request_path, output_path = map(Path, sys.argv[1:])
    request = json.loads(request_path.read_text())
    root = Path(request["root"]).resolve()
    os.chdir(root)
    sys.path[:0] = [str(root), str(root / "src")]
    os.environ.pop("PYTEST_ADDOPTS", None)
    import pytest

    evidence = TestEvidence()
    argv = [
        "-o",
        "addopts=",
        "-m",
        "not training and not slow",
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    if request["collect_only"]:
        argv.append("--collect-only")
    if request["collect_only"]:
        faulthandler.dump_traceback_later(30, file=sys.stderr)
    try:
        result = int(pytest.main([*argv, *request["targets"]], plugins=[evidence]))
    finally:
        if request["collect_only"]:
            faulthandler.cancel_dump_traceback_later()
    output = {
        "schema": "merge_test_workload/v1",
        "request_digest": request["request_digest"],
        "exit_code": result,
        "pytest_version": pytest.__version__,
        "nodes": evidence.nodes,
        "markers": evidence.markers,
        "deselected": evidence.deselected,
        "deselected_markers": evidence.deselected_markers,
        "collection_errors": evidence.collection_errors,
        "reports": evidence.reports,
    }
    # This is an untrusted attempt file, read only after child exit. A partial
    # write is rejected; it is not a published verification receipt.
    output_path.write_text(json.dumps(output, sort_keys=True))
    if request["collect_only"]:
        print(json.dumps({"collected": len(evidence.nodes), "exit_code": result}))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
