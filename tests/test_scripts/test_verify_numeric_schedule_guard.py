"""Tests for the RSC-A06 (SLM-242) numeric schedule static guard."""

from __future__ import annotations

from pathlib import Path

from scripts.verify_numeric_schedule_guard import DEFAULT_SCAN_PATHS, ROOT, build_report


def _write(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def test_flags_min_of_two_len_calls(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "truncate.py",
        "def f(a, b):\n    usable = min(len(a), len(b))\n    return usable\n",
    )
    report = build_report(scan_paths=[str(path)])
    patterns = {h.pattern for h in report.hits}
    assert "TRUNCATE" in patterns
    assert report.unsuppressed  # unsuppressed by default -- no marker comment present


def test_does_not_flag_min_of_len_and_scalar(tmp_path: Path) -> None:
    """One side not len(...) is a normal clamp, not a truncation-selection pattern."""
    path = _write(
        tmp_path,
        "clamp.py",
        "def f(a, cap):\n    return min(len(a), cap)\n",
    )
    report = build_report(scan_paths=[str(path)])
    assert not report.hits


def test_flags_unguarded_sum_then_positive_check(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "unguarded.py",
        (
            "def f(weights):\n"
            "    total_w = sum(weights)\n"
            "    if total_w > 0.0:\n"
            "        return total_w\n"
            "    return None\n"
        ),
    )
    report = build_report(scan_paths=[str(path)])
    patterns = {h.pattern for h in report.hits}
    assert "UNGUARDED_SUM" in patterns


def test_does_not_flag_sum_check_against_other_value(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "unrelated.py",
        (
            "def f(weights, threshold):\n"
            "    total_w = sum(weights)\n"
            "    if threshold > 0.0:\n"
            "        return total_w\n"
            "    return None\n"
        ),
    )
    report = build_report(scan_paths=[str(path)])
    assert not report.hits


def test_flags_unused_loop_weight_variable(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "unused_weight.py",
        (
            "def f(weights, xs):\n"
            "    out = []\n"
            "    for d, w in enumerate(weights):\n"
            "        out.append(xs[d])\n"
            "    return out\n"
        ),
    )
    report = build_report(scan_paths=[str(path)])
    patterns = {h.pattern for h in report.hits}
    assert "UNUSED_LOOP_WEIGHT" in patterns


def test_does_not_flag_used_loop_weight_variable(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "used_weight.py",
        (
            "def f(weights, xs):\n"
            "    total = 0.0\n"
            "    for d, w in enumerate(weights):\n"
            "        total += w * xs[d]\n"
            "    return total\n"
        ),
    )
    report = build_report(scan_paths=[str(path)])
    assert not report.hits


def test_does_not_flag_non_weight_loop_variable(tmp_path: Path) -> None:
    """Only weight-shaped names (weight/^w$/_w$) are considered."""
    path = _write(
        tmp_path,
        "generic_loop.py",
        "def f(xs):\n    for i, value in enumerate(xs):\n        pass\n    return xs\n",
    )
    report = build_report(scan_paths=[str(path)])
    assert not report.hits


def test_same_line_suppression_is_honored(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "suppressed_same_line.py",
        (
            "def f(a, b):\n"
            "    usable = min(len(a), len(b))  "
            "# schedule-guard: allow TRUNCATE reason=demo test=tests/x.py::test_y\n"
            "    return usable\n"
        ),
    )
    report = build_report(scan_paths=[str(path)])
    assert report.hits
    assert all(h.suppressed for h in report.hits)
    assert report.unsuppressed == []


def test_line_above_suppression_is_honored(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "suppressed_above.py",
        (
            "def f(a, b):\n"
            "    # schedule-guard: allow TRUNCATE reason=demo test=tests/x.py::test_y\n"
            "    usable = min(len(a), len(b))\n"
            "    return usable\n"
        ),
    )
    report = build_report(scan_paths=[str(path)])
    assert report.hits
    assert all(h.suppressed for h in report.hits)


def test_unsuppressed_hit_fails_the_report() -> None:
    """Sanity: a synthetic unsuppressed hit makes ``report.unsuppressed`` non-empty
    (the property ``main()`` uses to decide the exit code)."""
    report = build_report(scan_paths=[])
    assert report.unsuppressed == []


def test_repo_scan_has_no_unsuppressed_hits() -> None:
    """Run against the real repo scope: every hit found in the canonical
    model-build path must be a documented suppression (RSC-A06/SLM-242's own
    fix, or a tracked known defect), never a silent false positive."""
    report = build_report(scan_paths=DEFAULT_SCAN_PATHS)
    assert report.files_scanned > 0
    unsuppressed = [
        f"{h.pattern} {h.path}:{h.line} -- {h.detail}" for h in report.unsuppressed
    ]
    assert unsuppressed == [], "unsuppressed schedule-guard hits: " + "; ".join(
        unsuppressed
    )


def test_repo_scan_finds_the_three_documented_slm138_suppressions() -> None:
    """The known RSC-A06 suppressions in twotower.py's deep-supervision block
    must still be present and correctly attributed -- regresses if someone
    edits that block without updating the markers."""
    report = build_report(scan_paths=DEFAULT_SCAN_PATHS)
    twotower_hits = {
        h.pattern: h
        for h in report.hits
        if h.path == "src/slm_training/models/twotower.py"
    }
    assert set(twotower_hits) == {"TRUNCATE", "UNGUARDED_SUM", "UNUSED_LOOP_WEIGHT"}
    for hit in twotower_hits.values():
        assert hit.suppressed
        assert hit.suppression_test is not None
        assert (ROOT / hit.suppression_test.split("::")[0]).is_file()
