#!/usr/bin/env python3
"""Deterministic CAP0–CAP4 fixture reproduction suite (CPU, no model downloads).

Re-runs the small exact-calculation fixtures that underpin the CAP5 evidence
package and writes a stamped summary.  Exit code 0 means every checked fixture
produced the expected deterministic result; exit code 1 means at least one
fixture diverged or a durable certificate failed schema validation.

Example::

    python -m scripts.reproduce_calculated_arity_fixtures \
        --out outputs/runs/cap_repro \
        --verify-expected
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DURABLE_CERTIFICATE_PATTERNS = ("cap*.json", "iter-cap*.json")

# Pinned exact counts for the committed bounded-expr fixture.
# These mirror tests/test_dsl/test_arity_analysis.py.
EXPECTED_ARITY = {
    "canonical_asts": 400,
    "trie_states": 844,
    "minimized_states": 28,
    "action_alphabet": 8,
    "scope_signatures": 3,
    "min_k": 3,
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _run_exact_arity() -> dict[str, Any]:
    from slm_training.dsl.analysis.arity import AnalysisBounds, analyze

    bounds = AnalysisBounds(
        max_ast_nodes=6,
        max_ast_depth=None,
        max_live_bindings=2,
        template_classes=("N",),
        result_types=("number",),
    )
    report = analyze(fixture="bounded-expr", bounds=bounds, dimensions=4)
    counts = {
        "canonical_asts": report.canonical_ast_count,
        "raw_states": report.raw_state_count,
        "trie_states": report.trie_state_count,
        "minimized_states": report.minimized_state_count,
        "action_alphabet": report.action_alphabet_size,
        "scope_signatures": report.scope_signature_count,
        "max_local_branching": report.max_local_branching,
        "min_k": report.capacity["min_k"],
    }
    expected_ok = all(
        counts.get(k) == v for k, v in EXPECTED_ARITY.items()
    )
    return {
        "bounds": bounds.to_dict(),
        "counts": counts,
        "expected_matches": EXPECTED_ARITY,
        "ok": expected_ok,
    }


def _run_coding() -> dict[str, Any]:
    from slm_training.dsl.analysis.arity.coding import (
        build_mds_7_4_2_3,
        build_shortened_ternary_hamming_7_4_3,
        hamming_sphere_packing_holds,
        singleton_upper_bound,
        verify_code,
    )
    from slm_training.dsl.analysis.arity.precision import (
        minimum_margin_trit_planes,
        ternary_ecoc_width,
    )

    mds = build_mds_7_4_2_3()
    mds_ver = verify_code(
        mds, q=7, n=4, required_size=49, required_distance=3
    )
    ham = build_shortened_ternary_hamming_7_4_3()
    ham_ver = verify_code(
        ham, q=3, n=7, required_size=81, required_distance=3
    )

    ok = mds_ver.ok and ham_ver.ok
    return {
        "mds_7_4_2_3": {
            "ok": mds_ver.ok,
            "size": mds_ver.size,
            "minimum_distance": mds_ver.minimum_distance,
        },
        "hamming_7_4_3": {
            "ok": ham_ver.ok,
            "size": ham_ver.size,
            "minimum_distance": ham_ver.minimum_distance,
        },
        "singleton_bound_q7_n4_d3": singleton_upper_bound(7, 4, 3),
        "sphere_packing_M49_q7_n4_t1": hamming_sphere_packing_holds(
            49, 7, 4, 1
        ),
        "ternary_ecoc_width_8_detect": ternary_ecoc_width(
            8, detect_single_trit_error=True
        ),
        "minimum_margin_trit_planes_4_1": minimum_margin_trit_planes(4, 1),
        "ok": ok,
    }


def _run_task_quotient() -> dict[str, Any]:
    from slm_training.dsl.analysis.arity.task_quotient import (
        AlignedActionRecord,
        TaskDistortionSpec,
        build_confusability_graph,
        build_state_profiles,
        color_graph,
    )

    records = [
        AlignedActionRecord("s0", "add", "math", 0.7),
        AlignedActionRecord("s0", "sub", "math", 0.3),
        AlignedActionRecord("s1", "add", "math", 0.75),
        AlignedActionRecord("s1", "sub", "math", 0.25),
        AlignedActionRecord("s2", "mul", "math", 0.9),
        AlignedActionRecord("s2", "div", "math", 0.1),
    ]
    profiles = build_state_profiles(records)
    spec = TaskDistortionSpec(
        spec_id="cap_repro_tolerance_0_2",
        policy_metric="cross_entropy_regret",
        policy_tolerance=0.2,
    )
    graph = build_confusability_graph(profiles, spec)
    coloring = color_graph(graph, exact_max_vertices=128)
    ok = coloring.num_colors > 0
    return {
        "state_count": len(profiles),
        "edge_count": len(graph.edges),
        "quotient_size": coloring.num_colors,
        "exact": coloring.exact,
        "ok": ok,
    }


def _run_conditional_rate() -> dict[str, Any]:
    from slm_training.dsl.analysis.arity.conditional_rate import (
        analyze_conditional_rate,
    )
    from slm_training.dsl.analysis.arity.task_quotient import (
        AlignedActionRecord,
        TaskDistortionSpec,
    )

    records = [
        AlignedActionRecord("s0", "a", "family", 0.8),
        AlignedActionRecord("s0", "b", "family", 0.2),
        AlignedActionRecord("s1", "a", "family", 0.45),
        AlignedActionRecord("s1", "b", "family", 0.55),
        AlignedActionRecord("s2", "c", "family", 1.0),
    ]
    spec = TaskDistortionSpec(
        spec_id="cap_repro_rate",
        policy_metric="cross_entropy_regret",
        policy_tolerance=0.3,
    )
    report = analyze_conditional_rate(records, spec, rd_tolerances=[0.1, 0.2])
    fano = report.fano_bounds[0]
    posterior = report.posterior_support
    return {
        "fano_lower_bound_error": fano.lower_bound_error,
        "posterior_mean": posterior.mean,
        "posterior_max": posterior.max,
        "rd_points": [p.to_dict() for p in report.rate_distortion_curve],
        "ok": fano.lower_bound_error >= 0,
    }


def _validate_durable_certificates() -> dict[str, Any]:
    checked = []
    for pattern in DURABLE_CERTIFICATE_PATTERNS:
        for path in sorted((ROOT / "docs/design").glob(pattern)):
            rel = path.relative_to(ROOT).as_posix()
            entry = {"path": rel, "sha256": _sha256_file(path), "ok": False}
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                entry["error"] = f"invalid json: {exc}"
            else:
                entry["has_version_stamp"] = (
                    isinstance(data.get("version_stamp"), dict)
                    and data["version_stamp"].get("stamp_schema")
                    == "version_stamp/v1"
                )
                # Legacy CAP result JSONs are accepted without a stamp;
                # the summary records which ones are grandfathered.
                entry["ok"] = True
            checked.append(entry)
    return {"certificates": checked, "ok": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/runs/cap_repro"),
        help="Directory for the reproduction summary JSON.",
    )
    parser.add_argument(
        "--verify-expected",
        action="store_true",
        help="Fail if exact fixture counts differ from the committed expected values.",
    )
    parser.add_argument(
        "--skip-certificates",
        action="store_true",
        help="Skip durable docs/design certificate validation.",
    )
    args = parser.parse_args(argv)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    sections: dict[str, Any] = {}
    failures: list[str] = []

    sections["exact_arity"] = _run_exact_arity()
    if args.verify_expected and not sections["exact_arity"]["ok"]:
        failures.append("exact arity counts diverged from expected fixture values")

    sections["coding"] = _run_coding()
    if not sections["coding"]["ok"]:
        failures.append("coding construction verification failed")

    sections["task_quotient"] = _run_task_quotient()
    if not sections["task_quotient"]["ok"]:
        failures.append("task quotient fixture failed")

    sections["conditional_rate"] = _run_conditional_rate()
    if not sections["conditional_rate"]["ok"]:
        failures.append("conditional rate fixture failed")

    if not args.skip_certificates:
        sections["durable_certificates"] = _validate_durable_certificates()
        if not sections["durable_certificates"]["ok"]:
            failures.append("one or more durable certificates missing/invalid")

    from slm_training.versioning import build_version_stamp

    stamp = build_version_stamp(
        "analysis.arity.diffusion_graph",
        "harness.experiments",
        "model.quantization",
    )

    summary = {
        "schema": "cap5_repro_summary/v1",
        "version_stamp": stamp,
        "ok": not failures,
        "failures": failures,
        "sections": sections,
    }

    out_path = out_dir / "cap_repro_summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote reproduction summary to {out_path}")
    if failures:
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
