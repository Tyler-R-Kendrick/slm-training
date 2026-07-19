"""CAP4-05 wiring fixture: quotient-state diffusion graph diagnostics.

Builds small exact graphs, runs connectivity / mixing / information-schedule
diagnostics, compares matched kernels, and writes a version-stamped result
bundle. This is evidence-only wiring: no model train or ship claim.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.dsl.analysis.arity.diffusion_graph import (
    QuotientDiffusionGraph,
    Transition,
    build_ast_subtree_kernel,
    build_posterior_weighted_kernel,
    build_production_mask_kernel,
    build_quotient_random_walk_kernel,
    build_surface_token_kernel,
    build_typed_hole_kernel,
    compare_kernels_at_matched_loss,
    information_schedule,
    recommend_information_balanced_schedule,
)
from slm_training.versioning import build_version_stamp


def _make_json_safe(value: Any) -> Any:
    """Replace non-finite floats with None so the JSON is RFC-compliant."""
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    if isinstance(value, dict):
        return {k: _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(v) for v in value]
    return value


def _directed_ring(n: int) -> QuotientDiffusionGraph:
    graph = QuotientDiffusionGraph()
    states = [f"q{i}" for i in range(n)]
    for i in range(n):
        graph.add_transition(
            Transition(states[i], "step", states[(i + 1) % n], edge_type="kernel")
        )
    return graph


def _undirected_barbell() -> QuotientDiffusionGraph:
    """Two 4-cliques connected by a single bidirectional bridge edge."""
    graph = QuotientDiffusionGraph()
    left = [f"l{i}" for i in range(4)]
    right = [f"r{i}" for i in range(4)]
    for clique in (left, right):
        for i in range(len(clique)):
            for j in range(len(clique)):
                if i != j:
                    graph.add_transition(
                        Transition(clique[i], "clique", clique[j], edge_type="kernel")
                    )
    graph.add_transition(Transition(left[0], "bridge", right[0], edge_type="kernel"))
    graph.add_transition(Transition(right[0], "bridge", left[0], edge_type="kernel"))
    return graph


def _path_with_self_loops(n: int) -> QuotientDiffusionGraph:
    graph = QuotientDiffusionGraph()
    states = [f"p{i}" for i in range(n)]
    for i in range(n):
        graph.add_transition(Transition(states[i], "stay", states[i], edge_type="kernel"))
        if i + 1 < n:
            graph.add_transition(
                Transition(states[i], "forward", states[i + 1], edge_type="kernel")
            )
    return graph


def _build_synthetic_traces(seed: int = 0, n: int = 20) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    for t in range(n):
        records.append(
            {
                "run_id": "fixture",
                "example_id": "e0",
                "seed": seed,
                "decision_index": t + 1,
                "state_fingerprint": f"s{t % 5}",
                "selected_action_id": f"a{t % 3}",
                "diffusion_timestep": t,
                "posterior_entropy_bits": max(0.1, 4.0 * (1.0 - t / n) + rng.gauss(0, 0.1)),
                "completion_support_size_exact": max(1, int(16 * (1.0 - t / n))),
            }
        )
    return records


def _kernel_comparison_fixture() -> dict[str, Any]:
    states = [f"k{i}" for i in range(6)]
    options = {
        "k0": ["k1", "k2"],
        "k1": ["k0", "k2", "k3"],
        "k2": ["k0", "k1", "k3"],
        "k3": ["k1", "k2", "k4", "k5"],
        "k4": ["k3", "k5"],
        "k5": ["k3", "k4"],
    }
    kernels = [
        build_surface_token_kernel(states, options),
        build_production_mask_kernel(states, options),
        build_ast_subtree_kernel(states, options),
        build_typed_hole_kernel(states, options),
    ]
    comparison = compare_kernels_at_matched_loss(kernels, states, target_loss_bits=2.0)
    return comparison.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/cap4-05-quotient-diffusion"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir: Path = args.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    ring = _directed_ring(6)
    barbell = _undirected_barbell()
    path = _path_with_self_loops(5)

    traces = _build_synthetic_traces(seed=args.seed)
    trace_graph = QuotientDiffusionGraph.from_traces(traces)
    schedule = information_schedule(traces)
    balanced = recommend_information_balanced_schedule(schedule, n_steps=8)

    graph_reports = {
        "directed_ring_n6": ring.diagnostics(),
        "undirected_barbell_4_4": barbell.diagnostics(),
        "path_with_self_loops_n5": path.diagnostics(),
        "trace_graph": trace_graph.diagnostics(),
    }

    # Add a quotient-random-walk kernel for the ring and barbell.
    ring_kernel = build_quotient_random_walk_kernel(ring)
    barbell_kernel = build_quotient_random_walk_kernel(barbell)
    target_states = ["k3", "k4"]
    posterior_kernel = build_posterior_weighted_kernel(
        [f"k{i}" for i in range(6)], target_states, temperature=0.5
    )

    result: dict[str, Any] = {
        "version_stamp": build_version_stamp("analysis.arity.diffusion_graph"),
        "run_id": run_id,
        "schema": "cap4-05/quotient_diffusion_fixture/v1",
        "graphs": graph_reports,
        "kernels": {
            "ring_random_walk": ring_kernel.to_dict(),
            "barbell_random_walk": barbell_kernel.to_dict(),
            "posterior_weighted": posterior_kernel.to_dict(),
            "matched_comparison": _kernel_comparison_fixture(),
        },
        "information_schedule": {
            "points": [p.to_dict() for p in schedule],
            "recommended_balanced_schedule": balanced,
            "record_count": len(traces),
        },
        "honest_caveats": [
            "Exact metrics are feasible only for the small fixtures above; real quotient graphs are sampled/estimated.",
            "No diffusion model train was run; kernels are defined on abstract state spaces.",
            "Denoising comparisons require a future train/eval run with matched conditional information loss.",
            "Spectral gap for the directed ring is zero because the chain is periodic.",
        ],
    }

    json_path = out_dir / "quotient_diffusion_fixture.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(_make_json_safe(result), fh, indent=2, sort_keys=True)

    md_path = out_dir / "README.md"
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# CAP4-05 quotient diffusion fixture ({run_id})\n\n")
        fh.write("Wiring-only diagnostic run. See `quotient_diffusion_fixture.json` for full metrics.\n\n")
        fh.write("## Highlights\n\n")
        for name, report in graph_reports.items():
            fh.write(f"- **{name}**: {report['vertex_count']} vertices, "
                     f"{report['edge_count']} edges, "
                     f"diameter={report['diameter']['value']}, "
                     f"spectral_gap={report['spectral_gap']['value']:.4g}, "
                     f"conductance={report['conductance']['value']:.4g}\n")
        fh.write("\n## Honest caveats\n\n")
        for note in result["honest_caveats"]:
            fh.write(f"- {note}\n")
        fh.write(f"\nArtifact: `{json_path}`\n")

    print(f"Wrote CAP4-05 fixture to {out_dir}")


if __name__ == "__main__":
    main()
