#!/usr/bin/env python3
"""Run and publish the SLM-199 exact-rate and production-adapter fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.harnesses.experiments.slm199_legal_edit_flow import run_fixture
from slm_training.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
DESIGN_JSON = ROOT / "docs/design/iter-slm199-legal-edit-flow-20260723.json"
DESIGN_MD = ROOT / "docs/design/iter-slm199-legal-edit-flow-20260723.md"
AGENTV_DIR = ROOT / "docs/design/iter-slm199-legal-edit-flow-agentv-20260723"


def _portable(value: Any) -> Any:
    prefix = str(AGENTV_DIR.resolve())
    if isinstance(value, str) and value.startswith(prefix):
        return "agentv-dir://" + value[len(prefix) :].lstrip("/")
    if isinstance(value, list):
        return [_portable(item) for item in value]
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()}
    return value


def _rewrite_agentv_paths() -> None:
    prefixes = {
        str(AGENTV_DIR.resolve()): "agentv-dir://",
        quote(str(AGENTV_DIR.resolve()), safe=""): quote("agentv-dir://", safe=""),
    }
    for path in (AGENTV_DIR / "agentv").rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md"}:
            continue
        content = path.read_text(encoding="utf-8")
        for prefix, replacement in prefixes.items():
            content = content.replace(prefix, replacement)
        path.write_text(content, encoding="utf-8")


def _cases(report: dict[str, Any]) -> list[dict[str, Any]]:
    exact = report["exact_oracle"]
    production = report["production_adapter"]
    samples = production["samples"]
    return [
        {
            "id": "exact-generator-contract",
            "criteria": "The learned exact rate table reconstructs the closed generator with finite nonnegative legal rates and zero illegal rate.",
            "pass": (
                exact["closed"]
                and not exact["generator_errors"]
                and exact["illegal_edge_rate_sum"] == 0.0
                and exact["rate_fit"]["max_abs_error"] < 1e-3
            ),
            "result": exact["rate_fit"],
        },
        {
            "id": "exact-endpoint-and-event-count",
            "criteria": "The learned-rate generator and its matched-horizon Gillespie sampler reproduce exact endpoint and derived event-count distributions.",
            "pass": (
                exact["analytic_endpoint_tv"] < 0.01
                and exact["empirical_endpoint_tv"] < 0.05
                and exact["event_count_tv"] < 0.05
                and exact["exact_terminal_mass"] > 0.999
            ),
            "result": {
                "analytic_endpoint_tv": exact["analytic_endpoint_tv"],
                "empirical_endpoint_tv": exact["empirical_endpoint_tv"],
                "event_count_tv": exact["event_count_tv"],
                "exact_event_count_distribution": exact[
                    "exact_event_count_distribution"
                ],
                "samples": exact["samples"],
            },
        },
        {
            "id": "unknown-remains-live",
            "criteria": "Bridge UNKNOWN candidates remain live and receive no direct zero-rate edge or hazard label; indirect set-ranking pressure is disclosed.",
            "pass": (
                not production["unknown_supervised_as_negative"]
                and all(item["unknown_candidate_events"] > 0 for item in samples)
            ),
            "result": {
                "fidelity": production["fidelity"],
                "unknown_rate_mass_after_fit": production[
                    "unknown_rate_mass_after_fit"
                ],
                "unknown_events": [
                    item["unknown_candidate_events"] for item in samples
                ],
            },
        },
        {
            "id": "bounded-production-safety",
            "criteria": "Every production sample refreshes exact candidates, commits only live edits, respects K=2, and returns parser-verified output.",
            "pass": all(
                item["verified_output"]
                and item["all_selected_live"]
                and item["candidate_sets_refreshed"]
                and item["edits"] <= 2
                for item in samples
            ),
            "result": {"samples": samples},
        },
        {
            "id": "honest-default-off-disposition",
            "criteria": "Flow remains default-off, writes no checkpoint, and defers confirmation to VFA1-02.",
            "pass": (
                not report["default_path"]["flow_enabled_by_default"]
                and not report["checkpoint"]["written"]
                and report["confirmation"]["status"] == "blocked"
            ),
            "result": {
                "verdict": report["honest_verdict"],
                "confirmation": report["confirmation"],
            },
        },
    ]


def _markdown(report: dict[str, Any]) -> str:
    exact = report["exact_oracle"]
    production = report["production_adapter"]
    exact_rate = sum(
        int(item["target_exact"]) for item in production["samples"]
    ) / max(1, len(production["samples"]))
    return "\n".join(
        [
            "# SLM-199 (VFA1-01): discrete legal-edit rate matching",
            "",
            "**Status:** measured fixture wiring; no ship or flow-win claim.",
            f"**Verdict:** `{report['honest_verdict']}`.",
            "",
            "## Exact closed CTMC oracle",
            "",
            f"- Closed choice graph: `{exact['closed']}` ({exact['states']} states, {exact['transitions']} transitions).",
            f"- Illegal edge-rate sum: `{exact['illegal_edge_rate_sum']:.6f}`.",
            f"- Exact-rate fit MSE/max error: `{exact['rate_fit']['final_mse']:.8f}` / `{exact['rate_fit']['max_abs_error']:.8f}`.",
            f"- Exact/predicted/empirical terminal mass: `{exact['exact_terminal_mass']:.6f}` / `{exact['predicted_terminal_mass']:.6f}` / `{exact['empirical_terminal_mass']:.6f}`.",
            f"- Analytic/empirical endpoint TV at matched horizon `{exact['horizon']}`: `{exact['analytic_endpoint_tv']:.6f}` / `{exact['empirical_endpoint_tv']:.6f}`.",
            f"- Exact/empirical event-count distributions: `{exact['exact_event_count_distribution']}` / `{exact['empirical_event_count_distribution']}` (TV `{exact['event_count_tv']:.6f}`).",
            "",
            "This acceptance oracle deliberately uses the closed acyclic choice fixture.",
            "The earlier SLM-190 toy/canonical graphs were bounded/inconclusive and",
            "are not re-labelled as passing evidence.",
            "",
            "## OpenUI production adapter",
            "",
            f"- Fidelity: `{production['fidelity']}`.",
            f"- Rows/train rows: `{production['rows']}` / `{production['train_rows']}`.",
            f"- UNKNOWN supervised as negative: `{production['unknown_supervised_as_negative']}`.",
            f"- UNKNOWN rate mass after fit: `{production['unknown_rate_mass_after_fit']:.6f}`.",
            f"- Parser-verified outputs: `{sum(int(item['verified_output']) for item in production['samples'])}/{len(production['samples'])}`.",
            f"- Fixture target-exact rate: `{exact_rate:.3f}` (descriptive only).",
            "",
            "The adapter re-enumerates exact live candidates after every one-edit",
            "commit, keeps UNKNOWN candidates live, uses positive rates and a frozen",
            "fixed-K termination policy, and returns verified syntax or explicit",
            "UNKNOWN/abstention. Bridge holding times are unobserved, so unit-hazard",
            "targets are labelled adapted path approximations, not faithful DFM.",
            "UNKNOWN receives no direct edge or hazard regression label, though it",
            "remains in normalization and therefore receives disclosed indirect",
            "set-ranking pressure.",
            "",
            "## Recipe and disposition",
            "",
            f"- Device/backend: `{report['recipe']['device']}` / `{report['recipe']['backend']}`.",
            f"- Train steps/seeds: `{report['recipe']['train_steps']}` / `{report['recipe']['seeds']}`.",
            f"- Exact samples: `{report['recipe']['exact_samples']}`.",
            f"- Wall: `{report['elapsed_seconds']:.3f}s` (cap `{report['recipe']['max_wall_minutes']}m`).",
            f"- AgentV: `{report['agentv']['summary']}`.",
            "",
            "Flow is default-off, no checkpoint was written, and existing direct",
            "training/decode paths are unchanged. VFA1-02 owns powered confirmation;",
            "this fixture does not establish a held-out flow improvement.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs/slm199"))
    parser.add_argument("--train-steps", type=int, default=8)
    parser.add_argument("--exact-samples", type=int, default=256)
    parser.add_argument("--max-wall-minutes", type=float, default=2.8)
    args = parser.parse_args(argv)
    report = run_fixture(
        train_steps=args.train_steps,
        exact_samples=args.exact_samples,
        max_wall_minutes=args.max_wall_minutes,
    )
    report["version_stamp"] = build_version_stamp(
        "harness.experiments.slm199_legal_edit_flow"
    )
    AGENTV_DIR.mkdir(parents=True, exist_ok=True)
    report["agentv"] = _portable(
        publish_agentv_evaluation(
            AGENTV_DIR,
            name="slm199-legal-edit-flow-fixture",
            claim="exact_rate_and_adapted_production_contract",
            cases=_cases(report),
        )
    )
    _rewrite_agentv_paths()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    DESIGN_JSON.write_text(
        json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    DESIGN_MD.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"report": str(DESIGN_JSON), "verdict": report["honest_verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
