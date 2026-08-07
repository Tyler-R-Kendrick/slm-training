#!/usr/bin/env python3
"""Mine completed continuous deliveries for thrash residual steering.

Cheap (JSON-only) offline fold over ``outputs/autoresearch`` campaign
``sdlc_delivery.json`` files. Does not train or eval.

Usage::

    python -m scripts.mine_continuous_residuals \\
      --root outputs/autoresearch --loop-id continuous-openui-local

    python -m scripts.mine_continuous_residuals --write-ledger
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def _load_module():
    # Prefer package import when PYTHONPATH includes src/.
    try:
        from slm_training.autoresearch.thrash_residuals import (  # noqa: WPS433
            aggregate_slug_stats,
            classify_delivery_residual,
            residual_boosts_from_observations,
        )

        return classify_delivery_residual, aggregate_slug_stats, residual_boosts_from_observations
    except ImportError:
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "src"))
        from slm_training.autoresearch.thrash_residuals import (  # noqa: WPS433
            aggregate_slug_stats,
            classify_delivery_residual,
            residual_boosts_from_observations,
        )

        return classify_delivery_residual, aggregate_slug_stats, residual_boosts_from_observations


def _cycle_index_from_path(path: Path) -> int | None:
    match = re.search(r"-c(\d+)$", path.parent.name)
    if not match:
        match = re.search(r"-c(\d+)$", path.name)
    return int(match.group(1)) if match else None


def _iter_deliveries(root: Path, loop_id: str | None) -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.glob("**/sdlc_delivery.json")):
        parent = path.parent.name
        if not parent.startswith("continuous-loop-"):
            continue
        if loop_id and loop_id not in parent:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        rows.append((path, data))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/autoresearch"),
        help="Campaign root (default: outputs/autoresearch)",
    )
    parser.add_argument("--loop-id", default="continuous-openui-local")
    parser.add_argument(
        "--high-ss-band",
        type=float,
        default=0.35,
        help="Absolute structural_similarity band for high_band_absolute",
    )
    parser.add_argument(
        "--write-ledger",
        action="store_true",
        help="Write loops/<loop-id>/interesting_residuals.jsonl + slug_stats.json",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="How many residual rows / slug stats to print",
    )
    args = parser.parse_args(argv)

    classify, aggregate, boosts_fn = _load_module()
    root = args.root if args.root.is_absolute() else Path.cwd() / args.root
    if not root.is_dir():
        print(f"root missing: {root}", file=sys.stderr)
        return 2

    deliveries: list[dict[str, Any]] = []
    residuals = []
    for path, data in _iter_deliveries(root, args.loop_id):
        camp = path.parent.name
        data = {**data, "campaign_id": data.get("campaign_id") or camp}
        deliveries.append(data)
        obs = classify(
            data,
            campaign_id=camp,
            cycle_index=_cycle_index_from_path(path),
            high_ss_band=float(args.high_ss_band),
        )
        if obs is not None:
            residuals.append(obs)

    residuals.sort(
        key=lambda o: (
            -(o.score or 0.0),
            -(o.cycle_index or 0),
        )
    )
    stats = aggregate(deliveries, residuals=residuals)
    ranked = sorted(
        stats.values(),
        key=lambda s: (-s.prior_score(), -s.n_complete, s.slug),
    )
    boosts = boosts_fn(residuals)

    print(
        f"deliveries={len(deliveries)} residuals={len(residuals)} "
        f"slugs={len(stats)} loop_id={args.loop_id}"
    )
    print("\n# Top residuals")
    for obs in residuals[: max(1, int(args.top))]:
        print(
            f"  c{obs.cycle_index} class={obs.residual_class} "
            f"slug={obs.slug} score={obs.score:.3f} "
            f"ss={obs.ss_control}/{obs.ss_cand} "
            f"pos={obs.positive}"
        )
    print("\n# Slug prior (top)")
    for st in ranked[: max(1, int(args.top))]:
        print(
            f"  {st.slug}: n={st.n_complete} win={st.win_rate:.2f} "
            f"Δss={st.mean_delta_ss:+.4f} binder_fail={st.binder_fail_rate:.2f} "
            f"resid={st.residual_hits} prior={st.prior_score():+.4f}"
        )
    if boosts:
        print("\n# Residual boosts")
        for slug, boost in sorted(boosts.items(), key=lambda kv: -kv[1])[
            : max(1, int(args.top))
        ]:
            print(f"  {slug}: {boost:.3f}")

    if args.write_ledger:
        loop_dir = root / "loops" / args.loop_id
        loop_dir.mkdir(parents=True, exist_ok=True)
        residual_path = loop_dir / "interesting_residuals.jsonl"
        stats_path = loop_dir / "slug_stats.json"
        with residual_path.open("w", encoding="utf-8") as fh:
            for obs in residuals:
                fh.write(json.dumps(obs.as_dict(), sort_keys=True) + "\n")
        stats_path.write_text(
            json.dumps(
                {
                    "schema": "thrash_slug_stats/v1",
                    "loop_id": args.loop_id,
                    "n_deliveries": len(deliveries),
                    "n_residuals": len(residuals),
                    "slugs": {s.slug: s.as_dict() for s in ranked},
                    "residual_boosts": boosts,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {residual_path}")
        print(f"wrote {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
