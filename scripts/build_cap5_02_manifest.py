#!/usr/bin/env python3
"""Build the preregistered CAP5-02 campaign manifest.

Example::

    python -m scripts.build_cap5_02_manifest \
        --manifest-version v1 \
        --out outputs/cap5-02/campaign_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.experiments.cap5_02_campaign import (
    CampaignArm,
    build_cap5_campaign_manifest,
    validate_cap5_campaign_manifest,
)


# Core arms from SLM-101 acceptance criteria, with honest omission reasons for
# mechanisms that require later CAP evidence in this repository state.
DEFAULT_ARMS: list[CampaignArm] = [
    CampaignArm(
        arm_id="baseline_full_precision",
        hypothesis_id="H0_baseline",
        mechanism="existing_best_full_precision_baseline",
        eligible=True,
        selection_evidence="CAP0-CAP4 diagnostic baseline",
    ),
    CampaignArm(
        arm_id="compiler_owned_scorer",
        hypothesis_id="H1_compiler_owned",
        mechanism="compiler_owned_local_action_scorer",
        eligible=True,
        selection_evidence="CAP2-04 compiler-ownership results",
    ),
    CampaignArm(
        arm_id="implicit_state_control",
        hypothesis_id="H2_implicit_state",
        mechanism="implicit_state_control",
        eligible=True,
        selection_evidence="CAP2 representative budget",
    ),
    CampaignArm(
        arm_id="best_continuous_latent",
        hypothesis_id="H3_continuous_latent",
        mechanism="best_continuous_soft_latent_control",
        eligible=True,
        selection_evidence="CAP2 bottleneck results",
    ),
    CampaignArm(
        arm_id="best_mixed_radix_latent",
        hypothesis_id="H4_mixed_radix",
        mechanism="best_calculated_mixed_radix_discrete_latent",
        eligible=True,
        selection_evidence="CAP2-CAP3 width/precision ladder",
    ),
    CampaignArm(
        arm_id="ternary_scorer",
        hypothesis_id="H5_ternary",
        mechanism="ternary_local_scorer_backbone",
        eligible=True,
        selection_evidence="CAP3 quantization ladder",
    ),
    CampaignArm(
        arm_id="four_level_equal_storage",
        hypothesis_id="H6_four_level",
        mechanism="learned_four_level_with_zero_equal_storage",
        eligible=True,
        selection_evidence="CAP3 width/precision ladder",
    ),
    CampaignArm(
        arm_id="int4_control",
        hypothesis_id="H7_int4",
        mechanism="INT4_control",
        eligible=True,
        selection_evidence="CAP3 low-bit results",
    ),
    CampaignArm(
        arm_id="mixed_precision_allocation",
        hypothesis_id="H8_mixed_precision",
        mechanism="selected_mixed_precision_allocation",
        eligible=True,
        selection_evidence="CAP3 mixed-precision results",
    ),
    CampaignArm(
        arm_id="wider_low_bit_equal_byte",
        hypothesis_id="H9_wider_low_bit",
        mechanism="wider_low_bit_equal_byte_finalist",
        eligible=True,
        selection_evidence="CAP3-05 equal-byte planner",
    ),
    CampaignArm(
        arm_id="adaptive_plane",
        hypothesis_id="H10_adaptive_plane",
        mechanism="adaptive_plane_candidate",
        eligible=False,
        omission_reason="CAP4-02 measured Pareto cost gain not yet available",
    ),
    CampaignArm(
        arm_id="structured_sparsity",
        hypothesis_id="H11_sparsity",
        mechanism="structured_sparsity_candidate",
        eligible=False,
        omission_reason="CAP4-04 measured Pareto cost gain not yet available",
    ),
    CampaignArm(
        arm_id="exact_lattice_energy",
        hypothesis_id="H12_exact_lattice",
        mechanism="exact_lattice_low_arity_energy_candidate",
        eligible=False,
        omission_reason="CAP4-03 energy gate not yet passed",
    ),
    CampaignArm(
        arm_id="quotient_diffusion",
        hypothesis_id="H13_quotient_diffusion",
        mechanism="quotient_diffusion_candidate",
        eligible=False,
        omission_reason="CAP4-05 graph and denoising gates not yet passed",
    ),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-version",
        default="v1",
        help="Campaign manifest version; bump after any post-hoc change.",
    )
    parser.add_argument(
        "--primary-metric",
        default="binding_aware_meaningful_program_rate",
        help="Primary preregistered comparison metric.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination JSON manifest path.",
    )
    parser.add_argument(
        "--note",
        default="CAP5-02 preregistered quality/cost campaign manifest (wiring slice).",
        help="Free-form manifest note.",
    )
    args = parser.parse_args(argv)

    manifest = build_cap5_campaign_manifest(
        DEFAULT_ARMS,
        manifest_version=args.manifest_version,
        primary_metric=args.primary_metric,
        note=args.note,
    )
    errors = validate_cap5_campaign_manifest(manifest.to_dict())
    if errors:
        for error in errors:
            print(f"manifest validation error: {error}")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {args.out}")
    print(
        f"eligible={sum(1 for a in manifest.arms if a.eligible)} "
        f"omitted={sum(1 for a in manifest.arms if not a.eligible)} "
        f"manifest_hash={manifest.manifest_hash}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
