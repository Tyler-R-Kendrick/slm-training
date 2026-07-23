#!/usr/bin/env python3
"""Run the SLM-138 shared recursive denoiser fixture.

Builds tiny TwoTower models with ``denoiser_arch="stacked"`` and
``"shared_recursive"``, exercises forward / training_loss on synthetic records,
verifies shapes, parameter counts, weight sharing across recursions, and the
checkpoint migration helper.  Emits version-stamped JSON + markdown artifacts.

SLM-239 (RSC-A03): the fixture's RNG usage is now an explicit, disjoint
namespace contract (:mod:`slm_training.models.rng_contract`) instead of
implicit global-RNG consumption order -- see that module's docstring and
``docs/design/iter-rsc-a03-*.md``. This does **not** change SLM-237/238's
training-loss semantics, only the fixture's control of RNG around it.

SLM-241 (RSC-A05): also emits a ``control_arm_table`` -- real, measured
resource accounting (parameters/block-evaluations/estimated FLOPs) for every
built matched-control arm (A/B/C/D/E/F/G; see
:mod:`slm_training.models.recursive_control_arms`), never a raw loss or a
winner. Only H remains explicitly deferred -- see ``docs/design/iter-rsc-a05-*``.
Arm F (unshared depth-matched tower) additionally gets an ``arm_f_dual_view``
field: its block-evaluation-matched construction (the ``control_arm_table``
row) necessarily has MORE parameters than arm B (nothing is shared), so a
second, real, measured parameter-nearest construction is reported alongside
it with its own (necessarily nonzero) block-evaluation residual -- never a
bare "matched" claim on both dimensions at once. Arm E (stacked + matched
state capacity) needs no such dual view: its state/state_ctx_proj tensors
are shape-matched exactly to arm B's z-state delta, so its ``control_arm_table``
row alone reports the exact (never approximate) parameter-delta match.

Example:
  python -m scripts.run_slm138_recursive_denoiser_fixture --mode plan-only
  python -m scripts.run_slm138_recursive_denoiser_fixture --mode fixture
  python -m scripts.run_slm138_recursive_denoiser_fixture --mode fixture --allow-dirty
  python -m scripts.run_slm138_recursive_denoiser_fixture --mode determinism
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.dsl.schema import ExampleRecord
from slm_training.models.recursive_control_arms import (
    BUILT_ARM_IDS,
    DEFERRED_ARM_IDS,
    build_arm_f_dual_view,
    build_control_arm_table,
)
from slm_training.models.recursive_denoiser import (
    ArchitectureComparisonReportV1,
    SharedRecursiveDenoiserTower,
    compare_denoiser_architectures,
)
from slm_training.models.rng_contract import (
    RNG_CONTRACT_VERSION,
    isolated_draw,
    rng_namespace_report,
    seed_training_corruption,
)
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel
from slm_training.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    "hero = Card([hero_title, hero_body])"
)
CTA = 'root = Stack([cta])\ncta = Button(":cta.label")'


def _today_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _fixture_records() -> list[ExampleRecord]:
    return [
        ExampleRecord(id="a", prompt="Hero layout", openui=HERO, split="train"),
        ExampleRecord(id="b", prompt="CTA layout", openui=CTA, split="train"),
    ]


def _stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _records_hash(records: list[ExampleRecord]) -> str:
    return _stable_hash([asdict(r) for r in records])


def _config_hash(config: TwoTowerConfig) -> str:
    return _stable_hash(asdict(config))


def _tokenizer_hash(tokenizer: Any) -> str:
    vocab = getattr(tokenizer, "token_to_id", None)
    if isinstance(vocab, dict):
        return _stable_hash(vocab)
    return _stable_hash({"vocab_size": tokenizer.vocab_size})


def _build_model(arch: str, seed: int = 0) -> TwoTowerModel:
    # SLM-237 (RSC-A01): recursive_depth_supervision_weights only applies to
    # architectures that expose recursive_outputs. Applying it to "stacked"
    # here was historical failure mode #6 (silently ignored pre-fix); the
    # fail-closed validator now correctly rejects that combination, so this
    # fixture only sets the weights for "shared_recursive".
    ds_weights = (0.5, 1.0) if arch == "shared_recursive" else ()
    return TwoTowerModel.from_records(
        _fixture_records(),
        config=TwoTowerConfig(
            d_model=32,
            n_heads=2,
            context_layers=1,
            denoiser_layers=2,
            denoiser_arch=arch,  # type: ignore[arg-type]
            recursive_steps=2,
            recursive_transition_layers=2,
            recursive_depth_supervision_weights=ds_weights,
            recursive_depth_aux_mode=("legacy_all_depths" if ds_weights else None),
            grammar_constrained=False,
            gen_steps=2,
            seed=seed,
        ),
        device="cpu",
    )


def _count_params(model: TwoTowerModel) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def _shape_probe(model: TwoTowerModel, base_seed: int) -> Any:
    """Deterministic forward-shape probe.

    Both draws run inside :func:`isolated_draw` (``torch.random.fork_rng``),
    so the outer global RNG stream that ``training_loss`` will later read for
    ``training_corruption`` is byte-identical whether this probe runs, runs
    twice, or never runs -- SLM-239's "harmless call-order permutation"
    guarantee is structural, not incidental.
    """
    import torch

    noisy = isolated_draw(
        base_seed,
        "shape_probe_inputs",
        lambda: torch.randint(1, model.tokenizer.vocab_size, (2, 6)),
    )
    ctx = isolated_draw(
        base_seed,
        "shape_probe_context",
        lambda: torch.randn(2, 3, model.config.d_model),
    )
    return model.denoiser(noisy, ctx, pad_id=model.tokenizer.pad_id)


def _extra_harmless_probe(base_seed: int) -> None:
    """An extra, otherwise-unused random draw under the ``control_only``
    namespace -- exercised only when ``insert_extra_probe=True`` to prove
    that inserting a harmless random call does not perturb downstream
    training-loss/deep-supervision values."""
    import torch

    isolated_draw(base_seed, "control_only", lambda: torch.randn(4, 4))


def _architecture_comparison(
    stacked: TwoTowerModel, recursive: TwoTowerModel, base_seed: int
) -> ArchitectureComparisonReportV1:
    """Real, measured ``ArchitectureComparisonReportV1`` for this fixture's
    stacked vs shared_recursive denoiser towers (SLM-240/RSC-A04).

    Replaces the retracted single "same parameter count / layer names" claim
    with independently named, independently falsifiable fields -- see
    ``slm_training.models.recursive_denoiser`` module docstring. Reuses the
    same deterministic ``shape_probe_inputs``/``shape_probe_context``
    namespaces as ``_shape_probe`` above: ``isolated_draw``'s ``fork_rng``
    guarantee (SLM-239/RSC-A03) means calling it again here is harmless to
    the outer RNG stream and reproduces a byte-identical input batch.
    """
    import torch

    noisy = isolated_draw(
        base_seed,
        "shape_probe_inputs",
        lambda: torch.randint(1, stacked.tokenizer.vocab_size, (2, 6)),
    )
    ctx = isolated_draw(
        base_seed,
        "shape_probe_context",
        lambda: torch.randn(2, 3, stacked.config.d_model),
    )
    return compare_denoiser_architectures(
        stacked.denoiser,
        recursive.denoiser,
        noisy_ids=noisy,
        context=ctx,
        pad_id=stacked.tokenizer.pad_id,
    )


def _control_arm_table(stacked: TwoTowerModel, base_seed: int) -> list[dict[str, Any]]:
    """SLM-241 (RSC-A05): real, measured resource-accounting table for every
    built control arm (A/B/C/D/E/F/G) -- requirement #11's "complete comparison
    table for every arm you built, without raw-loss winner language". Reuses
    the same deterministic ``shape_probe_inputs``/``shape_probe_context``
    namespaces as ``_architecture_comparison`` above.
    """
    import torch

    noisy = isolated_draw(
        base_seed,
        "shape_probe_inputs",
        lambda: torch.randint(1, stacked.tokenizer.vocab_size, (2, 6)),
    )
    ctx = isolated_draw(
        base_seed,
        "shape_probe_context",
        lambda: torch.randn(2, 3, stacked.config.d_model),
    )
    reports = build_control_arm_table(
        BUILT_ARM_IDS,
        vocab_size=stacked.tokenizer.vocab_size,
        d_model=stacked.config.d_model,
        n_layers=stacked.config.denoiser_layers,
        n_heads=stacked.config.n_heads,
        max_len=stacked.config.max_target_len,
        recursive_steps=2,
        recursive_transition_layers=stacked.config.denoiser_layers,
        noisy_ids=noisy,
        context=ctx,
        pad_id=stacked.tokenizer.pad_id,
    )
    return [report.as_dict() for report in reports]


def _arm_f_dual_view(stacked: TwoTowerModel, base_seed: int) -> dict[str, Any]:
    """SLM-241 (RSC-A05) follow-up: arm F's two honest matching views against
    arm B -- block-evaluation-matched (the primary ``control_arm_table`` "F"
    row) and parameter-nearest (a separate construction, real and measured,
    reported alongside its own residual). Reuses the same deterministic
    ``shape_probe_inputs``/``shape_probe_context`` namespaces as the other
    fixture-level comparisons above.
    """
    import torch

    noisy = isolated_draw(
        base_seed,
        "shape_probe_inputs",
        lambda: torch.randint(1, stacked.tokenizer.vocab_size, (2, 6)),
    )
    ctx = isolated_draw(
        base_seed,
        "shape_probe_context",
        lambda: torch.randn(2, 3, stacked.config.d_model),
    )
    return build_arm_f_dual_view(
        vocab_size=stacked.tokenizer.vocab_size,
        d_model=stacked.config.d_model,
        n_heads=stacked.config.n_heads,
        max_len=stacked.config.max_target_len,
        recursive_steps=2,
        recursive_transition_layers=stacked.config.denoiser_layers,
        noisy_ids=noisy,
        context=ctx,
        pad_id=stacked.tokenizer.pad_id,
    )


def _clean_tree_gate(*, code_dirty: bool | None, allow_dirty: bool) -> dict[str, Any]:
    """Pure clean-tree evidence classification (SLM-239 requirement #3).

    Never raises -- ``_run_fixture`` itself must stay safe to call from
    tests/other scripts regardless of the caller's working-tree state (it is
    a functional/regression fixture, not the evidence-persistence boundary).
    ``code_dirty=None`` (git unavailable/unknowable) is treated like dirty --
    fail closed on the *classification*, never silently claiming clean
    provenance. The hard refusal-to-persist-without---allow-dirty gate lives
    in ``main()``'s ``--mode fixture`` path, which is the actual boundary
    that writes checked-in ``docs/design`` evidence.
    """
    dirty = code_dirty is None or bool(code_dirty)
    comparable = not dirty
    return {"comparable": comparable, "claim_grade": comparable}


def _diff_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    diff = result.stdout
    if not diff:
        return None
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()


def _run_fixture(
    *,
    base_seed: int = 0,
    probe_order: str = "stacked_first",
    insert_extra_probe: bool = False,
    training_corruption_seed: int | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Run the SLM-138 fixture under the SLM-239 RNG contract.

    Execution is split into the six phases required by SLM-239: (1)
    construction, (2) deterministic forward-shape probes, (3) deterministic
    pre-update objective decomposition, (4) one optimizer step, (5)
    deterministic post-update verification (restored corruption-RNG
    checkpoint -- never an implicitly-advanced second call), (6) checkpoint
    round-trip.
    """
    import torch

    if probe_order not in ("stacked_first", "recursive_first"):
        raise ValueError(f"unknown probe_order {probe_order!r}")

    records = _fixture_records()

    # (1) Construction. TwoTowerModel.__init__ reseeds
    # torch.manual_seed(config.seed) itself -- unchanged SLM-237/238 behavior
    # -- so each model's weights are reproducible regardless of build order.
    stacked = _build_model("stacked", seed=base_seed)
    recursive = _build_model("shared_recursive", seed=base_seed)

    # (2) Deterministic forward-shape probes. Order/insertion is a harmless
    # permutation by construction (isolated_draw forks and restores the
    # global stream), which is exactly what this issue requires.
    def _probe_stacked() -> Any:
        return _shape_probe(stacked, base_seed)

    def _probe_recursive() -> Any:
        return _shape_probe(recursive, base_seed)

    if insert_extra_probe:
        _extra_harmless_probe(base_seed)
    if probe_order == "stacked_first":
        stacked_forward = _probe_stacked()
        recursive_forward = _probe_recursive()
    else:
        recursive_forward = _probe_recursive()
        stacked_forward = _probe_stacked()
    if insert_extra_probe:
        _extra_harmless_probe(base_seed)

    # SLM-240 (RSC-A04): real, measured architecture comparison report --
    # deterministic, RNG-isolated (see _architecture_comparison docstring),
    # so its placement here (between the shape probes and the pre-update
    # objective decomposition) does not perturb SLM-239's guarantees.
    architecture_comparison = _architecture_comparison(stacked, recursive, base_seed)

    # SLM-241 (RSC-A05): the built control-arm resource-accounting table
    # (A/B/C/D/F/G). Deterministic/RNG-isolated for the same reason as
    # architecture_comparison above.
    control_arm_table = _control_arm_table(stacked, base_seed)

    # SLM-241 (RSC-A05) follow-up: arm F's paired block-evaluation-matched /
    # parameter-nearest views -- the control_arm_table row above is the
    # block-evaluation-matched construction only; this adds the honest
    # parameter-nearest alternative + its residual.
    arm_f_dual_view = _arm_f_dual_view(stacked, base_seed)

    # (3) Deterministic pre-update objective decomposition. Explicit
    # named-namespace seed immediately before each training_loss call --
    # architecture-common (both models share the same synthetic batch), so
    # the same derived/override seed is used for both, per SLM-239's "use the
    # same intended namespace for architecture-common tensors" requirement.
    stacked_checkpoint = seed_training_corruption(
        base_seed, stacked, override_seed=training_corruption_seed
    )
    stacked_loss = stacked.training_loss(records)

    recursive_checkpoint = seed_training_corruption(
        base_seed, recursive, override_seed=training_corruption_seed
    )
    recursive_loss = recursive.training_loss(records)
    recursive_pre_metrics = dict(recursive.last_training_metrics)

    # (4) One optimizer step (each).
    stacked.train()
    recursive.train()
    opt_s = torch.optim.AdamW(stacked.trainable_parameters(), lr=1e-3)
    opt_r = torch.optim.AdamW(recursive.trainable_parameters(), lr=1e-3)
    opt_s.zero_grad(set_to_none=True)
    opt_r.zero_grad(set_to_none=True)
    stacked_loss.backward()
    recursive_loss.backward()
    opt_s.step()
    opt_r.step()

    # (5) Deterministic post-update verification: restore the *same*
    # corruption-RNG checkpoint captured before the pre-update call rather
    # than letting this second training_loss call implicitly consume the
    # next draws in the stream (SLM-239 requirement #2).
    stacked_checkpoint.restore(stacked)
    stacked_post_loss = stacked.training_loss(records)
    recursive_checkpoint.restore(recursive)
    recursive_post_loss = recursive.training_loss(records)

    # Weight sharing: recursive tower reuses the same layer objects each step.
    rec_tower: SharedRecursiveDenoiserTower = recursive.denoiser  # type: ignore[assignment]
    f_ids = {id(layer) for layer in rec_tower._f_layers}
    g_ids = {id(layer) for layer in rec_tower._g_layers}

    # Deep-supervision metrics: the pre-update decomposition (matches the
    # metrics whose gradient was actually backpropagated in phase 4).
    deep_metrics = {
        k: v for k, v in recursive_pre_metrics.items() if k.startswith("recursive_depth")
    }

    # (6) Round-trip save/load for the recursive model.
    tmp = Path("outputs/runs/slm138-recursive-denoiser-tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    ckpt = tmp / "recursive.pt"
    recursive.save(ckpt)
    loaded = TwoTowerModel.from_checkpoint(ckpt, device="cpu")
    loaded_ok = (
        loaded.config.denoiser_arch == "shared_recursive"
        and isinstance(loaded.denoiser, SharedRecursiveDenoiserTower)
    )

    version_stamp = build_version_stamp("model.recursive_denoiser")
    code_dirty = version_stamp.get("code_dirty")
    gate = _clean_tree_gate(code_dirty=code_dirty, allow_dirty=allow_dirty)
    diff_hash = _diff_hash() if code_dirty else None

    resolved_training_corruption_seed = (
        training_corruption_seed
        if training_corruption_seed is not None
        else rng_namespace_report(base_seed)["training_corruption"]
    )

    return {
        "matrix_set": "slm138-shared-recursive-denoiser",
        "matrix_version": "slm138-v1",
        "run_id": "slm138_recursive_denoiser_fixture",
        "status": "wiring_only",
        "claim_class": "wiring",
        "denoiser_architectures": ["stacked", "shared_recursive"],
        "stacked_params": _count_params(stacked),
        "recursive_params": _count_params(recursive),
        "forward_shapes": {
            "stacked": list(stacked_forward.shape),
            "recursive": list(recursive_forward.shape),
        },
        "losses": {
            "stacked": float(stacked_loss.detach().cpu()),
            "recursive": float(recursive_loss.detach().cpu()),
        },
        "post_update_verification": {
            "stacked_loss": float(stacked_post_loss.detach().cpu()),
            "recursive_loss": float(recursive_post_loss.detach().cpu()),
            "note": (
                "training_loss re-evaluated after one optimizer step with the "
                "training_corruption RNG checkpoint restored to its "
                "pre-update state -- the only source of difference from "
                "'losses' above is the parameter update, never a shifted "
                "corruption draw."
            ),
        },
        "recursive_weight_sharing": {
            "f_layer_object_count": len(f_ids),
            "g_layer_object_count": len(g_ids),
            "total_shared_layers": len(rec_tower.layers),
        },
        "architecture_comparison": architecture_comparison.as_dict(),
        "control_arm_table": control_arm_table,
        "control_arms_built": list(BUILT_ARM_IDS),
        "control_arms_deferred": list(DEFERRED_ARM_IDS),
        "arm_f_dual_view": arm_f_dual_view,
        "deep_supervision_metrics": deep_metrics,
        "checkpoint_roundtrip_ok": loaded_ok,
        "rng_contract": {
            "version": RNG_CONTRACT_VERSION,
            "base_seed": base_seed,
            "probe_order": probe_order,
            "insert_extra_probe": insert_extra_probe,
            "training_corruption_seed": resolved_training_corruption_seed,
            "namespace_seeds": rng_namespace_report(base_seed),
            "namespace_seeds_note": (
                "model_initialization uses config.seed directly via "
                "TwoTowerModel.__init__ (unchanged SLM-237/238 behavior); "
                "shape_probe_* namespaces are isolated via fork_rng and "
                "provably do not perturb training_corruption; "
                "training_batch_order/control_only are declared for the "
                "contract but training_batch_order is not exercised by this "
                "single-fixed-batch fixture."
            ),
        },
        "evidence_gate": {
            **gate,
            "code_dirty": code_dirty,
            "diff_hash": diff_hash,
            "allow_dirty": allow_dirty,
        },
        "provenance_hashes": {
            "config_hash": _config_hash(stacked.config),
            "recursive_config_hash": _config_hash(recursive.config),
            "data_record_hash": _records_hash(records),
            "tokenizer_hash": _tokenizer_hash(stacked.tokenizer),
        },
        "note": (
            "Wiring-only evidence. Full matched-block evaluation arms and GPU "
            "training are deferred."
        ),
        "version_stamp": version_stamp,
    }


def _plan_only_report() -> dict[str, Any]:
    return {
        "matrix_set": "slm138-shared-recursive-denoiser",
        "matrix_version": "slm138-v1",
        "run_id": "slm138_recursive_denoiser_plan",
        "status": "plan_only",
        "claim_class": "wiring",
        "denoiser_architectures": ["stacked", "shared_recursive"],
        "note": "plan-only: no models instantiated or trained",
        "version_stamp": build_version_stamp("model.recursive_denoiser"),
    }


def _canonical_json(report: dict[str, Any]) -> str:
    """JSON text used for byte-identity comparison across runs -- excludes
    the one structurally-nondeterministic field (``version_stamp.stamped_at``,
    a wall-clock timestamp)."""
    trimmed = json.loads(json.dumps(report, default=str))
    stamp = trimmed.get("version_stamp")
    if isinstance(stamp, dict):
        stamp.pop("stamped_at", None)
    return json.dumps(trimmed, indent=2, sort_keys=True)


def _digest(report: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(report).encode("utf-8")).hexdigest()


def _classify_field(name: str, a: Any, b: Any) -> str:
    if a == b:
        return "exact"
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if abs(float(a) - float(b)) <= 1e-9:
            return "tolerance"
    return "mismatch"


#: Fields intentionally excluded from the "must be identical" comparison:
#: ``version_stamp`` carries a wall-clock timestamp, and ``rng_contract`` is
#: the *configuration echo* of which base_seed/probe_order/insert_extra_probe/
#: training_corruption_seed a run used -- by construction those differ across
#: the deliberate permutation/seed-override runs this report compares. The
#: RNG namespace *seeds themselves* (``rng_contract.namespace_seeds``) are
#: identical for a fixed base_seed regardless of permutation and are checked
#: separately via the top-level ``rng_namespace_seeds`` field.
_COMPARISON_EXCLUDED_FIELDS = {"version_stamp", "rng_contract"}


def _compare_reports(base: dict[str, Any], other: dict[str, Any]) -> dict[str, str]:
    keys = (set(base) | set(other)) - _COMPARISON_EXCLUDED_FIELDS
    classification: dict[str, str] = {}
    for key in sorted(keys):
        classification[key] = _classify_field(key, base.get(key), other.get(key))
    return classification


def _determinism_report(*, base_seed: int = 0) -> dict[str, Any]:
    """Build ``FixtureDeterminismReportV1`` from real repeated executions +
    call-order permutations (SLM-239 required artifact)."""
    run_a = _run_fixture(base_seed=base_seed, allow_dirty=True)
    run_b = _run_fixture(base_seed=base_seed, allow_dirty=True)
    permuted_order = _run_fixture(
        base_seed=base_seed, probe_order="recursive_first", allow_dirty=True
    )
    extra_probe = _run_fixture(
        base_seed=base_seed, insert_extra_probe=True, allow_dirty=True
    )
    different_corruption = _run_fixture(
        base_seed=base_seed, training_corruption_seed=999_999, allow_dirty=True
    )

    digest_a = _digest(run_a)
    digest_b = _digest(run_b)
    digest_permuted = _digest(permuted_order)
    digest_extra = _digest(extra_probe)

    ab_classification = _compare_reports(run_a, run_b)
    permuted_classification = _compare_reports(run_a, permuted_order)
    extra_classification = _compare_reports(run_a, extra_probe)
    corruption_classification = _compare_reports(run_a, different_corruption)

    def _no_mismatch(classification: dict[str, str]) -> bool:
        """True if every field is 'exact' or within the documented float
        tolerance -- i.e. nothing crossed into a real 'mismatch'."""
        return all(v != "mismatch" for v in classification.values())

    def _all_exact(classification: dict[str, str]) -> bool:
        return all(v == "exact" for v in classification.values())

    # The determinism verdict is about repeatability + call-order/insertion
    # permutations only -- it does NOT fold in the different-corruption-seed
    # comparison below, which is *expected* to differ (that's a namespace
    # isolation check, not a reproducibility check). Run A vs run B share an
    # identical config, so full-artifact digest equality is the correct
    # strict check; the permutation/extra-probe runs deliberately echo a
    # different ``rng_contract`` config (excluded from
    # ``_compare_reports``/digest is not applicable there), so those are
    # judged on their *measured* field classification instead.
    repeat_bit_exact = digest_a == digest_b
    bit_exact = (
        repeat_bit_exact
        and _all_exact(permuted_classification)
        and _all_exact(extra_classification)
    )
    within_tolerance = (
        _no_mismatch(ab_classification)
        and _no_mismatch(permuted_classification)
        and _no_mismatch(extra_classification)
    )
    if bit_exact:
        verdict = "bit_exact"
    elif within_tolerance:
        verdict = "numerically_stable"
    else:
        verdict = "failed"

    # Namespace isolation check (reported alongside, not folded into
    # `verdict`): a different declared training_corruption seed must change
    # only the corruption-dependent fields and no others (SLM-239 test #4).
    corruption_only_fields = {
        "losses",
        "post_update_verification",
        "deep_supervision_metrics",
        "rng_contract",
    }
    corruption_changed_expected = corruption_classification.get("losses") != "exact"
    corruption_unexpected_changes = {
        k: v
        for k, v in corruption_classification.items()
        if v != "exact" and k not in corruption_only_fields
    }
    namespace_isolation_ok = (
        corruption_changed_expected and not corruption_unexpected_changes
    )

    return {
        "report_schema": "FixtureDeterminismReportV1",
        "base_seed": base_seed,
        "rng_contract_version": RNG_CONTRACT_VERSION,
        "run_a_digest": digest_a,
        "run_b_digest": digest_b,
        "permutation_digests": {
            "recursive_first_probe_order": digest_permuted,
            "extra_harmless_probe": digest_extra,
        },
        "field_classification": {
            "run_a_vs_run_b": ab_classification,
            "run_a_vs_permuted_probe_order": permuted_classification,
            "run_a_vs_extra_probe": extra_classification,
            "run_a_vs_different_training_corruption_seed": corruption_classification,
        },
        "different_training_corruption_seed_unexpected_changes": (
            corruption_unexpected_changes
        ),
        "namespace_isolation_ok": namespace_isolation_ok,
        "rng_namespace_seeds": rng_namespace_report(base_seed),
        "code_commit": run_a["version_stamp"].get("code_commit"),
        "code_dirty": run_a["version_stamp"].get("code_dirty"),
        "config_hash": run_a["provenance_hashes"]["config_hash"],
        "data_record_hash": run_a["provenance_hashes"]["data_record_hash"],
        "tokenizer_hash": run_a["provenance_hashes"]["tokenizer_hash"],
        "verdict": verdict,
        "note": (
            "Two isolated in-process fixture executions (run A/B), a "
            "call-order permutation (recursive-first shape probes), and an "
            "inserted-extra-harmless-probe permutation are compared field by "
            "field after excluding version_stamp.stamped_at. A distinct "
            "training-corruption-seed run is also compared and is *expected* "
            "to differ only in losses/post_update_verification/"
            "deep_supervision_metrics/rng_contract -- any other field "
            "changing under a different corruption seed would be a real "
            "namespace-isolation defect, listed in "
            "different_training_corruption_seed_unexpected_changes."
        ),
        "version_stamp": build_version_stamp("model.recursive_denoiser"),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# SLM-138: Shared recursive denoiser tower fixture ({report.get('run_id')})",
        "",
        f"Matrix set: `{report.get('matrix_set')}`  ",
        f"Version: `{report.get('matrix_version')}`  ",
        f"Status: **{report.get('status')}**",
        "",
        "## What this exercises",
        "",
        "A drop-in ``SharedRecursiveDenoiserTower`` that preserves the ``DenoiserTower`` "
        "public contract. The fixture builds tiny TwoTower models for both ``stacked`` "
        "and ``shared_recursive`` denoiser architectures, runs forward passes and "
        "training_loss, verifies shapes/gradients, confirms object-identity weight "
        "sharing across recursions, and round-trips a recursive checkpoint. SLM-239 "
        "(RSC-A03): RNG usage now follows an explicit, disjoint namespace contract -- "
        "see the ``rng_contract`` field below. SLM-240 (RSC-A04): the retracted "
        "same-parameter-count/layer-names claim is replaced by a real, measured "
        "``ArchitectureComparisonReportV1`` -- see 'Architecture comparison' below.",
        "",
        "## Architectures",
        "",
        ", ".join(f"`{a}`" for a in report.get("denoiser_architectures", [])),
        "",
    ]

    if "forward_shapes" in report:
        lines.extend(
            [
                "## Forward shapes",
                "",
                f"- stacked: `{report['forward_shapes']['stacked']}`",
                f"- recursive: `{report['forward_shapes']['recursive']}`",
                "",
            ]
        )

    comparison = report.get("architecture_comparison")
    if comparison:
        param_matched = (
            comparison["parameter_count_total"]["stacked"]
            == comparison["parameter_count_total"]["recursive"]
        )
        delta = comparison["parameter_count_delta"]
        # Percentage against the *whole TwoTowerModel* (stacked_params/
        # recursive_params below), matching how this delta is normally cited
        # (e.g. "+14.23% over a 64,994-parameter stacked model") -- the delta
        # itself is identical whether measured at the tower level
        # (comparison['parameter_count_total'], context-tower-free) or the
        # whole-model level, since the context tower/tokenizer embeddings are
        # identical between the two configs and cancel out of the subtraction.
        whole_model_stacked = report.get("stacked_params")
        whole_model_recursive = report.get("recursive_params")
        if whole_model_stacked and whole_model_recursive:
            assert whole_model_recursive - whole_model_stacked == delta, (
                "whole-model parameter delta must equal the tower-level "
                "architecture_comparison delta (context tower is identical "
                "across both configs)"
            )
            delta_pct = (delta / whole_model_stacked) * 100.0
        else:
            delta_pct = comparison["parameter_count_delta_pct"]
        behavioral_parity = (
            "not claimed"
            if not comparison["behaviorally_equivalent_under_declared_degeneracy"]
            else "claimed (measured equivalent under this exact input)"
        )
        lines.extend(
            [
                "## Architecture comparison (SLM-240 / RSC-A04)",
                "",
                "Independently measured comparison dimensions -- never a single "
                "collapsed `parity` claim (see `ArchitectureComparisonReportV1`).",
                "",
                f"- interface-compatible: **{str(comparison['interface_compatible']).lower()}**",
                f"- output-shape-compatible: **{str(comparison['output_shape_compatible']).lower()}**",
                f"- parameter-matched: **{str(param_matched).lower()}**",
                f"- parameter delta: `{delta:+d}` ({delta_pct:+.2f}%) -- "
                f"reproduced from `recursive_zstate_parameter_delta(d_model="
                f"{comparison['d_model']}, max_len={comparison['max_len']})`, "
                "exactly `z_latent` + `ctx_proj`",
                f"- parameter_count_denoiser (transition layers only, "
                f"architecture-independent): `{comparison['parameter_count_denoiser']}`",
                f"- behavioral parity: **{behavioral_parity}**",
                f"- claim class: **{comparison['claim_class']}**",
                f"- block evaluations per forward: `{comparison['block_evaluations_per_forward']}`",
                "",
            ]
        )

    control_arms = report.get("control_arm_table")
    if control_arms:
        lines.extend(
            [
                "## SLM-241 (RSC-A05) control arm table",
                "",
                "Real, measured resource accounting per built control arm -- "
                "never a raw loss or a winner (see `docs/design/iter-rsc-a05-*`"
                " for the full formulas/residuals). Built arms: "
                f"{', '.join(report.get('control_arms_built', []))}. Deferred: "
                f"{', '.join(report.get('control_arms_deferred', []))}.",
                "",
                "| arm | denoiser_arch | z_state_mode | params (Δ vs A) | "
                "block evals | matched? |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for arm in control_arms:
            lines.append(
                f"| {arm['arm_id']} | `{arm['denoiser_arch']}` | "
                f"`{arm['z_state_mode']}` | {arm['parameter_count_total']} "
                f"({arm['parameter_count_delta_vs_baseline']:+d}) | "
                f"{arm['block_evaluations_per_forward']} | "
                f"{arm['within_matching_tolerance']} |"
            )
        lines.append("")

    dual = report.get("arm_f_dual_view")
    if dual:
        bm = dual["block_evaluation_matched"]
        pn = dual["parameter_nearest"]
        lines.extend(
            [
                "## Arm F dual view (block-evaluation-matched vs parameter-nearest)",
                "",
                "Arm F (unshared depth-matched tower) has exactly one free dial "
                "(`n_layers`), so it cannot match both arm B's block-evaluation "
                "count and its parameter count simultaneously -- both real, "
                "measured constructions are reported below with an explicit "
                "residual on whichever dimension is not matched.",
                "",
                f"- Target arm: **{dual['target_arm']}** -- "
                f"`{dual['target_total_parameters']}` parameters, "
                f"`{dual['target_block_evaluations_per_forward']}` block "
                "evaluations per forward.",
                f"- Per-layer parameter cost (measured from real 1-layer/"
                "2-layer towers, never hard-coded): "
                f"`{dual['per_layer_parameter_cost_formula']['per_layer_parameters']}` "
                "per layer, "
                f"`{dual['per_layer_parameter_cost_formula']['common_parameters']}` "
                "common (non-block) parameters.",
                "",
                "| view | n_layers | block evals | Δ block evals vs B | params | "
                "Δ params vs B |",
                "| --- | --- | --- | --- | --- | --- |",
                f"| block_evaluation_matched | {bm['n_layers']} | "
                f"{bm['report']['block_evaluations_per_forward']} | 0 | "
                f"{bm['report']['parameter_count_total']} | "
                f"{bm['parameter_count_delta_vs_target_arm_b']:+d} |",
                f"| parameter_nearest | {pn['n_layers']} | "
                f"{pn['report']['block_evaluations_per_forward']} | "
                f"{pn['block_evaluations_delta_vs_target_arm_b']:+d} | "
                f"{pn['report']['parameter_count_total']} | "
                f"{pn['parameter_count_delta_vs_target_arm_b']:+d} |",
                "",
                "Neither row is a 'matched' claim on both dimensions at once -- "
                "`block_evaluation_matched` is the `control_arm_table` \"F\" row "
                "above; `parameter_nearest` is a separate construction reported "
                "only here.",
                "",
            ]
        )

    if "losses" in report:
        lines.extend(
            [
                "## Losses",
                "",
                "**Objective-decomposition warning:** the raw scalar losses below "
                "are *not* a quality/parameter-matched comparison -- the two "
                "architectures have different parameter counts (see "
                "'Architecture comparison' above) and the recursive arm's loss "
                "includes deep-supervision terms whose exact weighting/mode is "
                "governed by SLM-238 (RSC-A02)'s `recursive_depth_aux_mode` "
                "(see `deep_supervision_metrics` below and "
                "`docs/design/iter-rsc-a02-*`); placing these two numbers "
                "side by side never implies one architecture is better.",
                "",
                f"- stacked: `{report['losses']['stacked']:.6f}`",
                f"- recursive: `{report['losses']['recursive']:.6f}`",
                "",
            ]
        )

    post = report.get("post_update_verification")
    if post:
        lines.extend(
            [
                "## Post-update verification (restored corruption-RNG checkpoint)",
                "",
                f"- stacked: `{post['stacked_loss']:.6f}`",
                f"- recursive: `{post['recursive_loss']:.6f}`",
                "",
            ]
        )

    sharing = report.get("recursive_weight_sharing")
    if sharing:
        lines.extend(
            [
                "## Recursive weight sharing",
                "",
                f"- F-update distinct layer objects: {sharing['f_layer_object_count']}",
                f"- G-update distinct layer objects: {sharing['g_layer_object_count']}",
                f"- Total shared transition layers: {sharing['total_shared_layers']}",
                "",
            ]
        )

    deep = report.get("deep_supervision_metrics")
    if deep:
        lines.extend(
            [
                "## Deep-supervision metrics",
                "",
            ]
        )
        for k, v in deep.items():
            lines.append(f"- `{k}`: {v}")
        lines.append("")

    rng = report.get("rng_contract")
    if rng:
        lines.extend(
            [
                "## RNG contract",
                "",
                f"- Contract version: `{rng['version']}`",
                f"- Base seed: `{rng['base_seed']}`",
                f"- Probe order: `{rng['probe_order']}`",
                f"- Training-corruption seed: `{rng['training_corruption_seed']}`",
                f"- Namespace seeds: `{rng['namespace_seeds']}`",
                "",
            ]
        )

    gate = report.get("evidence_gate")
    if gate:
        lines.extend(
            [
                "## Clean-tree evidence gate",
                "",
                f"- Comparable/claim-grade: **{gate['comparable']}**",
                f"- code_dirty: `{gate['code_dirty']}`",
                f"- diff_hash: `{gate['diff_hash']}`",
                "",
            ]
        )

    if "checkpoint_roundtrip_ok" in report:
        lines.extend(
            [
                "## Checkpoint round-trip",
                "",
                f"Recursive checkpoint save/load OK: **{report['checkpoint_roundtrip_ok']}**",
                "",
            ]
        )

    lines.extend(
        [
            "## Fixture caveat",
            "",
            report.get(
                "note",
                "Wiring-only evidence. Full matched-block evaluation arms and GPU "
                "training are deferred.",
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _render_determinism_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# FixtureDeterminismReportV1: SLM-138 recursive denoiser fixture",
        "",
        f"Verdict: **{report['verdict']}**",
        f"RNG contract version: `{report['rng_contract_version']}`",
        f"Code commit: `{report['code_commit']}` (dirty={report['code_dirty']})",
        "",
        "## Digests",
        "",
        f"- Run A: `{report['run_a_digest']}`",
        f"- Run B: `{report['run_b_digest']}`",
        f"- Permuted probe order (recursive-first): `{report['permutation_digests']['recursive_first_probe_order']}`",
        f"- Extra harmless probe inserted: `{report['permutation_digests']['extra_harmless_probe']}`",
        "",
        "## Different training-corruption seed -- unexpected field changes",
        "",
        (
            f"`{report['different_training_corruption_seed_unexpected_changes']}`"
            if report["different_training_corruption_seed_unexpected_changes"]
            else "None -- only losses/post_update_verification/deep_supervision_metrics/rng_contract changed, as expected."
        ),
        "",
        report["note"],
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-138 shared recursive denoiser fixture"
    )
    parser.add_argument(
        "--mode",
        choices=("plan-only", "fixture", "determinism"),
        default="plan-only",
        help=(
            "plan-only emits the matrix skeleton; "
            "fixture exercises both denoiser architectures; "
            "determinism emits a FixtureDeterminismReportV1 from repeated "
            "runs + call-order permutations"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"outputs/runs/slm138-recursive-denoiser-{_today_slug()}"),
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "Debug override: run --mode fixture on a dirty tree anyway. The "
            "emitted artifact is marked non-comparable/claim_grade=false and "
            "must never be checked in as claim-grade evidence."
        ),
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="Base seed the RNG namespace contract derives all seeds from.",
    )
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "determinism":
        report = _determinism_report(base_seed=args.base_seed)
        report_path = output_dir / "fixture_determinism_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        markdown = _render_determinism_markdown(report)
        (output_dir / "fixture_determinism_report.md").write_text(
            markdown, encoding="utf-8"
        )
        print(markdown)
        print(f"\nReport JSON: {report_path}")
        return 0

    design_json = Path(f"docs/design/iter-slm138-recursive-denoiser-{_today_slug()}.json")
    design_md = Path(f"docs/design/iter-slm138-recursive-denoiser-{_today_slug()}.md")

    write_design_docs = True
    if args.mode == "plan-only":
        report = _plan_only_report()
    else:
        report = _run_fixture(base_seed=args.base_seed, allow_dirty=args.allow_dirty)
        gate = report["evidence_gate"]
        # SLM-239 requirement #3: fixture mode always runs, even on a dirty
        # tree, for local debugging. What is gated is the canonical
        # docs/design/ location -- a dirty run without an explicit
        # --allow-dirty acknowledgement never lands there; the local
        # outputs/runs/ artifact is still written either way so debugging
        # isn't blocked.
        if not gate["comparable"] and not args.allow_dirty:
            write_design_docs = False
            print(
                "warning: working tree is dirty "
                f"(code_dirty={gate['code_dirty']!r}); writing local "
                f"run-only evidence under {output_dir} but refusing to write "
                "docs/design/ claim-grade evidence. Commit/stash first, or "
                "pass --allow-dirty to also write a non-comparable "
                "docs/design debug artifact.",
                file=sys.stderr,
            )

    report_text = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    report_path = output_dir / "slm138_recursive_denoiser_report.json"
    report_path.write_text(report_text, encoding="utf-8")
    markdown = _render_markdown(report)
    (output_dir / "slm138_recursive_denoiser_report.md").write_text(
        markdown, encoding="utf-8"
    )

    if write_design_docs:
        design_json.parent.mkdir(parents=True, exist_ok=True)
        design_json.write_text(report_text, encoding="utf-8")
        design_md.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"\nReport JSON: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
