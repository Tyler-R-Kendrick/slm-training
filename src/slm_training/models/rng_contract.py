"""Explicit RNG namespace contract for deterministic recursive-denoiser fixtures.

SLM-239 (RSC-A03). See ``docs/design/iter-rsc-a03-*.md``.

Background: the SLM-138 fixture (``scripts/run_slm138_recursive_denoiser_fixture.py``)
builds tiny ``TwoTowerModel`` instances and calls ``training_loss`` on them.
``TwoTowerModel.__init__`` already reseeds the *global* torch RNG
(``torch.manual_seed(config.seed)``) at construction time, so model weights are
reproducible regardless of what happened before construction. But
``training_loss``'s internal corruption sampling (``_mask_targets``: the mask
rate, the noise draw, the visible-token corruption) reads the *global* torch
RNG too, unconditionally, without any reseed of its own -- by design, this
module does not change that (SLM-237/238's objective semantics are frozen).
That means any global-RNG-consuming code that runs *between* model
construction and the ``training_loss`` call -- e.g. a "harmless" forward
shape probe -- shifts the corruption draws deterministically, but the shift
itself depends on incidental call order/count. A fixture used for regression
evidence must not depend on that order.

``_mask_targets`` also reads a *second*, independent RNG source: each
``TwoTowerModel`` keeps a persistent per-instance ``random.Random(config.seed)``
(``self._rng``) used for the "ensure at least one predictable token per row"
fallback and mixed-pattern statement-span selection. Restoring only the
global torch RNG state is *not* sufficient for a byte-identical repeated
``training_loss`` evaluation (verified empirically: doing so without also
restoring ``self._rng`` reproduces a different loss on the second call) --
:func:`seed_training_corruption`/:class:`RngCheckpoint` manage both sources
together for exactly this reason.

This module gives the fixture (and any test that wants the same guarantee) a
small, explicit, disjoint RNG namespace contract:

- ``model_initialization`` -- unchanged: ``TwoTowerModel.__init__`` seeds
  ``torch.manual_seed(config.seed)`` itself. Namespace offset 0 documents that
  this module does not touch it.
- ``shape_probe_inputs`` / ``shape_probe_context`` -- synthetic forward-probe
  tensors (token ids / context floats). Drawn via :func:`isolated_draw`, which
  runs the draw inside ``torch.random.fork_rng`` so the *outer* global stream
  is provably unaffected by the probe's existence, order, or count.
- ``training_corruption`` -- the stream ``training_loss``'s internal
  ``_mask_targets`` reads. Seeded explicitly via
  :func:`seed_training_corruption` immediately before each ``training_loss``
  call so its draws depend only on the declared corruption seed, never on
  what ran earlier in the process.
- ``training_batch_order`` -- reserved for future batch-shuffling; the
  SLM-138 fixture uses a fixed 1-2 record batch with no shuffling, so this
  namespace is declared but not exercised here.
- ``control_only`` -- reserved for any other incidental randomness (e.g.
  auxiliary-head init already isolated by ``TwoTowerModel``'s own
  ``isolated_aux_init`` helper) that must never leak into the corruption
  stream.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import torch

#: Bump only on a breaking change to the namespace offsets or semantics below
#: (never silently -- old reports would stop being comparable to new ones).
RNG_CONTRACT_VERSION = "FixtureRngContractV1"

#: Disjoint, fixed-forever-once-shipped offsets from a fixture's base seed.
#: Large gaps leave headroom for per-record/per-depth sub-draws without
#: namespaces colliding.
NAMESPACE_OFFSETS: dict[str, int] = {
    "model_initialization": 0,
    "shape_probe_inputs": 10_000,
    "shape_probe_context": 20_000,
    "training_corruption": 30_000,
    "training_batch_order": 40_000,
    "control_only": 50_000,
    # SLM-241 (RSC-A05): declared, disjoint per-architecture seeds for the
    # matched recursive control arms (see
    # slm_training.models.recursive_control_arms /
    # RecursiveControlInitializationV1 below). These are *reserved contract
    # surface*, not literally consumed mid-construction by the current
    # single-pass tower constructors -- see RecursiveControlInitializationV1's
    # docstring for exactly what these seeds do and do not guarantee today.
    "arch_specific:stacked": 60_000,
    "arch_specific:shared_recursive": 70_000,
    "arch_specific:shared_recursive_y_only": 80_000,
    "arch_specific:shared_recursive_no_extra_capacity": 90_000,
}

DECLARED_NAMESPACES = tuple(sorted(NAMESPACE_OFFSETS))

T = TypeVar("T")


def derive_seed(base_seed: int, namespace: str) -> int:
    """Deterministic, disjoint seed for ``namespace`` given a fixture's
    ``base_seed``. Raises on an undeclared namespace -- fail closed, matching
    the SLM-237/238 validator convention in ``twotower.py``."""
    if namespace not in NAMESPACE_OFFSETS:
        raise ValueError(
            f"unknown RNG namespace {namespace!r}; declared namespaces are "
            f"{DECLARED_NAMESPACES}"
        )
    return int(base_seed) + NAMESPACE_OFFSETS[namespace]


def isolated_draw(base_seed: int, namespace: str, fn: Callable[[], T]) -> T:
    """Run ``fn()`` under a namespace-derived seed inside ``fork_rng`` so the
    *outer* global RNG state is byte-identical before and after, regardless
    of what ``fn`` consumes internally. Use for shape probes / other
    incidental draws that must never perturb ``training_corruption``.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(derive_seed(base_seed, namespace))
        return fn()


@dataclass(frozen=True)
class RngCheckpoint:
    """A snapshot of the global CPU torch RNG state, plus (optionally) a
    ``TwoTowerModel`` instance's private ``self._rng`` (``random.Random``)
    state -- ``_mask_targets`` reads both, so a byte-identical repeated
    ``training_loss`` evaluation requires restoring both together."""

    cpu_state: torch.Tensor
    python_random_state: tuple[Any, ...] | None = None

    @classmethod
    def capture(cls, model: Any | None = None) -> "RngCheckpoint":
        py_state = None
        rng = getattr(model, "_rng", None)
        if isinstance(rng, random.Random):
            py_state = rng.getstate()
        return cls(cpu_state=torch.get_rng_state().clone(), python_random_state=py_state)

    def restore(self, model: Any | None = None) -> None:
        torch.set_rng_state(self.cpu_state)
        rng = getattr(model, "_rng", None)
        if isinstance(rng, random.Random) and self.python_random_state is not None:
            rng.setstate(self.python_random_state)

    def digest(self) -> str:
        return state_digest(self.cpu_state)


def state_digest(state: torch.Tensor) -> str:
    """Stable hex digest of a torch RNG state tensor (or any CPU byte tensor)."""
    return hashlib.sha256(state.cpu().contiguous().numpy().tobytes()).hexdigest()


def seed_training_corruption(
    base_seed: int,
    model: Any | None = None,
    *,
    namespace: str = "training_corruption",
    override_seed: int | None = None,
) -> RngCheckpoint:
    """Deterministically seed both RNG sources a ``training_loss`` call's
    internal corruption sampling reads (unchanged legacy behavior -- see
    ``TwoTowerModel._mask_targets``): the *global* torch RNG, and -- when
    ``model`` is given -- that model's private ``self._rng``
    (``random.Random``), reseeded in place from the same derived seed.

    Returns a :class:`RngCheckpoint` captured immediately after seeding, so a
    caller needing a second evaluation with identical corruption draws (e.g.
    verifying a loss before/after one optimizer step) can ``.restore(model)``
    it instead of letting the second ``training_loss`` call silently consume
    the *next* draws in both streams -- the SLM-239 "do not call
    training_loss twice with an implicitly advanced corruption RNG"
    requirement.

    ``override_seed`` bypasses the namespace derivation entirely (used to
    vary *only* the training-corruption seed while holding
    ``model_initialization``/shape-probe seeds fixed, e.g. to prove that a
    different corruption seed changes corruption-dependent fields and no
    others).
    """
    seed = int(override_seed) if override_seed is not None else derive_seed(
        base_seed, namespace
    )
    torch.manual_seed(seed)
    rng = getattr(model, "_rng", None)
    if isinstance(rng, random.Random):
        rng.seed(seed)
    return RngCheckpoint.capture(model)


def rng_namespace_report(base_seed: int) -> dict[str, int]:
    """Seeds actually assigned to each declared namespace for ``base_seed`` --
    persisted verbatim into fixture/determinism-report JSON so the contract
    version + concrete seeds are part of the evidence, not just the code."""
    return {ns: derive_seed(base_seed, ns) for ns in DECLARED_NAMESPACES}


# ---------------------------------------------------------------------------
# SLM-241 (RSC-A05): RecursiveControlInitializationV1 -- the fair-init
# contract a matched recursive-control campaign needs. Extends this module's
# namespace machinery rather than forking a parallel one (see the
# ``arch_specific:*`` namespaces above).
# ---------------------------------------------------------------------------

RECURSIVE_CONTROL_INIT_VERSION = "RecursiveControlInitializationV1"


@dataclass(frozen=True)
class RecursiveControlInitializationV1:
    """Fairness/init evidence for a set of constructed control-arm towers.

    Built only by :func:`build_recursive_control_initialization` from real
    constructed modules -- nothing here is hand-assembled.

    ``common_tensor_hashes_match_across_arms`` is *measured*, not assumed: all
    the towers this module builds from a shared ``base_seed`` reach every
    common tensor (``tok``/``pos``/``kind``/``layers.*``/``norm``/``lm_head``)
    through an identical construction order and shape before any
    architecture-specific tensor is registered (verified empirically by
    ``tests/test_models/test_recursive_denoiser.py``), so re-seeding the
    global RNG to the same ``model_initialization`` seed immediately before
    each tower's construction -- exactly what ``TwoTowerModel.__init__``
    already does per instance -- produces bit-identical common tensors.

    ``architecture_specific_seeds`` are declared, pairwise-disjoint seeds
    from :data:`NAMESPACE_OFFSETS`'s ``arch_specific:*`` entries -- **not**
    literally consumed inside ``SharedRecursiveDenoiserTower.__init__`` today.
    The current constructors are single-pass: architecture-specific tensors
    (``z_latent``/``ctx_proj`` for ``z_state_mode="full"``; none at all for
    ``"y_only"``/``"parameter_free"``) are registered by ordinary Python
    attribute assignment *after* every common tensor in the same
    ``model_initialization``-seeded call, and (per the measurement above)
    that placement already does not perturb the common tensors' draws. These
    seeds are reserved contract surface: real, deterministic, and disjoint,
    for any future two-phase constructor or standalone telemetry that draws
    architecture-specific tensors independently of the common-prefix draw
    count -- reported here honestly as declared rather than claimed as
    actually-consumed.
    """

    contract_version: str
    base_seed: int
    common_tensor_names: tuple[str, ...]
    common_tensor_hashes: dict[str, str]
    common_tensor_hashes_match_across_arms: bool
    architecture_specific_seeds: dict[str, int]
    architecture_specific_tensor_names_and_shapes: dict[str, dict[str, list[int]]]
    optimizer_group_membership: dict[str, dict[str, list[str]]]
    optimizer_initial_state_hash: str
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.contract_version != RECURSIVE_CONTROL_INIT_VERSION:
            raise ValueError(
                f"contract_version={self.contract_version!r} does not match "
                f"{RECURSIVE_CONTROL_INIT_VERSION!r}."
            )
        seeds = list(self.architecture_specific_seeds.values())
        if len(set(seeds)) != len(seeds):
            raise ValueError(
                "architecture_specific_seeds must be pairwise disjoint, got "
                f"{self.architecture_specific_seeds!r}."
            )
        if not self.common_tensor_hashes_match_across_arms:
            raise ValueError(
                "common_tensor_hashes_match_across_arms is False -- a "
                "matched-control fairness report must never be built from "
                "towers whose common tensors were not actually identically "
                "initialized. Reseed to the same model_initialization seed "
                "immediately before constructing each arm."
            )


def build_recursive_control_initialization(
    *,
    base_seed: int,
    arm_towers: dict[str, Any],
    arm_denoiser_arch: dict[str, str],
    arm_optimizer_group_membership: dict[str, dict[str, list[str]]] | None = None,
) -> RecursiveControlInitializationV1:
    """Build a real :class:`RecursiveControlInitializationV1` from constructed
    control-arm towers.

    ``arm_towers`` maps an arm id (e.g. ``"A"``/``"B"``/``"C"``/``"D"``) to an
    already-constructed ``nn.Module`` -- the caller is responsible for
    reseeding the global RNG to ``derive_seed(base_seed,
    "model_initialization")`` immediately before constructing each one (the
    same discipline ``TwoTowerModel.__init__`` already applies per instance),
    so common tensors are actually bit-identical; this function *measures*
    that outcome rather than assuming it.

    ``arm_denoiser_arch`` maps the same arm ids to their canonical
    ``denoiser_arch`` string (see
    ``slm_training.models.twotower.KNOWN_DENOISER_ARCHES`` /
    ``slm_training.models.recursive_control_arms.ARM_DENOISER_ARCH``), used
    only to select each arm's declared ``arch_specific:<denoiser_arch>``
    seed.
    """
    if not arm_towers:
        raise ValueError("arm_towers must be non-empty")

    named_by_arm = {arm: dict(tower.named_parameters()) for arm, tower in arm_towers.items()}
    arm_ids = sorted(named_by_arm)
    common_names = set.intersection(*(set(d) for d in named_by_arm.values()))
    common_names = {
        name
        for name in common_names
        if len({tuple(named_by_arm[arm][name].shape) for arm in arm_ids}) == 1
    }

    common_hashes: dict[str, str] = {}
    mismatched: list[str] = []
    for name in sorted(common_names):
        per_arm_hashes = {
            arm: state_digest(named_by_arm[arm][name].detach()) for arm in arm_ids
        }
        digests = set(per_arm_hashes.values())
        if len(digests) != 1:
            mismatched.append(name)
            continue
        common_hashes[name] = next(iter(digests))
    hashes_match = not mismatched

    architecture_specific_seeds: dict[str, int] = {}
    architecture_specific_tensors: dict[str, dict[str, list[int]]] = {}
    for arm in arm_ids:
        arch = arm_denoiser_arch.get(arm)
        if arch is not None:
            architecture_specific_seeds[arm] = derive_seed(
                base_seed, f"arch_specific:{arch}"
            )
        architecture_specific_tensors[arm] = {
            name: list(tensor.shape)
            for name, tensor in named_by_arm[arm].items()
            if name not in common_names
        }

    optimizer_groups = arm_optimizer_group_membership or {
        arm: {
            "base": [
                name
                for name, tensor in named_by_arm[arm].items()
                if tensor.requires_grad
            ]
        }
        for arm in arm_ids
    }
    # AdamW (and every other torch optimizer) lazily allocates per-parameter
    # state on the first .step() call -- state_dict()["state"] is genuinely
    # empty before that. The hash below pins that fact deterministically
    # rather than fabricating a nonexistent initial moment/variance value.
    optimizer_initial_state_hash = _stable_dict_hash({})

    notes = (
        "common_tensor_hashes cover only names present with identical shape "
        "in every provided arm (tok/pos/kind/layers.*/norm/lm_head when "
        "recursive_transition_layers matches the stacked baseline's "
        "n_layers); architecture-specific tensors (e.g. z_latent/ctx_proj "
        "for the 'full' z_state_mode) are listed under "
        "architecture_specific_tensor_names_and_shapes instead.",
        "architecture_specific_seeds are declared/reserved, not literally "
        "consumed mid-construction by the current single-pass tower "
        "constructors -- see the class docstring.",
        "optimizer_initial_state_hash is the hash of an empty optimizer "
        "state (pre-first-step); AdamW allocates exp_avg/exp_avg_sq lazily.",
    )

    return RecursiveControlInitializationV1(
        contract_version=RECURSIVE_CONTROL_INIT_VERSION,
        base_seed=base_seed,
        common_tensor_names=tuple(sorted(common_hashes)),
        common_tensor_hashes=common_hashes,
        common_tensor_hashes_match_across_arms=hashes_match,
        architecture_specific_seeds=architecture_specific_seeds,
        architecture_specific_tensor_names_and_shapes=architecture_specific_tensors,
        optimizer_group_membership=optimizer_groups,
        optimizer_initial_state_hash=optimizer_initial_state_hash,
        notes=notes,
    )


def _stable_dict_hash(obj: dict[str, Any]) -> str:
    import json

    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
