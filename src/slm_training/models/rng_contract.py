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
