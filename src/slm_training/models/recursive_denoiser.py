"""Shared-recursive denoiser tower (SLM-138).

A compact denoiser that recurses a small shared transition instead of stacking
independent TransformerBlocks.  It preserves the ``DenoiserTower`` public
contract (``forward`` / ``encode`` / ``project`` / ``set_runtime_symbol_features``
plus ``tok`` / ``kind`` / ``lm_head`` / ``max_len`` / ``layers``) so it can be
dropped into ``TwoTowerModel`` without changing masking, decode, or checkpoint
shapes.

Architecture (V1 primary):
  y_0 = token + position + kind + request-local symbol features
  z_0 = learned latent + projected pooled context + position
  for r in 1..R:
      z_r = z_{r-1} + F_theta(norm(z_{r-1} + y_{r-1}), context)
      y_r = y_{r-1} + G_theta(norm(y_{r-1} + z_r),     context)
      h_r = norm(y_r)
      logits_r = lm_head(h_r)

F_theta and G_theta are built from the same small set of
``TransformerBlock(cross_attn=True)`` instances; they are reused by object
identity every recursion, so the shared-transition parameter count is
independent of ``recursive_steps`` -- adding recursion steps changes compute
(more block evaluations per forward), never parameter count.

SLM-241 (RSC-A05) -- ``z_state_mode`` control arms. The constructor now
accepts a ``z_state_mode`` in ``{"full", "y_only", "parameter_free"}`` so a
matched-control campaign can isolate the y/z split from shared repeated
depth without a parallel/ad hoc implementation:

* ``"full"`` (default, byte-identical to the historical V1 behavior / arm
  **B** in ``docs/design/iter-rsc-a05-*``): learned ``z_latent`` +
  ``ctx_proj`` as described above.
* ``"y_only"`` (arm **C**): no distinct z state at all -- every recursion
  step runs the F-layers directly on ``norm(y)`` (in place of the z-update)
  and the G-layers on the result, so both updates flow through ``y`` alone.
  No ``z_latent``/``ctx_proj`` parameter exists; ``recursive_outputs``'s
  per-step block-evaluation count (``len(_f_layers) + len(_g_layers)`` per
  recursion) is unchanged from ``"full"``.
* ``"parameter_free"`` (arm **D**): the y/z split is kept structurally, but
  ``z``'s initial value is a parameter-free pooled-context broadcast
  (``context.mean(dim=1)`` expanded across positions, no learned
  ``max_len``-sized bank, no learned projection) -- removing exactly the two
  parameter tensors ``recursive_zstate_parameter_delta`` accounts for, so
  this arm's total parameter count matches a stacked baseline with the same
  ``recursive_transition_layers``/``n_layers`` to within 0 parameters (see
  ``docs/design/iter-rsc-a05-*`` for the measured residual).

See :mod:`slm_training.models.recursive_control_arms` for the named arm
registry (A-H; all eight arms are constructed as of this SLM-241 iteration --
none remain deferred) and resource accounting built on top of these modes.

SLM-241 (RSC-A05) second follow-up -- ``detach_between_steps`` (arm **H**,
stop-gradient recurrence). The constructor now also accepts
``detach_between_steps: bool = False``. When ``True``, ``recursive_outputs``
runs the *identical* forward recurrence (same ``y``/``z`` update equations,
same shared blocks, same shapes -- byte-identical forward values to
``detach_between_steps=False`` for the same seed/weights/inputs) but calls
``.detach()`` on the carried-forward ``y`` (and ``z``, when the z-state path
is active) at the end of every recursion step except the last, before the
*next* step reads them. ``.detach()`` only cuts the autograd graph -- it never
changes a tensor's numeric value -- so this flag changes gradient flow only:
a later step's loss can still backpropagate into the shared block weights
through *that step's own* application of them, but not through the recurrent
chain back into an earlier step's application. Arm H
(``denoiser_arch="shared_recursive"`` -- reused, not a new arch string, same
as arm G's reuse -- plus ``detach_between_steps=True`` on the tower) is the
mirror question to every other arm: does any gain from recurrence require
genuine backprop-through-recurrence ("recurrent credit assignment"), or does
merely re-applying the same shared weights repeatedly (without BPTT) capture
it? See ``docs/design/iter-rsc-a05-*`` for the forward-identity and
gradient-divergence evidence.

SLM-241 (RSC-A05) follow-up -- :class:`StackedMatchedStateDenoiserTower`
(arm E, ``denoiser_arch="stacked_matched_state"``): the mirror image of arm D.
D keeps the recursive y/z split but makes z's initial value parameter-free;
E keeps the stacked baseline's unshared, non-recursive block structure but
gives it the *same* parameter budget V1's z-state path adds -- a learned
``state[max_len, d_model]`` bank plus a ``state_ctx_proj`` context projection,
injected into the initial hidden state **exactly once**, before any
transition block runs. Unlike B/G, there is no recurrence and no repeated
consumption of this state across depth -- it isolates "added state capacity"
from "repeated shared computation" as two independent resource dimensions.
See :class:`StackedMatchedStateDenoiserTower`'s own docstring for the exact
formula and injection point.

SLM-240 (RSC-A04) correction -- read this before citing any "parity" claim
for this tower:

* **No same-parameter-count claim exists.** When ``recursive_transition_layers``
  equals ``DenoiserTower``'s ``n_layers``, the shared transition blocks are a
  1:1 name/shape match against ``DenoiserTower.layers`` -- but V1 *always*
  adds two new parameter tensors on top of that: ``z_latent``
  (``[max_len, d_model]``) and ``ctx_proj`` (``Linear(d_model, d_model)``).
  The total parameter count is therefore **never** equal to ``DenoiserTower``'s;
  the exact delta is ``d_model * (max_len + d_model + 1)`` (see
  :func:`recursive_zstate_parameter_delta`), reproduced from a formula, not a
  hard-coded constant. For the SLM-138 fixture config (``d_model=32``,
  ``max_len=256``) that delta is 9,248 parameters (+14.23% over the 64,994-
  parameter stacked model) -- see ``docs/design/iter-rsc-a04-*``.
* **``recursive_steps=1`` preserves the public interface and compatible
  tensor shapes, NOT output equivalence.** ``forward``/``encode``/``project``/
  ``set_runtime_symbol_features`` plus ``tok``/``kind``/``lm_head``/``max_len``/
  ``layers`` all exist with matching shapes, so a caller can swap towers
  without changing masking, decode, or downstream tensor plumbing. The
  z-state path (``z_latent``/``ctx_proj``) is active at every ``R`` including
  ``R=1``, so recursive and stacked outputs are numerically different by
  construction -- there is no configuration of this tower that reproduces
  ``DenoiserTower``'s outputs exactly.
* **Checkpoint layer-name compatibility is partial.** Only the shared
  transition-block keys (``layers.*``) plus the shared embedding/norm/head
  keys (``tok``/``pos``/``kind``/``norm``/``lm_head``) are name/shape
  compatible with ``DenoiserTower``. The z-state keys (``z_latent``,
  ``ctx_proj.weight``, ``ctx_proj.bias``) have no ``DenoiserTower``
  counterpart and require explicit initialization/migration -- see
  :func:`slm_training.models.checkpoint_migrate.migrate_to_shared_recursive_denoiser`,
  whose migration report lists them under ``initialized_keys``.
* **No parameter-efficiency or quality claim exists** for this tower vs
  ``DenoiserTower`` until a matched control campaign is run (a separate,
  later, out-of-scope issue that builds actual parameter-matched control
  constructors). This module's :class:`ArchitectureComparisonReportV1` /
  :func:`compare_denoiser_architectures` report each comparison dimension
  (interface, output shape, parameter counts, checkpoint bytes, per-forward
  block-evaluation counts, estimated FLOPs) independently and never collapse
  them into one ``parity`` boolean.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm_training.models.blocks import DenoiserTower, RMSNorm, TransformerBlock


#: SLM-241 (RSC-A05): valid ``z_state_mode`` values. Fail closed on anything
#: else -- a typo silently falling back to ``"full"`` would defeat the whole
#: point of a matched-control campaign that needs *no* z-state parameters for
#: arms C/D.
Z_STATE_MODES: tuple[str, ...] = ("full", "y_only", "parameter_free")
RECURSIVE_DIAGNOSTIC_UPDATE_MODES: tuple[str, ...] = ("as_is", "residual_delta")


@dataclass(frozen=True)
class RecursiveDepthDiagnosticsV1:
    """Detached, per-example recurrence health at one logical depth.

    State/update tensors have shape ``[B, T, D]``. Norms are per-example
    Frobenius norms over ``[T, D]``; each update/state ratio divides the update
    norm by the corresponding *pre-update* state norm (clamped at machine
    epsilon). Task metrics are per-example masked-token means. KL uses
    ``KL(p_depth || p_reference)``; the final depth therefore has zero
    ``kl_to_final`` and no ``kl_to_next``.
    """

    contract_version: str
    step: int
    update_mode: str
    y: torch.Tensor
    z: torch.Tensor | None
    y_update: torch.Tensor
    z_update: torch.Tensor | None
    y_norm: torch.Tensor
    z_norm: torch.Tensor | None
    y_update_norm: torch.Tensor
    z_update_norm: torch.Tensor | None
    y_update_state_ratio: torch.Tensor
    z_update_state_ratio: torch.Tensor | None
    target_count: torch.Tensor | None
    cross_entropy: torch.Tensor | None
    accuracy: torch.Tensor | None
    entropy: torch.Tensor | None
    kl_to_next: torch.Tensor | None
    kl_to_final: torch.Tensor | None


class SharedRecursiveDenoiserTower(nn.Module):
    """Recursive shared-transition denoiser matching the ``DenoiserTower`` contract."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        max_len: int = 512,
        dropout: float = 0.0,
        *,
        kind_ids: list[int] | None = None,
        n_kinds: int = 0,
        recursive_steps: int = 1,
        recursive_transition_layers: int | None = None,
        tie_output_embedding: bool = True,
        z_state_mode: str = "full",
        detach_between_steps: bool = False,
    ) -> None:
        super().__init__()
        if z_state_mode not in Z_STATE_MODES:
            raise ValueError(
                f"z_state_mode={z_state_mode!r} is not one of {Z_STATE_MODES!r}"
            )
        self.z_state_mode = z_state_mode
        # SLM-241 (RSC-A05) arm H: stop-gradient recurrence. Purely a
        # backward-graph flag -- adds no parameter, changes no forward
        # computation. See the module docstring above for the exact
        # semantics and slm_training.models.recursive_control_arms for the
        # arm registry.
        self.detach_between_steps = bool(detach_between_steps)
        self.d_model = d_model
        self.max_len = max_len
        self.recursive_steps = max(1, int(recursive_steps))
        self.recursive_transition_layers = (
            recursive_transition_layers
            if recursive_transition_layers is not None
            else n_layers
        )

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.kind: nn.Embedding | None = None
        if kind_ids is not None and n_kinds > 0:
            self.kind = nn.Embedding(n_kinds, d_model)
            lookup = torch.tensor(
                [int(k) for k in kind_ids[:vocab_size]]
                + [0] * max(0, vocab_size - len(kind_ids)),
                dtype=torch.long,
            )
            self.register_buffer("kind_lookup", lookup, persistent=True)
        else:
            self.register_buffer(
                "kind_lookup",
                torch.zeros(max(vocab_size, 1), dtype=torch.long),
                persistent=False,
            )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, dropout=dropout, cross_attn=True)
                for _ in range(self.recursive_transition_layers)
            ]
        )
        # Split the shared transition into the z-update (F) and y-update (G).
        # For n=1 this puts the single block into G, making R=1 behave like a
        # single cross-attention block applied to y+z.
        f_end = self.recursive_transition_layers // 2
        self._f_layers = self.layers[:f_end]
        self._g_layers = self.layers[f_end:]

        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.tie_output_embedding = bool(tie_output_embedding)
        if self.tie_output_embedding:
            self.lm_head.weight = self.tok.weight
        else:
            self.lm_head.weight.data.copy_(self.tok.weight.data)

        # z-state path: learned latent + projected pooled context + position.
        # SLM-241 (RSC-A05): only the "full" mode (arm B / historical V1)
        # declares these parameter tensors at all -- "y_only" (arm C) and
        # "parameter_free" (arm D) never register a ``z_latent``/``ctx_proj``
        # attribute, so ``named_parameters()``/``state_dict()`` genuinely
        # contain no z-shaped parameter bank for those modes (never a
        # zeroed-out but still-declared tensor).
        if self.z_state_mode == "full":
            self.z_latent = nn.Parameter(torch.zeros(max_len, d_model))
            self.ctx_proj = nn.Linear(d_model, d_model)

        self._runtime_symbol_features: torch.Tensor | None = None

    def set_runtime_symbol_features(self, features: torch.Tensor | None) -> None:
        """Attach request-local vocabulary-row deltas (not checkpoint state)."""
        self._runtime_symbol_features = features

    def _features_for_batch(self, batch_size: int) -> torch.Tensor | None:
        features = self._runtime_symbol_features
        if features is None:
            return None
        if features.size(0) == batch_size:
            return features
        if features.size(0) == 1:
            return features.expand(batch_size, -1, -1)
        raise ValueError(
            f"runtime symbol feature batch {features.size(0)} != {batch_size}"
        )

    def _apply_layers(
        self,
        x: torch.Tensor,
        layers: list[TransformerBlock] | nn.ModuleList,
        self_pad_mask: torch.Tensor | None,
        ctx: torch.Tensor,
        ctx_pad_mask: torch.Tensor | None,
        *,
        return_last_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply a sequence of TransformerBlocks, optionally returning last attn."""
        attn: torch.Tensor | None = None
        last = len(layers) - 1
        for i, layer in enumerate(layers):
            want_attn = return_last_attn and i == last
            if want_attn:
                out = layer(
                    x,
                    self_pad_mask=self_pad_mask,
                    ctx=ctx,
                    ctx_pad_mask=ctx_pad_mask,
                    return_self_attn=True,
                )
                assert isinstance(out, tuple)
                x, attn = out
            else:
                out = layer(
                    x,
                    self_pad_mask=self_pad_mask,
                    ctx=ctx,
                    ctx_pad_mask=ctx_pad_mask,
                )
                x = out if not isinstance(out, tuple) else out[0]
        if return_last_attn:
            assert attn is not None
            return x, attn
        return x

    def initial_transition_state(
        self,
        noisy_ids: torch.Tensor,
        context: torch.Tensor,
        pad_id: int,
        ctx_pad_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Materialize the exact state and fixed conditioning for recurrence.

        The returned tensors are sufficient to replay :meth:`transition_step`
        without reading request-local symbol state again.  This is an
        evaluation-neutral boundary for JVP/VJP instrumentation; callers must
        treat ``context``, masks, and ``runtime_symbol_features`` as fixed when
        differentiating with respect to ``y``/``z``.
        """
        bsz, seq = noisy_ids.shape
        if seq > self.max_len:
            noisy_ids = noisy_ids[:, : self.max_len]
            seq = self.max_len
        pos = torch.arange(seq, device=noisy_ids.device).unsqueeze(0).expand(bsz, -1)
        runtime_features = self._features_for_batch(bsz)
        y = self.tok(noisy_ids) + self.pos(pos)
        if runtime_features is not None:
            row = torch.arange(bsz, device=noisy_ids.device).unsqueeze(1)
            y = (
                y
                + runtime_features[
                    row, noisy_ids.clamp(0, runtime_features.size(1) - 1)
                ]
            )
        if self.kind is not None:
            safe = noisy_ids.clamp(min=0, max=self.kind_lookup.numel() - 1)
            y = y + self.kind(self.kind_lookup[safe])

        if ctx_pad_mask is None:
            pooled = context.mean(dim=1)
        else:
            mask = ctx_pad_mask.logical_not().unsqueeze(-1).to(context.dtype)
            pooled = (context * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        z: torch.Tensor | None
        if self.z_state_mode == "full":
            z = self.z_latent[pos]
            z = z + self.ctx_proj(pooled).unsqueeze(1)
            z = z + self.pos(pos)
        elif self.z_state_mode == "parameter_free":
            z = pooled.unsqueeze(1).expand(-1, seq, -1)
        else:
            z = None
        return {
            "y": y,
            "z": z,
            "self_pad_mask": noisy_ids.eq(pad_id),
            "runtime_symbol_features": runtime_features,
        }

    def transition_step(
        self,
        y: torch.Tensor,
        z: torch.Tensor | None,
        context: torch.Tensor,
        self_pad_mask: torch.Tensor | None,
        ctx_pad_mask: torch.Tensor | None = None,
        *,
        runtime_symbol_features: torch.Tensor | None = None,
        return_attn: bool = False,
        update_mode: str = "as_is",
    ) -> dict[str, torch.Tensor | None]:
        """Apply one canonical shared recurrence transition without mutation."""
        if update_mode not in RECURSIVE_DIAGNOSTIC_UPDATE_MODES:
            raise ValueError(
                f"update_mode={update_mode!r} is not one of "
                f"{RECURSIVE_DIAGNOSTIC_UPDATE_MODES!r}"
            )
        y_before = y
        z_before = z
        if z is None:
            f_in = self.norm(y)
            f_out = self._apply_layers(
                f_in, self._f_layers, self_pad_mask, context, ctx_pad_mask
            )
            assert isinstance(f_out, torch.Tensor)
            f_update = f_out if update_mode == "as_is" else f_out - f_in
            y = y + f_update
            g_in = self.norm(y)
        else:
            f_in = self.norm(z + y)
            f_out = self._apply_layers(
                f_in, self._f_layers, self_pad_mask, context, ctx_pad_mask
            )
            assert isinstance(f_out, torch.Tensor)
            f_update = f_out if update_mode == "as_is" else f_out - f_in
            z = z + f_update
            g_in = self.norm(y + z)

        g_out = self._apply_layers(
            g_in,
            self._g_layers,
            self_pad_mask,
            context,
            ctx_pad_mask,
            return_last_attn=return_attn,
        )
        attn: torch.Tensor | None = None
        if return_attn:
            assert isinstance(g_out, tuple)
            g_out, attn = g_out
        else:
            assert isinstance(g_out, torch.Tensor)
        g_update = g_out if update_mode == "as_is" else g_out - g_in
        y = y + g_update
        hidden = self.norm(y)
        logits = self.lm_head(hidden)
        if runtime_symbol_features is not None:
            logits = logits + torch.einsum(
                "btd,bvd->btv", hidden, runtime_symbol_features
            )
        return {
            "y": y,
            "z": z,
            "y_update": y - y_before,
            "z_update": None if z is None else z - z_before,
            "hidden": hidden,
            "logits": logits,
            "attn": attn,
        }

    def recursive_outputs(
        self,
        noisy_ids: torch.Tensor,
        context: torch.Tensor,
        pad_id: int,
        ctx_pad_mask: torch.Tensor | None = None,
        *,
        return_hidden: bool = False,
        return_attn: bool = False,
        return_step_boundaries: bool = False,
        diagnostics: bool = False,
        diagnostic_update_mode: str = "as_is",
        diagnostic_targets: torch.Tensor | None = None,
        diagnostic_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        Run the full recursive recurrence and expose per-depth outputs.

        Returns a dict with:
          - ``logits``: final logits [B, T, V]
          - ``hidden``: final hidden [B, T, D] (only if ``return_hidden=True``)
          - ``depth_hiddens``: list of [B, T, D] for each recursion step
          - ``depth_logits``: list of [B, T, V] for each recursion step
          - ``attn``: last-layer self-attention [B, T, T] (only if ``return_attn=True``)
          - ``step_boundaries``: (SLM-241/RSC-A05 arm H test support, only if
            ``return_step_boundaries=True``) a list of ``{"step": r, "y": ...,
            "z": ...}`` dicts, one per recursion step, holding the *real*
            graph-connected ``y``/``z`` tensors exactly as computed at the end
            of that step -- captured **before** any ``detach_between_steps``
            detach is applied, so a test can register an autograd hook on
            them to observe whether gradient from a later step's loss
            actually reaches that point (arm B/G: yes; arm H: no, once the
            detach has replaced what the *next* step actually consumes).
            Never used by ``forward``/``encode`` -- opt-in and additive only.
          - ``diagnostics``: (only if ``diagnostics=True``) detached
            :class:`RecursiveDepthDiagnosticsV1` records. ``"as_is"`` uses
            the historical residualized block output as the outer update.
            Fixture-only ``"residual_delta"`` instead subtracts each layer
            stack's input from its output before the existing outer addition.
            Targets enable per-example masked CE, accuracy, entropy, and KL
            curves; an omitted mask selects every non-pad target.
        """
        if not isinstance(diagnostics, bool):
            raise TypeError("diagnostics must be a bool")
        if diagnostic_update_mode not in RECURSIVE_DIAGNOSTIC_UPDATE_MODES:
            raise ValueError(
                f"diagnostic_update_mode={diagnostic_update_mode!r} is not one of "
                f"{RECURSIVE_DIAGNOSTIC_UPDATE_MODES!r}"
            )
        if not diagnostics and (
            diagnostic_update_mode != "as_is"
            or diagnostic_targets is not None
            or diagnostic_mask is not None
        ):
            raise ValueError(
                "diagnostic update modes, targets, and masks require diagnostics=True"
            )
        if diagnostic_mask is not None and diagnostic_targets is None:
            raise ValueError("diagnostic_mask requires diagnostic_targets")
        if diagnostic_targets is not None:
            if diagnostic_targets.shape != noisy_ids.shape:
                raise ValueError(
                    "diagnostic_targets shape must match noisy_ids: "
                    f"{tuple(diagnostic_targets.shape)} != {tuple(noisy_ids.shape)}"
                )
            if diagnostic_targets.dtype != torch.long:
                raise TypeError("diagnostic_targets must have dtype torch.long")
            if diagnostic_targets.device != noisy_ids.device:
                raise ValueError("diagnostic_targets must be on noisy_ids.device")
            vocab_size = self.lm_head.weight.size(0)
            if diagnostic_targets.numel() and (
                diagnostic_targets.min().item() < 0
                or diagnostic_targets.max().item() >= vocab_size
            ):
                raise ValueError(f"diagnostic_targets must be in [0, {vocab_size})")
            if diagnostic_mask is not None:
                if diagnostic_mask.shape != noisy_ids.shape:
                    raise ValueError(
                        "diagnostic_mask shape must match noisy_ids: "
                        f"{tuple(diagnostic_mask.shape)} != {tuple(noisy_ids.shape)}"
                    )
                if diagnostic_mask.dtype != torch.bool:
                    raise TypeError("diagnostic_mask must have dtype torch.bool")
                if diagnostic_mask.device != noisy_ids.device:
                    raise ValueError("diagnostic_mask must be on noisy_ids.device")

        bsz, seq = noisy_ids.shape
        if seq > self.max_len:
            noisy_ids = noisy_ids[:, : self.max_len]
            if diagnostic_targets is not None:
                diagnostic_targets = diagnostic_targets[:, : self.max_len]
            if diagnostic_mask is not None:
                diagnostic_mask = diagnostic_mask[:, : self.max_len]
            seq = self.max_len
        initial = self.initial_transition_state(
            noisy_ids, context, pad_id, ctx_pad_mask
        )
        y = initial["y"]
        z = initial["z"]
        self_pad = initial["self_pad_mask"]
        features = initial["runtime_symbol_features"]
        assert isinstance(y, torch.Tensor)
        assert z is None or isinstance(z, torch.Tensor)
        assert isinstance(self_pad, torch.Tensor)
        assert features is None or isinstance(features, torch.Tensor)

        depth_hiddens: list[torch.Tensor] = []
        depth_logits: list[torch.Tensor] = []
        attn: torch.Tensor | None = None
        step_boundaries: list[dict[str, Any]] = []
        diagnostic_states: list[
            tuple[
                torch.Tensor,
                torch.Tensor | None,
                torch.Tensor,
                torch.Tensor | None,
                torch.Tensor,
                torch.Tensor | None,
            ]
        ] = []

        for r in range(1, self.recursive_steps + 1):
            y_before = y
            z_before = z
            return_last_attn = return_attn and r == self.recursive_steps
            step = self.transition_step(
                y,
                z,
                context,
                self_pad,
                ctx_pad_mask,
                runtime_symbol_features=features,
                return_attn=return_last_attn,
                update_mode=diagnostic_update_mode,
            )
            y = step["y"]
            z = step["z"]
            h = step["hidden"]
            logits = step["logits"]
            assert isinstance(y, torch.Tensor)
            assert z is None or isinstance(z, torch.Tensor)
            assert isinstance(h, torch.Tensor)
            assert isinstance(logits, torch.Tensor)
            if return_last_attn:
                attn = step["attn"]
                assert isinstance(attn, torch.Tensor)
            depth_hiddens.append(h)
            depth_logits.append(logits)

            if diagnostics:
                diagnostic_states.append(
                    (
                        y.detach(),
                        None if z is None else z.detach(),
                        (y - y_before).detach(),
                        None if z is None else (z - z_before).detach(),
                        y_before.detach(),
                        None if z_before is None else z_before.detach(),
                    )
                )

            if return_step_boundaries:
                # Captured before any detach below -- the real, graph-
                # connected tensor at this step's exit point, regardless of
                # detach_between_steps.
                step_boundaries.append({"step": r, "y": y, "z": z})

            # SLM-241 (RSC-A05) arm H: cut the recurrent backward path here,
            # never the forward value. Only between steps -- the final
            # step's y/z must stay attached so the model's actual output
            # loss can still backpropagate into the last step's block
            # application normally.
            if self.detach_between_steps and r < self.recursive_steps:
                y = y.detach()
                if z is not None:
                    z = z.detach()

        final_hidden = depth_hiddens[-1]
        final_logits = depth_logits[-1]

        result: dict[str, Any] = {
            "logits": final_logits,
            "depth_hiddens": depth_hiddens,
            "depth_logits": depth_logits,
        }
        if return_hidden:
            result["hidden"] = final_hidden
        if return_attn and attn is not None:
            result["attn"] = attn
        if return_step_boundaries:
            result["step_boundaries"] = step_boundaries
        if diagnostics:
            mask: torch.Tensor | None = None
            counts: torch.Tensor | None = None
            task_metrics: list[
                tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor | None,
                    torch.Tensor,
                ]
                | None
            ] = [None] * len(depth_logits)
            if diagnostic_targets is not None:
                mask = (
                    diagnostic_targets.ne(pad_id)
                    if diagnostic_mask is None
                    else diagnostic_mask
                )
                counts = mask.sum(dim=1)
                if torch.any(counts == 0):
                    raise ValueError(
                        "diagnostic_mask must select at least one target per example"
                    )
                weights = mask.to(torch.float32)
                denominators = counts.to(torch.float32)
                log_probs = [
                    F.log_softmax(logits.detach().float(), dim=-1)
                    for logits in depth_logits
                ]
                final_log_probs = log_probs[-1]
                for index, current_log_probs in enumerate(log_probs):
                    token_ce = F.nll_loss(
                        current_log_probs.transpose(1, 2),
                        diagnostic_targets,
                        reduction="none",
                    )
                    token_accuracy = current_log_probs.argmax(dim=-1).eq(
                        diagnostic_targets
                    )
                    probabilities = current_log_probs.exp()
                    token_entropy = -(probabilities * current_log_probs).sum(dim=-1)

                    def _masked_mean(values: torch.Tensor) -> torch.Tensor:
                        return (values * weights).sum(dim=1) / denominators

                    def _kl(reference: torch.Tensor) -> torch.Tensor:
                        token_kl = (
                            probabilities * (current_log_probs - reference)
                        ).sum(dim=-1)
                        return _masked_mean(token_kl)

                    next_kl = (
                        None
                        if index + 1 == len(log_probs)
                        else _kl(log_probs[index + 1])
                    )
                    task_metrics[index] = (
                        _masked_mean(token_ce),
                        _masked_mean(token_accuracy.to(torch.float32)),
                        _masked_mean(token_entropy),
                        next_kl,
                        _kl(final_log_probs),
                    )

            records: list[RecursiveDepthDiagnosticsV1] = []

            def _batch_l2(value: torch.Tensor) -> torch.Tensor:
                return value.float().flatten(1).norm(dim=1)

            for index, (
                y_state,
                z_state,
                y_update,
                z_update,
                y_before,
                z_before,
            ) in enumerate(diagnostic_states):
                y_norm = _batch_l2(y_state)
                y_update_norm = _batch_l2(y_update)
                y_before_norm = _batch_l2(y_before)
                z_norm = None if z_state is None else _batch_l2(z_state)
                z_update_norm = None if z_update is None else _batch_l2(z_update)
                z_before_norm = None if z_before is None else _batch_l2(z_before)
                metrics = task_metrics[index]
                records.append(
                    RecursiveDepthDiagnosticsV1(
                        contract_version="RecursiveDepthDiagnosticsV1",
                        step=index + 1,
                        update_mode=diagnostic_update_mode,
                        y=y_state,
                        z=z_state,
                        y_update=y_update,
                        z_update=z_update,
                        y_norm=y_norm,
                        z_norm=z_norm,
                        y_update_norm=y_update_norm,
                        z_update_norm=z_update_norm,
                        y_update_state_ratio=y_update_norm
                        / y_before_norm.clamp_min(torch.finfo(torch.float32).eps),
                        z_update_state_ratio=(
                            None
                            if z_update_norm is None or z_before_norm is None
                            else z_update_norm
                            / z_before_norm.clamp_min(torch.finfo(torch.float32).eps)
                        ),
                        target_count=None if counts is None else counts.detach(),
                        cross_entropy=None if metrics is None else metrics[0],
                        accuracy=None if metrics is None else metrics[1],
                        entropy=None if metrics is None else metrics[2],
                        kl_to_next=None if metrics is None else metrics[3],
                        kl_to_final=None if metrics is None else metrics[4],
                    )
                )
            result["diagnostics"] = records
        return result

    def encode(
        self,
        noisy_ids: torch.Tensor,
        context: torch.Tensor,
        pad_id: int,
        ctx_pad_mask: torch.Tensor | None = None,
        *,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode a noisy canvas without paying for the vocabulary projection."""
        out = self.recursive_outputs(
            noisy_ids,
            context,
            pad_id,
            ctx_pad_mask,
            return_hidden=True,
            return_attn=return_attn,
        )
        hidden = out["hidden"]
        assert isinstance(hidden, torch.Tensor)
        if return_attn:
            attn = out["attn"]
            assert isinstance(attn, torch.Tensor)
            return hidden, attn
        return hidden

    def project(
        self,
        hidden: torch.Tensor,
        candidate_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project hidden states to the full vocabulary or gathered candidates."""
        if self._runtime_symbol_features is not None and hidden.dim() != 3:
            if int(self._runtime_symbol_features.size(0)) != 1:
                raise ValueError(
                    "runtime symbol features require [B,T,D] hidden states "
                    "when more than one request is active"
                )
            flat = hidden.reshape(1, -1, hidden.size(-1))
            out = self.project(flat, candidate_ids)
            return out.reshape(*hidden.shape[:-1], out.size(-1))
        features = self._features_for_batch(hidden.size(0))
        if candidate_ids is None:
            logits = self.lm_head(hidden)
            if features is not None:
                logits = logits + torch.einsum("btd,bvd->btv", hidden, features)
            return logits
        raw_weight = self.lm_head.weight
        weight = raw_weight() if callable(raw_weight) else raw_weight
        if weight.is_quantized:
            weight = weight.dequantize()
        weight = weight.index_select(0, candidate_ids)
        logits = F.linear(hidden, weight)
        if features is not None:
            selected = features.index_select(1, candidate_ids)
            logits = logits + torch.einsum("btd,bkd->btk", hidden, selected)
        return logits

    def forward(
        self,
        noisy_ids: torch.Tensor,
        context: torch.Tensor,
        pad_id: int,
        ctx_pad_mask: torch.Tensor | None = None,
        *,
        return_hidden: bool = False,
        return_attn: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Run the full-vocabulary path with the same returns as ``DenoiserTower``."""
        out = self.recursive_outputs(
            noisy_ids,
            context,
            pad_id,
            ctx_pad_mask,
            return_hidden=return_hidden or return_attn,
            return_attn=return_attn,
        )
        logits = out["logits"]
        assert isinstance(logits, torch.Tensor)
        if return_attn:
            attn = out["attn"]
            assert isinstance(attn, torch.Tensor)
            hidden = out["hidden"]
            assert isinstance(hidden, torch.Tensor)
            return logits, hidden, attn
        if return_hidden:
            hidden = out["hidden"]
            assert isinstance(hidden, torch.Tensor)
            return logits, hidden
        return logits


class StackedMatchedStateDenoiserTower(DenoiserTower):
    """SLM-241 (RSC-A05) arm E -- stacked baseline + matched state capacity.

    Subclasses :class:`~slm_training.models.blocks.DenoiserTower` unmodified
    (same ``tok``/``pos``/``kind``/``layers``/``norm``/``lm_head``
    construction, same ``n_layers`` unshared ``TransformerBlock`` instances
    called exactly once each per forward -- **no recurrence, no weight
    sharing added**) and adds exactly two new parameter tensors, shape-matched
    to :class:`SharedRecursiveDenoiserTower`'s (arm B) z-state path so the
    added parameter budget is identical by construction, not by coincidence:

    * ``state``: a learned ``[max_len, d_model]`` bank, indexed by target
      position -- same shape as arm B's ``z_latent``.
    * ``state_ctx_proj``: a ``Linear(d_model, d_model)`` projection of the
      (masked-)mean-pooled context -- same shape as arm B's ``ctx_proj``.

    ``recursive_zstate_parameter_delta(d_model, max_len)`` (below) is exactly
    ``state.numel() + state_ctx_proj.weight.numel() +
    state_ctx_proj.bias.numel()``, so this tower's total parameter count
    equals a same-``n_layers`` ``DenoiserTower`` (arm A) plus that formula's
    value, exactly -- never merely close.

    Both new tensors are injected into the initial hidden state **exactly
    once**, before the first transition block runs::

        y_0 = token + position + kind (+ symbol features)
        y_0 = y_0 + state[position] + state_ctx_proj(mean_pool(context))
        for block in layers:               # n_layers blocks, each called once
            y = block(y, context)          # no recurrence, no re-injection

    This is the mirror image of arm D (``z_state_mode="parameter_free"``):
    D keeps the recursive y/z split but strips the z-state parameters; E
    keeps the stacked baseline's unshared, single-pass block structure but
    adds the z-state parameters back, consumed once rather than every
    recursion step. Because the injection happens once and both tensors sit
    on the residual stream feeding every downstream block, they are not dead
    padding: zeroing ``state`` changes the forward output (verified in
    ``tests/test_models/test_recursive_denoiser.py``), and both receive
    nonzero gradient from a real backward pass.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        max_len: int = 512,
        dropout: float = 0.0,
        *,
        kind_ids: list[int] | None = None,
        n_kinds: int = 0,
    ) -> None:
        super().__init__(
            vocab_size,
            d_model,
            n_layers,
            n_heads,
            max_len,
            dropout,
            kind_ids=kind_ids,
            n_kinds=n_kinds,
        )
        self.state = nn.Parameter(torch.zeros(max_len, d_model))
        self.state_ctx_proj = nn.Linear(d_model, d_model)

    def encode(
        self,
        noisy_ids: torch.Tensor,
        context: torch.Tensor,
        pad_id: int,
        ctx_pad_mask: torch.Tensor | None = None,
        *,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Same as ``DenoiserTower.encode`` plus one-time matched-state
        injection before the (unshared, single-pass) transition blocks run."""
        bsz, seq = noisy_ids.shape
        if seq > self.max_len:
            noisy_ids = noisy_ids[:, : self.max_len]
            seq = self.max_len
        pos = torch.arange(seq, device=noisy_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.tok(noisy_ids) + self.pos(pos)
        features = self._features_for_batch(bsz)
        if features is not None:
            row = torch.arange(bsz, device=noisy_ids.device).unsqueeze(1)
            x = x + features[row, noisy_ids.clamp(0, features.size(1) - 1)]
        if self.kind is not None:
            safe = noisy_ids.clamp(min=0, max=self.kind_lookup.numel() - 1)
            x = x + self.kind(self.kind_lookup[safe])

        # SLM-241 (RSC-A05) arm E: inject the matched-state capacity exactly
        # once here -- never inside the block loop below, never re-applied
        # per layer. Pooling matches SharedRecursiveDenoiserTower's z-state
        # pooling exactly (masked mean over context when a pad mask is given).
        if ctx_pad_mask is None:
            pooled = context.mean(dim=1)
        else:
            mask = ctx_pad_mask.logical_not().unsqueeze(-1).to(context.dtype)
            pooled = (context * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        x = x + self.state[pos] + self.state_ctx_proj(pooled).unsqueeze(1)

        self_pad = noisy_ids.eq(pad_id)
        attn: torch.Tensor | None = None
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            if return_attn and i == last:
                out = layer(
                    x,
                    self_pad_mask=self_pad,
                    ctx=context,
                    ctx_pad_mask=ctx_pad_mask,
                    return_self_attn=True,
                )
                assert isinstance(out, tuple)
                x, attn = out
            else:
                out = layer(
                    x, self_pad_mask=self_pad, ctx=context, ctx_pad_mask=ctx_pad_mask
                )
                x = out if not isinstance(out, tuple) else out[0]
        hidden = self.norm(x)
        if return_attn:
            assert attn is not None
            return hidden, attn
        return hidden


# ---------------------------------------------------------------------------
# SLM-240 (RSC-A04): explicit multi-dimensional architecture comparison.
#
# Replaces the retracted single "same parameter count / layer names" claim in
# the module docstring above with independently named, independently
# falsifiable report fields. No dimension here implies any other -- a caller
# must not infer output/behavioral equivalence from interface compatibility,
# or a parameter-efficiency claim from a parameter-count comparison. There is
# deliberately no single ``parity`` field.
# ---------------------------------------------------------------------------

ARCHITECTURE_COMPARISON_REPORT_VERSION = "ArchitectureComparisonReportV1"

#: Public attributes/methods ``DenoiserTower`` and ``SharedRecursiveDenoiserTower``
#: both expose -- the "public contract" the module docstring's interface claim
#: refers to. Presence/shape-matchable, never behavioral.
_DENOISER_INTERFACE_MEMBERS = (
    "forward",
    "encode",
    "project",
    "set_runtime_symbol_features",
    "tok",
    "kind",
    "lm_head",
    "max_len",
    "layers",
)


def recursive_zstate_parameter_delta(*, d_model: int, max_len: int) -> int:
    """Exact parameter-count delta the V1 z-state path adds over ``DenoiserTower``.

    ``z_latent`` is a ``[max_len, d_model]`` free parameter and ``ctx_proj``
    is a ``Linear(d_model, d_model)`` (``d_model**2`` weight entries plus
    ``d_model`` bias entries). When ``recursive_transition_layers`` equals the
    stacked model's ``n_layers``, the shared transition blocks are a 1:1
    name/shape match against ``DenoiserTower.layers`` and contribute zero to
    this delta -- it is independent of ``vocab_size``, ``n_layers``, and
    ``recursive_steps`` (shared transition parameters do not scale with
    recursion depth). ``tests/test_models/test_recursive_denoiser.py`` checks
    this formula against real constructed towers across several
    ``(d_model, max_len)`` pairs, including the SLM-138 fixture's own
    ``d_model=32, max_len=256`` (delta 9,248) -- the number is reproduced from
    this formula, never hard-coded.

    SLM-241 (RSC-A05) follow-up: :class:`StackedMatchedStateDenoiserTower`
    (arm E)'s ``state``/``state_ctx_proj`` tensors are shape-matched to this
    exact formula too, so its parameter delta over a same-``n_layers`` arm A
    also equals this function's return value exactly, not merely to within a
    tolerance.
    """
    z_latent = int(max_len) * int(d_model)
    ctx_proj = int(d_model) * int(d_model) + int(d_model)
    return z_latent + ctx_proj


def _estimate_transformer_block_flops(
    *, seq_len: int, ctx_len: int, d_model: int, mlp_ratio: float = 4.0
) -> float:
    """Rough analytic FLOPs estimate for one cross-attention ``TransformerBlock``
    forward pass.

    A proxy for *relative* per-block cost comparison only (linear-projection
    terms dominate at fixture scale) -- not a profiler measurement, and never
    to be cited as an absolute latency/throughput claim.
    """
    self_attn = 4.0 * seq_len * d_model * d_model + 2.0 * seq_len * seq_len * d_model
    cross_attn = 4.0 * seq_len * d_model * d_model + 4.0 * seq_len * ctx_len * d_model
    hidden = d_model * mlp_ratio
    mlp = 4.0 * seq_len * d_model * hidden
    return self_attn + cross_attn + mlp


def estimate_transformer_block_flops(
    *, seq_len: int, ctx_len: int, d_model: int, mlp_ratio: float = 4.0
) -> float:
    """Public alias of :func:`_estimate_transformer_block_flops` -- SLM-241
    (RSC-A05)'s per-arm resource accounting reuses this exact estimator."""
    return _estimate_transformer_block_flops(
        seq_len=seq_len, ctx_len=ctx_len, d_model=d_model, mlp_ratio=mlp_ratio
    )


def checkpoint_state_dict_bytes(module: nn.Module) -> int:
    """Public helper: serialized ``state_dict()`` byte size of ``module``.

    Extracted from :func:`compare_denoiser_architectures`'s inline
    ``_checkpoint_bytes`` closure so SLM-241 (RSC-A05)'s per-arm resource
    accounting (:mod:`slm_training.models.recursive_control_arms`) reuses the
    identical measurement instead of a parallel implementation.
    """
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return len(buf.getvalue())


@dataclass(frozen=True)
class ArchitectureComparisonReportV1:
    """SLM-240 (RSC-A04): the required multi-dimensional comparison schema.

    Each field measures one independent claim dimension between a stacked
    ``DenoiserTower`` and a ``SharedRecursiveDenoiserTower``. Built only via
    :func:`compare_denoiser_architectures` (never hand-assembled), which
    derives every value from real constructed modules / a real forward pass
    -- nothing here is a hard-coded literal.
    """

    contract_version: str
    claim_class: str
    d_model: int
    max_len: int
    recursive_steps: int
    recursive_transition_layers: int
    interface_compatible: bool
    output_shape_compatible: bool
    parameter_count_total: dict[str, int]
    parameter_count_denoiser: dict[str, int]
    active_parameter_count: dict[str, int]
    checkpoint_bytes: dict[str, int]
    common_parameter_names_and_shapes: dict[str, list[int]]
    architecture_specific_parameter_names_and_shapes: dict[str, dict[str, list[int]]]
    parameter_count_delta: int
    parameter_count_delta_pct: float
    parameter_count_delta_matches_formula: bool
    block_evaluations_per_forward: dict[str, int]
    estimated_forward_flops: dict[str, float]
    behaviorally_equivalent_under_declared_degeneracy: bool

    def __post_init__(self) -> None:
        if self.contract_version != ARCHITECTURE_COMPARISON_REPORT_VERSION:
            raise ValueError(
                f"contract_version={self.contract_version!r} does not match "
                f"{ARCHITECTURE_COMPARISON_REPORT_VERSION!r}."
            )
        if self.claim_class != "wiring":
            raise ValueError(
                "ArchitectureComparisonReportV1 is wiring-only evidence; "
                f"claim_class={self.claim_class!r} is not a defined value "
                "(no quality/perf claim is ever encoded here)."
            )
        expected_delta = (
            self.parameter_count_total["recursive"]
            - self.parameter_count_total["stacked"]
        )
        if expected_delta != self.parameter_count_delta:
            raise ValueError(
                "parameter_count_delta "
                f"({self.parameter_count_delta}) does not match "
                f"parameter_count_total['recursive'] - "
                f"parameter_count_total['stacked'] ({expected_delta})."
            )
        formula_delta = recursive_zstate_parameter_delta(
            d_model=self.d_model, max_len=self.max_len
        )
        formula_matches = formula_delta == self.parameter_count_delta
        if formula_matches != self.parameter_count_delta_matches_formula:
            raise ValueError(
                "parameter_count_delta_matches_formula "
                f"({self.parameter_count_delta_matches_formula}) does not "
                f"reflect the actual formula comparison (formula={formula_delta}, "
                f"measured={self.parameter_count_delta})."
            )
        if "parity" in self.as_dict():
            raise ValueError(
                "ArchitectureComparisonReportV1 must never carry a field "
                "named 'parity' -- each comparison dimension is independent."
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _active_parameter_count(module: nn.Module, output: torch.Tensor) -> int:
    """Count parameter *elements* that receive a nonzero gradient from
    ``output`` -- i.e. actually touched by one concrete forward pass, as
    opposed to the full declared tensor sizes. For embedding-like parameters
    (``tok``/``pos``/``z_latent``) indexed by a short input sequence, only
    the indexed rows are active; the rest of the declared table is excluded.
    """
    for p in module.parameters():
        p.grad = None
    output.float().sum().backward()
    active = 0
    for p in module.parameters():
        if p.grad is not None:
            active += int((p.grad != 0).sum().item())
        p.grad = None
    return active


#: Public alias -- SLM-241 (RSC-A05)'s per-arm resource accounting
#: (:mod:`slm_training.models.recursive_control_arms`) reuses this exact
#: "elements that receive nonzero gradient from one concrete forward"
#: measurement instead of reimplementing it.
active_parameter_count = _active_parameter_count


def compare_denoiser_architectures(
    stacked: DenoiserTower,
    recursive: SharedRecursiveDenoiserTower,
    *,
    noisy_ids: torch.Tensor,
    context: torch.Tensor,
    pad_id: int,
) -> ArchitectureComparisonReportV1:
    """Build a real, measured :class:`ArchitectureComparisonReportV1`.

    ``stacked`` and ``recursive`` must share ``vocab_size``/``d_model``/
    ``max_len`` (the SLM-138 fixture configuration does). Runs one real
    forward pass on each tower with the given synthetic batch to measure
    ``output_shape_compatible`` and
    ``behaviorally_equivalent_under_declared_degeneracy`` -- neither is
    assumed from configuration alone.
    """
    interface_compatible = all(
        hasattr(stacked, member) and hasattr(recursive, member)
        for member in _DENOISER_INTERFACE_MEMBERS
    )

    with torch.no_grad():
        stacked_logits = stacked(noisy_ids, context, pad_id)
        recursive_logits = recursive(noisy_ids, context, pad_id)
    output_shape_compatible = stacked_logits.shape == recursive_logits.shape
    behaviorally_equivalent = bool(
        output_shape_compatible and torch.allclose(stacked_logits, recursive_logits)
    )

    # Canonical (deduplicated-by-identity) declared parameters. Both towers'
    # ``layers`` transition blocks are also aliased under ``_f_layers``/
    # ``_g_layers`` (slicing an ``nn.ModuleList`` returns a *new* ModuleList
    # wrapping the same block objects, which registers a second name for the
    # same tensors); the default ``named_parameters()`` dedup -- by tensor
    # identity, keeping the first-registered name -- collapses that back to
    # one canonical ``layers.N....`` entry per shared block, and likewise
    # collapses tied ``lm_head.weight``/``tok.weight`` to one name. This is
    # the *correct* declared-parameter view; ``remove_duplicate=False`` would
    # double-count the shared transition blocks as if they were distinct
    # architecture-specific parameters, which they are not.
    stacked_named = dict(stacked.named_parameters())
    recursive_named = dict(recursive.named_parameters())

    common_names = sorted(
        name
        for name in (set(stacked_named) & set(recursive_named))
        if stacked_named[name].shape == recursive_named[name].shape
    )
    common = {name: list(stacked_named[name].shape) for name in common_names}
    stacked_only = {
        name: list(t.shape)
        for name, t in stacked_named.items()
        if name not in common_names
    }
    recursive_only = {
        name: list(t.shape)
        for name, t in recursive_named.items()
        if name not in common_names
    }

    parameter_count_total = {
        "stacked": int(sum(p.numel() for p in stacked_named.values())),
        "recursive": int(sum(p.numel() for p in recursive_named.values())),
    }
    parameter_count_denoiser = {
        "stacked": int(
            sum(p.numel() for layer in stacked.layers for p in layer.parameters())
        ),
        "recursive": int(
            sum(p.numel() for layer in recursive.layers for p in layer.parameters())
        ),
    }

    # Active parameters: elements that actually receive nonzero gradient from
    # *this concrete batch* -- distinct from ``parameter_count_total``'s full
    # declared tensor sizes. ``tok``/``pos``/``z_latent`` are embedding-style
    # tables indexed by a short input sequence; only the indexed rows are
    # active, the remaining declared rows contribute exactly zero gradient
    # and are excluded here. Requires a fresh (grad-enabled) forward per
    # tower since the shape/equivalence check above ran under ``no_grad``.
    active_parameter_count = {
        "stacked": _active_parameter_count(
            stacked, stacked(noisy_ids, context, pad_id)
        ),
        "recursive": _active_parameter_count(
            recursive, recursive(noisy_ids, context, pad_id)
        ),
    }

    checkpoint_bytes = {
        "stacked": checkpoint_state_dict_bytes(stacked),
        "recursive": checkpoint_state_dict_bytes(recursive),
    }

    delta = parameter_count_total["recursive"] - parameter_count_total["stacked"]
    delta_pct = (
        (delta / parameter_count_total["stacked"]) * 100.0
        if parameter_count_total["stacked"]
        else float("nan")
    )
    formula_delta = recursive_zstate_parameter_delta(
        d_model=recursive.d_model, max_len=recursive.max_len
    )

    block_evaluations_per_forward = {
        "stacked": len(stacked.layers),
        "recursive": recursive.recursive_steps * recursive.recursive_transition_layers,
    }
    seq_len = int(noisy_ids.shape[1])
    ctx_len = int(context.shape[1])
    per_block_flops = _estimate_transformer_block_flops(
        seq_len=seq_len, ctx_len=ctx_len, d_model=recursive.d_model
    )
    estimated_forward_flops = {
        "stacked": per_block_flops * block_evaluations_per_forward["stacked"],
        "recursive": per_block_flops * block_evaluations_per_forward["recursive"],
    }

    return ArchitectureComparisonReportV1(
        contract_version=ARCHITECTURE_COMPARISON_REPORT_VERSION,
        claim_class="wiring",
        d_model=recursive.d_model,
        max_len=recursive.max_len,
        recursive_steps=recursive.recursive_steps,
        recursive_transition_layers=recursive.recursive_transition_layers,
        interface_compatible=interface_compatible,
        output_shape_compatible=output_shape_compatible,
        parameter_count_total=parameter_count_total,
        parameter_count_denoiser=parameter_count_denoiser,
        active_parameter_count=active_parameter_count,
        checkpoint_bytes=checkpoint_bytes,
        common_parameter_names_and_shapes=common,
        architecture_specific_parameter_names_and_shapes={
            "stacked_only": stacked_only,
            "recursive_only": recursive_only,
        },
        parameter_count_delta=delta,
        parameter_count_delta_pct=delta_pct,
        parameter_count_delta_matches_formula=(formula_delta == delta),
        block_evaluations_per_forward=block_evaluations_per_forward,
        estimated_forward_flops=estimated_forward_flops,
        behaviorally_equivalent_under_declared_degeneracy=behaviorally_equivalent,
    )
