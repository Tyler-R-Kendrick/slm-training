"""Training-free parallel unmasking policies for MaskGIT / discrete diffusion.

Adapted from MaskGIT schedules (Chang et al., 2022) plus confidence-threshold /
neighbor-spacing heuristics used in discrete diffusion LLM decode. Not a
faithful reimplementation of a single dLLM paper — see
``docs/design/research-lineage.md``. No auxiliary model — drops into MaskGIT.

V4 (E33): budgeted remask can mix grammar hard-errors, trust-gate scores, and
token entropy — remask, don't replace.

V6 (E50): CoRe-lite context-robust remask ranks known tokens by support drop
under masked-context perturbations (arXiv:2602.04096). Always remask→mask
(T2M / remask-don't-replace), never token-edit.

V7 (E70): LESS-lite mutual-stability signals (arXiv:2606.16908) — top-1
persistence across passes + inter-step Jensen–Shannon divergence. Used both to
gate commits (require persistent argmax) and to rank remask candidates.
See ``docs/design/speculative-denoising.md``.
"""

from __future__ import annotations

import math

import torch


def select_unmask_indices(
    conf: torch.Tensor,
    unknown: torch.Tensor,
    *,
    step: int,
    steps: int,
    mode: str = "adaptive",
    min_spacing: int = 2,
) -> list[int]:
    """
    Choose flat indices to unmask this step.

    conf/unknown: [B, T] (typically B=1 for MaskGIT path).
    Modes:
      - topk: classic MaskGIT ceil(remaining / remaining_steps)
      - confidence: all masked positions above adaptive tau
      - adaptive: confidence with mean-field spacing (skip neighbors)
    """
    flat_conf = conf.view(-1)
    flat_unk = unknown.view(-1)
    remaining = int(flat_unk.sum().item())
    if remaining <= 0:
        return []

    steps_left = max(1, steps - step)
    topk_n = max(1, math.ceil(remaining / steps_left))

    if mode == "topk":
        return flat_conf.topk(min(topk_n, remaining)).indices.tolist()

    masked_conf = flat_conf[flat_unk]
    if masked_conf.numel() == 0:
        return []
    # Rising threshold: early steps more aggressive, later more selective.
    frac = step / max(1, steps - 1)
    q = 0.35 + 0.45 * frac
    try:
        tau = float(torch.quantile(masked_conf.float(), q).item())
    except Exception:  # noqa: BLE001
        tau = float(masked_conf.median().item())
    cand = (flat_conf >= tau) & flat_unk
    idxs = cand.nonzero(as_tuple=False).flatten().tolist()
    if not idxs:
        return flat_conf.topk(min(topk_n, remaining)).indices.tolist()

    if mode == "confidence":
        # Cap to 2x classic topk to bound quality risk.
        if len(idxs) > topk_n * 2:
            scored = sorted(idxs, key=lambda i: float(flat_conf[i]), reverse=True)
            return scored[: topk_n * 2]
        return idxs

    # adaptive: greedy independent set with spacing (mean-field-lite).
    scored = sorted(idxs, key=lambda i: float(flat_conf[i]), reverse=True)
    chosen: list[int] = []
    taken_pos: set[int] = set()
    length = conf.size(-1)
    for flat in scored:
        t = flat % length
        if any(abs(t - p) < min_spacing for p in taken_pos):
            continue
        chosen.append(flat)
        taken_pos.add(t)
        if len(chosen) >= topk_n * 2:
            break
    if not chosen:
        return flat_conf.topk(min(topk_n, remaining)).indices.tolist()
    return chosen


def select_remask_indices(
    conf: torch.Tensor,
    known: torch.Tensor,
    *,
    remask_ratio: float = 0.15,
    protect_bos: bool = True,
) -> list[int]:
    """
    Remask lowest-confidence already-unmasked tokens (GIDD / ReMDM-lite).

    conf: [B, T] confidence of current committed tokens (higher = keep).
    known: [B, T] True where token is currently unmasked / committed.
    """
    if remask_ratio <= 0.0:
        return []
    flat_conf = conf.view(-1)
    flat_known = known.view(-1).clone()
    if protect_bos and flat_known.numel() > 0:
        # Position 0 is BOS on the single-sequence MaskGIT path.
        length = conf.size(-1)
        for b in range(conf.size(0)):
            flat_known[b * length] = False
    eligible = int(flat_known.sum().item())
    if eligible <= 0:
        return []
    k = max(1, int(math.ceil(eligible * float(remask_ratio))))
    k = min(k, eligible)
    # Lowest confidence among known positions.
    scores = flat_conf.clone()
    scores = scores.masked_fill(~flat_known, float("inf"))
    return scores.topk(k, largest=False).indices.tolist()


def select_remask_policy_indices(
    conf: torch.Tensor,
    known: torch.Tensor,
    *,
    remask_ratio: float = 0.15,
    protect_bos: bool = True,
    grammar_positions: list[int] | None = None,
    gate_trust: torch.Tensor | None = None,
    entropy: torch.Tensor | None = None,
    gate_threshold: float = 0.5,
) -> list[int]:
    """
    E33: budgeted remask mixing grammar hard-errors, trust gate, and entropy.

    Priority order for the remask budget:
      1. Grammar hard-error positions (always included if known)
      2. Lowest trust-gate scores beneath ``gate_threshold`` (if provided)
      3. Highest entropy among remaining known tokens (if provided)
      4. Lowest confidence (classic E22 fallback)

    Returns flat indices; size capped by ``ceil(eligible * remask_ratio)``
    (at least the grammar set when larger).
    """
    if remask_ratio <= 0.0 and not grammar_positions:
        return []
    flat_conf = conf.view(-1)
    flat_known = known.view(-1).clone()
    length = conf.size(-1)
    if protect_bos and flat_known.numel() > 0:
        for b in range(conf.size(0)):
            flat_known[b * length] = False
    eligible_idx = flat_known.nonzero(as_tuple=False).flatten().tolist()
    if not eligible_idx and not grammar_positions:
        return []

    chosen: list[int] = []
    seen: set[int] = set()

    def _add(idx: int) -> None:
        i = int(idx)
        if i in seen:
            return
        if i < 0 or i >= flat_known.numel():
            return
        if protect_bos and (i % length) == 0:
            return
        # Grammar remasks are allowed even if already unknown.
        if not bool(flat_known[i].item()) and i not in set(grammar_positions or []):
            return
        seen.add(i)
        chosen.append(i)

    for g in grammar_positions or []:
        _add(g)

    eligible = max(len(eligible_idx), len(chosen))
    if remask_ratio > 0:
        budget = max(1, int(math.ceil(eligible * float(remask_ratio))))
    else:
        budget = len(chosen)
    budget = max(budget, len(chosen))

    # Trust gate: remask low-trust known tokens.
    if gate_trust is not None and len(chosen) < budget:
        flat_gate = gate_trust.view(-1)
        scored = sorted(
            eligible_idx,
            key=lambda i: float(flat_gate[i].item()) if i < flat_gate.numel() else 1.0,
        )
        for i in scored:
            if len(chosen) >= budget:
                break
            trust = float(flat_gate[i].item()) if i < flat_gate.numel() else 1.0
            if trust <= float(gate_threshold):
                _add(i)

    # Entropy: remask high-entropy known tokens.
    if entropy is not None and len(chosen) < budget:
        flat_ent = entropy.view(-1)
        scored = sorted(
            eligible_idx,
            key=lambda i: float(flat_ent[i].item()) if i < flat_ent.numel() else 0.0,
            reverse=True,
        )
        for i in scored:
            if len(chosen) >= budget:
                break
            _add(i)

    # Confidence fallback to fill remaining budget.
    if len(chosen) < budget:
        scores = flat_conf.clone()
        scores = scores.masked_fill(~flat_known, float("inf"))
        for i in chosen:
            if i < scores.numel():
                scores[i] = float("inf")
        need = budget - len(chosen)
        known_count = int(flat_known.sum().item())
        if need > 0 and known_count > 0:
            for idx in scores.topk(
                min(need, known_count), largest=False
            ).indices.tolist():
                _add(idx)

    return chosen


def core_instability_scores(
    probs: torch.Tensor,
    probs_perturbed: torch.Tensor,
    committed_ids: torch.Tensor,
    known: torch.Tensor,
) -> torch.Tensor:
    """
    CoRe-lite: per-position support drop under a perturbed context.

    For each known position, score = p_orig[committed] - p_perturbed[committed].
    Higher scores = more context-brittle → prefer remask.
    """
    # Gather probability of the currently committed token under both distributions.
    gather_ids = committed_ids.unsqueeze(-1).clamp(min=0)
    p0 = probs.gather(-1, gather_ids).squeeze(-1)
    p1 = probs_perturbed.gather(-1, gather_ids).squeeze(-1)
    drop = (p0 - p1).clamp(min=0.0)
    return drop.masked_fill(~known, 0.0)


def select_remask_core_indices(
    conf: torch.Tensor,
    known: torch.Tensor,
    *,
    remask_ratio: float = 0.15,
    protect_bos: bool = True,
    instability: torch.Tensor | None = None,
    grammar_positions: list[int] | None = None,
    gate_trust: torch.Tensor | None = None,
    entropy: torch.Tensor | None = None,
    gate_threshold: float = 0.5,
    combine_policy: bool = False,
) -> list[int]:
    """
    E50: remask highest-instability known tokens (CoRe-lite).

    When ``combine_policy`` is True, grammar/gate/entropy fill first (E33),
    then remaining budget goes to highest CoRe instability (else confidence).
    """
    if remask_ratio <= 0.0 and not grammar_positions:
        return []
    flat_known = known.view(-1).clone()
    length = conf.size(-1)
    if protect_bos and flat_known.numel() > 0:
        for b in range(conf.size(0)):
            flat_known[b * length] = False
    eligible_idx = flat_known.nonzero(as_tuple=False).flatten().tolist()
    if not eligible_idx and not grammar_positions:
        return []

    if combine_policy:
        base = select_remask_policy_indices(
            conf,
            known,
            remask_ratio=remask_ratio,
            protect_bos=protect_bos,
            grammar_positions=grammar_positions,
            gate_trust=gate_trust,
            entropy=entropy,
            gate_threshold=gate_threshold,
        )
    else:
        base = list(grammar_positions or [])

    chosen: list[int] = []
    seen: set[int] = set()

    def _add(idx: int) -> None:
        i = int(idx)
        if i in seen:
            return
        if i < 0 or i >= flat_known.numel():
            return
        if protect_bos and (i % length) == 0:
            return
        if not bool(flat_known[i].item()) and i not in set(grammar_positions or []):
            return
        seen.add(i)
        chosen.append(i)

    for i in base:
        _add(i)

    eligible = max(len(eligible_idx), len(chosen))
    budget = (
        max(1, int(math.ceil(eligible * float(remask_ratio))))
        if remask_ratio > 0
        else len(chosen)
    )
    budget = max(budget, len(chosen))

    if instability is not None and len(chosen) < budget:
        flat_inst = instability.view(-1)
        scored = sorted(
            eligible_idx,
            key=lambda i: float(flat_inst[i].item()) if i < flat_inst.numel() else 0.0,
            reverse=True,
        )
        for i in scored:
            if len(chosen) >= budget:
                break
            _add(i)
    elif len(chosen) < budget:
        # Fall back to lowest confidence when no instability tensor provided.
        for i in select_remask_indices(
            conf, known, remask_ratio=remask_ratio, protect_bos=protect_bos
        ):
            if len(chosen) >= budget:
                break
            _add(i)

    return chosen


class StabilityTracker:
    """
    E70 (LESS-lite): track per-position top-1 persistence and inter-step
    Jensen–Shannon divergence across MaskGIT passes.

    Call :meth:`update` once per model pass with the full ``probs`` tensor
    ([B, T, V]). Positions whose argmax changed reset their persistence
    counter; the JSD measures how much the local distribution moved between
    consecutive passes (high JSD = context-sensitive, unstable prediction).
    """

    def __init__(self, *, jsd_weight: float = 1.0) -> None:
        self.jsd_weight = float(jsd_weight)
        self._prev_probs: torch.Tensor | None = None
        self._prev_argmax: torch.Tensor | None = None
        self.persistence: torch.Tensor | None = None  # [B, T] int
        self.jsd: torch.Tensor | None = None  # [B, T] float
        self.observations: int = 0

    @staticmethod
    def _jsd(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Jensen–Shannon divergence along the last dim (natural log)."""
        p = p.clamp(min=1e-9)
        q = q.clamp(min=1e-9)
        m = 0.5 * (p + q)
        kl_pm = (p * (p / m).log()).sum(dim=-1)
        kl_qm = (q * (q / m).log()).sum(dim=-1)
        return 0.5 * (kl_pm + kl_qm)

    def update(self, probs: torch.Tensor) -> None:
        """probs: [B, T, V] from the current pass."""
        cur_argmax = probs.argmax(dim=-1)
        if self._prev_probs is None or self._prev_probs.shape != probs.shape:
            self.persistence = torch.zeros_like(cur_argmax, dtype=torch.long)
            self.jsd = torch.zeros(
                cur_argmax.shape, dtype=probs.dtype, device=probs.device
            )
        else:
            assert self.persistence is not None and self._prev_argmax is not None
            same = cur_argmax.eq(self._prev_argmax)
            self.persistence = torch.where(
                same,
                self.persistence + 1,
                torch.zeros_like(self.persistence),
            )
            self.jsd = self._jsd(self._prev_probs, probs)
        self._prev_probs = probs.detach()
        self._prev_argmax = cur_argmax
        self.observations += 1

    def instability_scores(self) -> torch.Tensor | None:
        """
        [B, T] score where higher = less stable (prefer remask).

        ``jsd_weight * jsd + (1 - persistence_fraction)`` — JSD is in
        [0, ln 2]; persistence is normalized by the number of comparisons so
        the two terms stay commensurate.
        """
        if self.persistence is None or self.jsd is None:
            return None
        comparisons = max(1, self.observations - 1)
        persistence_frac = self.persistence.float() / float(comparisons)
        return self.jsd_weight * self.jsd.float() + (
            1.0 - persistence_frac.clamp(max=1.0)
        )

    def filter_commit_indices(
        self,
        flat_idx: list[int],
        *,
        length: int,
        min_persistence: int,
    ) -> list[int]:
        """
        E70 commit gate: keep only candidates whose argmax persisted for
        ``min_persistence`` consecutive passes. Early passes (fewer
        comparisons than required) are exempt so decode always progresses,
        and an empty result falls back to the original candidates.
        """
        if min_persistence <= 0 or self.persistence is None:
            return flat_idx
        comparisons = self.observations - 1
        if comparisons < min_persistence:
            return flat_idx
        flat_pers = self.persistence.view(-1)
        kept = [
            i
            for i in flat_idx
            if 0 <= i < flat_pers.numel()
            and int(flat_pers[i].item()) >= min_persistence
        ]
        return kept if kept else flat_idx


def select_remask_stability_indices(
    conf: torch.Tensor,
    known: torch.Tensor,
    *,
    remask_ratio: float = 0.15,
    protect_bos: bool = True,
    instability: torch.Tensor | None = None,
    grammar_positions: list[int] | None = None,
    gate_trust: torch.Tensor | None = None,
    entropy: torch.Tensor | None = None,
    gate_threshold: float = 0.5,
    combine_policy: bool = False,
) -> list[int]:
    """
    E70: remask committed tokens ranked by LESS-lite instability
    (low persistence + high inter-step JSD). Thin delegation onto the
    instability-ranked budget machinery shared with CoRe (E50): grammar
    hard errors always fill first; ``combine_policy`` mixes in gate/entropy.
    """
    return select_remask_core_indices(
        conf,
        known,
        remask_ratio=remask_ratio,
        protect_bos=protect_bos,
        instability=instability,
        grammar_positions=grammar_positions,
        gate_trust=gate_trust,
        entropy=entropy,
        gate_threshold=gate_threshold,
        combine_policy=combine_policy,
    )


def perturb_known_neighbors(
    ids: torch.Tensor,
    known: torch.Tensor,
    *,
    mask_id: int,
    perturb_frac: float = 0.25,
    protect_bos: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Build a CoRe-style context perturbation: randomly remask a fraction of
    known (non-BOS) neighbors so a second forward measures support stability.
    """
    out = ids.clone()
    if perturb_frac <= 0.0:
        return out
    flat_known = known.view(-1).clone()
    length = ids.size(-1)
    if protect_bos:
        for b in range(ids.size(0)):
            flat_known[b * length] = False
    eligible = flat_known.nonzero(as_tuple=False).flatten()
    if eligible.numel() == 0:
        return out
    k = max(1, int(math.ceil(float(eligible.numel()) * float(perturb_frac))))
    k = min(k, int(eligible.numel()))
    if generator is not None:
        perm = torch.randperm(eligible.numel(), generator=generator, device=eligible.device)
    else:
        perm = torch.randperm(eligible.numel(), device=eligible.device)
    pick = eligible[perm[:k]]
    flat_out = out.view(-1)
    flat_out[pick] = int(mask_id)
    return out
