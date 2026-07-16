"""V7 speculative denoising: dependency clusters, ordered verification,
outcome-conditioned successor cache.

Adapted (lite) from DAPD/DAWN attention-dependency decoding, Self-Speculative
Masked Diffusions' draft-verify split, DSpark survival scheduling, and
Saguaro speculative-speculative-decoding successor preparation — see
``docs/design/speculative-denoising.md`` and the V7 section of
``docs/design/research-lineage.md``. No draft LM is introduced; the grammar
acceptor remains the verifier.

All functions operate on the single-sequence MaskGIT path (B=1 canvases).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch


# ---------------------------------------------------------------------------
# E71: attention-derived dependency clusters
# ---------------------------------------------------------------------------


@dataclass
class Cluster:
    """A group of strongly-coupled candidate commit positions."""

    positions: list[int]
    anchor_score: float = 0.0
    survival: float = 1.0


def coupling_matrix(attn: torch.Tensor) -> torch.Tensor:
    """
    Symmetric coupling from a head-averaged self-attention map.

    attn: [T, T] (or [1, T, T]) row-stochastic attention.
    Returns [T, T] with c(i, j) = 0.5 * (attn[i, j] + attn[j, i]).
    """
    if attn.dim() == 3:
        attn = attn[0]
    return 0.5 * (attn + attn.transpose(0, 1))


def build_dependency_clusters(
    attn: torch.Tensor,
    candidates: list[int],
    *,
    threshold: float = 0.08,
    max_size: int = 4,
    conf: torch.Tensor | None = None,
    explicit_edges: set[tuple[int, int]] | None = None,
    use_attention: bool = True,
) -> list[list[int]]:
    """
    Greedy dependency clustering of candidate commit positions (DAPD/DAWN-lite).

    Seeds clusters from the highest-confidence unassigned candidate, then
    absorbs unassigned candidates whose symmetric attention coupling to any
    member is >= ``threshold`` (up to ``max_size``). Weakly coupled candidates
    end up as singleton clusters, preserving full parallelism where the model
    reports independence.

    attn: [T, T] or [1, T, T]; candidates: canvas positions (t indices);
    conf: optional [T] (or [1, T]) confidence used for seed order.
    """
    if not candidates:
        return []
    coup = coupling_matrix(attn)
    if conf is not None:
        flat_conf = conf.view(-1)
        seed_order = sorted(
            candidates,
            key=lambda t: float(flat_conf[t].item()) if t < flat_conf.numel() else 0.0,
            reverse=True,
        )
    else:
        seed_order = sorted(candidates)
    unassigned = set(candidates)
    clusters: list[list[int]] = []
    for seed in seed_order:
        if seed not in unassigned:
            continue
        unassigned.discard(seed)
        members = [seed]
        while len(members) < max_size:
            best_pos: int | None = None
            best_coup = float(threshold)
            for t in unassigned:
                linked = explicit_edges is not None and any(
                    (min(t, m), max(t, m)) in explicit_edges for m in members
                )
                c = (
                    max(float(coup[t, m].item()) for m in members)
                    if use_attention
                    else 0.0
                )
                if linked:
                    c = max(c, float(threshold))
                if c >= best_coup:
                    best_coup = c
                    best_pos = t
            if best_pos is None:
                break
            members.append(best_pos)
            unassigned.discard(best_pos)
        clusters.append(sorted(members))
    return clusters


def build_constraint_edges(token_ids: list[int], tokenizer: Any) -> set[tuple[int, int]]:
    """Cheap OpenUI graph: statements, repeated symbols, and delimiter pairs."""
    edges: set[tuple[int, int]] = set()
    try:
        spans = tokenizer.statement_spans(token_ids)
    except Exception:  # noqa: BLE001
        spans = []
    for start, end in spans:
        positions = list(range(start, end))
        for left, right in zip(positions, positions[1:]):
            edges.add((left, right))
    by_symbol: dict[int, list[int]] = {}
    for pos, token_id in enumerate(token_ids):
        try:
            if tokenizer.is_sym_id(token_id) or tokenizer.is_bind_id(token_id) or tokenizer.is_state_id(token_id):
                by_symbol.setdefault(int(token_id), []).append(pos)
        except Exception:  # noqa: BLE001
            continue
    for positions in by_symbol.values():
        for left, right in zip(positions, positions[1:]):
            edges.add((min(left, right), max(left, right)))
    open_to_close = {"(": ")", "[": "]", "{": "}"}
    stacks: dict[str, list[int]] = {key: [] for key in open_to_close}
    for pos, token_id in enumerate(token_ids):
        token = getattr(tokenizer, "id_to_token", {}).get(int(token_id), "")
        if token in stacks:
            stacks[token].append(pos)
            continue
        for opener, closer in open_to_close.items():
            if token == closer and stacks[opener]:
                left = stacks[opener].pop()
                edges.add((left, pos))
                break
    return edges


def order_clusters(
    clusters: list[list[int]],
    *,
    conf: torch.Tensor,
    attn: torch.Tensor | None = None,
    survival: torch.Tensor | None = None,
) -> list[Cluster]:
    """
    Anchor-first temporary order (E71/E72).

    Anchor score per the design doc: ``a(C) = mean_survival(C) *
    (1 + coupling_centrality(C))`` where centrality is the summed coupling
    from other candidate positions into the cluster. Survival falls back to
    confidence when no survival head is provided. High scores verify first;
    ambiguous strongly-coupled clusters verify last.
    """
    if not clusters:
        return []
    flat_conf = conf.view(-1)
    flat_surv = survival.view(-1) if survival is not None else None
    coup = coupling_matrix(attn) if attn is not None else None
    all_positions = {t for c in clusters for t in c}
    out: list[Cluster] = []
    for members in clusters:
        if flat_surv is not None:
            surv_vals = [
                float(flat_surv[t].item()) for t in members if t < flat_surv.numel()
            ]
        else:
            surv_vals = [
                float(flat_conf[t].item()) for t in members if t < flat_conf.numel()
            ]
        mean_surv = sum(surv_vals) / max(1, len(surv_vals))
        centrality = 0.0
        if coup is not None:
            others = all_positions.difference(members)
            for t in others:
                centrality += max(float(coup[t, m].item()) for m in members)
        out.append(
            Cluster(
                positions=sorted(members),
                anchor_score=mean_surv * (1.0 + centrality),
                survival=mean_surv,
            )
        )
    out.sort(key=lambda c: c.anchor_score, reverse=True)
    return out


# ---------------------------------------------------------------------------
# E73: survival-calibrated commit budget
# ---------------------------------------------------------------------------


def survival_commit_budget(
    ordered: list[Cluster],
    *,
    threshold: float = 0.3,
) -> list[Cluster]:
    """
    DSpark-lite cumulative-survival budget: keep the ordered-cluster prefix
    while the running product of cluster survival stays >= ``threshold``.
    Always keeps at least one cluster so decode progresses.
    """
    if not ordered or threshold <= 0.0:
        return list(ordered)
    kept: list[Cluster] = []
    cum = 1.0
    for cluster in ordered:
        cum *= max(0.0, min(1.0, cluster.survival))
        if kept and cum < threshold:
            break
        kept.append(cluster)
    return kept


def filter_by_cumulative_survival(
    flat_idx: list[int],
    survival: torch.Tensor,
    *,
    threshold: float = 0.3,
) -> list[int]:
    """
    Position-level variant for the non-cluster path: order candidates by
    survival (desc) and keep the prefix while the cumulative product stays
    >= ``threshold`` (always at least one).
    """
    if not flat_idx or threshold <= 0.0:
        return flat_idx
    flat_surv = survival.view(-1)
    scored = sorted(
        flat_idx,
        key=lambda i: float(flat_surv[i].item()) if i < flat_surv.numel() else 0.0,
        reverse=True,
    )
    kept: list[int] = []
    cum = 1.0
    for i in scored:
        s = float(flat_surv[i].item()) if i < flat_surv.numel() else 0.0
        cum *= max(0.0, min(1.0, s))
        if kept and cum < threshold:
            break
        kept.append(i)
    return kept


# ---------------------------------------------------------------------------
# E72: ordered cluster verification -> prefix-like outcome (j, repair)
# ---------------------------------------------------------------------------


@dataclass
class VerifyOutcome:
    """Prefix-like verifier outcome under the temporary cluster order."""

    accepted_clusters: int
    total_clusters: int
    accepted_positions: list[int]
    rejected_positions: list[int]

    @property
    def all_accepted(self) -> bool:
        return self.accepted_clusters >= self.total_clusters


def verify_clusters_ordered(
    ids: torch.Tensor,
    ordered: list[Cluster],
    proposals: dict[int, int],
    *,
    admit: Callable[[list[int]], bool] | None = None,
    stream_filter: Callable[[list[int], list[int]], list[int]] | None = None,
) -> VerifyOutcome:
    """
    Verify clusters in the temporary order; stop at the first rejection.

    ids: [1, T] current canvas (not mutated — commits are simulated).
    proposals: position -> proposed token id (positions without a proposal
    fail their cluster: no legal token was found).
    admit: canvas-level completability probe (CFG hole-admit).
    stream_filter: (trial, newly) -> hard-error positions among ``newly``.

    Returns the outcome ``o = (j, repair)``: clusters ``1..j`` accepted (their
    positions in ``accepted_positions``), cluster ``j+1`` rejected (positions
    in ``rejected_positions`` — to be remasked), later clusters deferred.
    """
    trial = ids[0].tolist()
    accepted_positions: list[int] = []
    j = 0
    for cluster in ordered:
        tentative = list(trial)
        newly: list[int] = []
        ok = True
        for t in sorted(cluster.positions):
            tok = proposals.get(t)
            if tok is None:
                ok = False
                break
            tentative[t] = int(tok)
            newly.append(t)
        if ok and admit is not None:
            try:
                ok = bool(admit(tentative))
            except Exception:  # noqa: BLE001
                ok = False
        if ok and stream_filter is not None:
            try:
                bad = stream_filter(tentative, newly)
            except Exception:  # noqa: BLE001
                bad = list(newly)
            ok = not bad
        if not ok:
            return VerifyOutcome(
                accepted_clusters=j,
                total_clusters=len(ordered),
                accepted_positions=accepted_positions,
                rejected_positions=sorted(cluster.positions),
            )
        trial = tentative
        accepted_positions.extend(newly)
        j += 1
    return VerifyOutcome(
        accepted_clusters=j,
        total_clusters=len(ordered),
        accepted_positions=accepted_positions,
        rejected_positions=[],
    )


# ---------------------------------------------------------------------------
# E74: outcome enumeration + successor cache
# ---------------------------------------------------------------------------


def enumerate_outcome_canvases(
    ids: torch.Tensor,
    ordered: list[Cluster],
    proposals: dict[int, int],
    *,
    fanout: int = 2,
    eos_id: int | None = None,
    pad_id: int | None = None,
) -> list[tuple[int, torch.Tensor]]:
    """
    Candidate successor canvases for the top-K likely verifier outcomes.

    Under ordered verification the outcome is a prefix length ``j``; the
    successor canvas commits clusters ``1..j`` and leaves the rest masked.
    Likelihood ranking: accept-all first, then failure at the lowest-survival
    cluster, then the next-lowest, ... Returns [(j, canvas [1, T])], deduped.

    When ``eos_id``/``pad_id`` are provided, the decode loop's deterministic
    EOS post-processing (pad everything after the first EOS) is simulated so
    speculated canvases match the real post-commit canvas exactly.
    """
    if not ordered or fanout <= 0:
        return []
    m = len(ordered)
    js: list[int] = [m]
    for idx in sorted(range(m), key=lambda i: ordered[i].survival):
        if len(js) >= fanout:
            break
        if idx not in js:
            js.append(idx)
    out: list[tuple[int, torch.Tensor]] = []
    seen: set[int] = set()
    for j in js[:fanout]:
        if j in seen:
            continue
        seen.add(j)
        canvas = ids.clone()
        committable = True
        for cluster in ordered[:j]:
            for t in sorted(cluster.positions):
                tok = proposals.get(t)
                if tok is None:
                    committable = False
                    break
                canvas[0, t] = int(tok)
            if not committable:
                break
        if not committable:
            continue
        if eos_id is not None and pad_id is not None:
            eos_positions = (canvas[0] == int(eos_id)).nonzero(as_tuple=False)
            if eos_positions.numel():
                end = int(eos_positions[0].item())
                if end + 1 < canvas.size(1):
                    canvas[0, end + 1 :] = int(pad_id)
        out.append((j, canvas))
    return out


class SuccessorCache:
    """
    Exact-match cache of precomputed next-pass results (Saguaro-SSD-lite).

    Entries map a concrete successor canvas to ``(logits, hidden, attn)``
    from a single batched forward. Lookup compares the full canvas tensor —
    any post-verification mutation (EOS padding, remask policies) that
    diverges from the speculated canvas is an honest miss.
    """

    def __init__(self) -> None:
        self._entries: list[tuple[torch.Tensor, tuple[Any, ...]]] = []

    def __len__(self) -> int:
        return len(self._entries)

    def put(self, ids: torch.Tensor, payload: tuple[Any, ...]) -> None:
        self._entries.append((ids.detach().clone(), payload))

    def get(self, ids: torch.Tensor) -> tuple[Any, ...] | None:
        for canvas, payload in self._entries:
            if canvas.shape == ids.shape and bool(torch.equal(canvas, ids)):
                return payload
        return None

    def clear(self) -> None:
        self._entries.clear()


@dataclass
class SpeculativeStats:
    """Per-model decode telemetry for the V7 MaskGIT path."""

    generates: int = 0
    denoiser_forwards: int = 0
    speculative_batches: int = 0
    speculative_canvases: int = 0
    successor_hits: int = 0
    successor_misses: int = 0
    clusters_proposed: int = 0
    clusters_accepted: int = 0
    clusters_rejected: int = 0
    remasked_positions: int = 0
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        lookups = self.successor_hits + self.successor_misses
        return {
            "generates": self.generates,
            "denoiser_forwards": self.denoiser_forwards,
            "forwards_per_generate": (
                self.denoiser_forwards / self.generates if self.generates else None
            ),
            "speculative_batches": self.speculative_batches,
            "speculative_canvases": self.speculative_canvases,
            "successor_hits": self.successor_hits,
            "successor_misses": self.successor_misses,
            "successor_hit_rate": (
                self.successor_hits / lookups if lookups else None
            ),
            "clusters_proposed": self.clusters_proposed,
            "clusters_accepted": self.clusters_accepted,
            "clusters_rejected": self.clusters_rejected,
            "remasked_positions": self.remasked_positions,
            **self.extra,
        }

    def reset(self) -> None:
        self.generates = 0
        self.denoiser_forwards = 0
        self.speculative_batches = 0
        self.speculative_canvases = 0
        self.successor_hits = 0
        self.successor_misses = 0
        self.clusters_proposed = 0
        self.clusters_accepted = 0
        self.clusters_rejected = 0
        self.remasked_positions = 0
        self.extra.clear()
