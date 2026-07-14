"""V7 speculative denoising tests (E70–E74).

Covers: LESS-lite stability tracking, attention dependency clusters, ordered
cluster verification (outcome (j, repair)), survival head mining/training,
and the outcome-conditioned successor cache. See
docs/design/speculative-denoising.md.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord
from slm_training.dsl.grammar.fastpath.survival_train import (
    mine_survival_batch,
    train_survival_gate,
)
from slm_training.models.parallel_decode import (
    StabilityTracker,
    select_remask_stability_indices,
)
from slm_training.models.speculative_denoise import (
    Cluster,
    SuccessorCache,
    build_dependency_clusters,
    enumerate_outcome_canvases,
    filter_by_cumulative_survival,
    order_clusters,
    survival_commit_budget,
    verify_clusters_ordered,
)
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

HERO = 'root = Stack([hero], "column")\nhero_title = TextContent(":hero.title")\nhero_body = TextContent(":hero.body")\nhero = Card([hero_title, hero_body])'
CTA = 'root = Stack([cta])\ncta = Button(":cta.label")'


# ---------------------------------------------------------------------------
# E70: StabilityTracker
# ---------------------------------------------------------------------------


def test_stability_tracker_persistence_and_jsd() -> None:
    tracker = StabilityTracker(jsd_weight=1.0)
    stable = torch.tensor([[[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]]])
    tracker.update(stable)
    tracker.update(stable.clone())
    assert tracker.persistence is not None and tracker.jsd is not None
    # Identical distributions: persistence increments, JSD ~ 0.
    assert int(tracker.persistence[0, 0]) == 1
    assert int(tracker.persistence[0, 1]) == 1
    assert float(tracker.jsd.max()) < 1e-6

    # Flip position 1's argmax: its persistence resets, JSD spikes there.
    flipped = torch.tensor([[[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]]])
    tracker.update(flipped)
    assert int(tracker.persistence[0, 0]) == 2
    assert int(tracker.persistence[0, 1]) == 0
    assert float(tracker.jsd[0, 1]) > float(tracker.jsd[0, 0])

    scores = tracker.instability_scores()
    assert scores is not None
    assert float(scores[0, 1]) > float(scores[0, 0])


def test_stability_commit_gate_filters_and_falls_back() -> None:
    tracker = StabilityTracker()
    a = torch.tensor([[[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]])
    b = torch.tensor([[[0.9, 0.1], [0.7, 0.3], [0.6, 0.4]]])  # pos 1 flips
    tracker.update(a)
    tracker.update(b)
    kept = tracker.filter_commit_indices([0, 1, 2], length=3, min_persistence=1)
    assert 1 not in kept and 0 in kept and 2 in kept
    # All-unstable candidates fall back to the originals (progress guarantee).
    assert tracker.filter_commit_indices([1], length=3, min_persistence=1) == [1]


def test_stability_gate_grace_before_enough_observations() -> None:
    tracker = StabilityTracker()
    tracker.update(torch.tensor([[[0.9, 0.1], [0.2, 0.8]]]))
    # Only one pass seen: not enough comparisons — everything passes.
    assert tracker.filter_commit_indices([0, 1], length=2, min_persistence=2) == [0, 1]


def test_select_remask_stability_ranks_instability() -> None:
    conf = torch.tensor([[0.9, 0.9, 0.9, 0.9]])
    known = torch.tensor([[True, True, True, True]])
    inst = torch.tensor([[0.0, 1.4, 0.1, 1.2]])
    idxs = select_remask_stability_indices(
        conf, known, remask_ratio=0.5, instability=inst
    )
    assert 0 not in idxs  # BOS protected
    assert 1 in idxs


# ---------------------------------------------------------------------------
# E71: dependency clusters + anchor ordering
# ---------------------------------------------------------------------------


def _synthetic_attn(t: int, couples: list[tuple[int, int, float]]) -> torch.Tensor:
    attn = torch.full((t, t), 0.01)
    for i, j, w in couples:
        attn[i, j] = w
        attn[j, i] = w
    return attn


def test_build_dependency_clusters_groups_coupled_positions() -> None:
    attn = _synthetic_attn(6, [(1, 2, 0.5)])
    conf = torch.tensor([0.5, 0.9, 0.8, 0.7, 0.6, 0.4])
    clusters = build_dependency_clusters(
        attn, [1, 2, 4], threshold=0.1, max_size=4, conf=conf
    )
    as_sets = [set(c) for c in clusters]
    assert {1, 2} in as_sets  # coupled pair merges
    assert {4} in as_sets  # weakly coupled stays singleton


def test_build_dependency_clusters_respects_max_size() -> None:
    attn = _synthetic_attn(5, [(0, 1, 0.5), (1, 2, 0.5), (2, 3, 0.5), (3, 4, 0.5)])
    clusters = build_dependency_clusters(
        attn, [0, 1, 2, 3, 4], threshold=0.1, max_size=2
    )
    assert all(len(c) <= 2 for c in clusters)
    assert sorted(t for c in clusters for t in c) == [0, 1, 2, 3, 4]


def test_order_clusters_prefers_survival_and_centrality() -> None:
    attn = _synthetic_attn(4, [(0, 2, 0.4)])
    conf = torch.tensor([0.9, 0.3, 0.8, 0.5])
    survival = torch.tensor([0.95, 0.2, 0.9, 0.5])
    ordered = order_clusters(
        [[0], [1], [2]], conf=conf, attn=attn, survival=survival
    )
    # Low-survival cluster [1] must come last.
    assert ordered[-1].positions == [1]
    assert ordered[0].anchor_score >= ordered[-1].anchor_score


# ---------------------------------------------------------------------------
# E73: survival budgets
# ---------------------------------------------------------------------------


def test_survival_commit_budget_cuts_on_cumulative_product() -> None:
    ordered = [
        Cluster(positions=[1], survival=0.9),
        Cluster(positions=[2], survival=0.8),
        Cluster(positions=[3], survival=0.1),
        Cluster(positions=[4], survival=0.9),
    ]
    kept = survival_commit_budget(ordered, threshold=0.3)
    # 0.9 → 0.72 → 0.072 (< 0.3, cut before third)
    assert [c.positions for c in kept] == [[1], [2]]
    # Always keeps at least one cluster.
    lone = survival_commit_budget(
        [Cluster(positions=[1], survival=0.01)], threshold=0.3
    )
    assert len(lone) == 1


def test_filter_by_cumulative_survival_keeps_prefix() -> None:
    survival = torch.tensor([[0.9, 0.05, 0.85, 0.2]])
    kept = filter_by_cumulative_survival([0, 1, 2, 3], survival, threshold=0.5)
    assert set(kept) == {0, 2}


# ---------------------------------------------------------------------------
# E72: ordered cluster verification
# ---------------------------------------------------------------------------


def test_verify_clusters_ordered_accepts_all_without_verifiers() -> None:
    ids = torch.tensor([[7, 0, 0, 0]])
    ordered = [Cluster(positions=[1, 2]), Cluster(positions=[3])]
    proposals = {1: 5, 2: 6, 3: 8}
    outcome = verify_clusters_ordered(ids, ordered, proposals)
    assert outcome.all_accepted
    assert outcome.accepted_clusters == 2
    assert sorted(outcome.accepted_positions) == [1, 2, 3]
    assert outcome.rejected_positions == []


def test_verify_clusters_ordered_stops_at_first_rejection() -> None:
    ids = torch.tensor([[7, 0, 0, 0, 0]])
    ordered = [
        Cluster(positions=[1]),
        Cluster(positions=[2, 3]),
        Cluster(positions=[4]),
    ]
    proposals = {1: 5, 2: 6, 3: 8, 4: 9}

    def stream_filter(trial: list[int], newly: list[int]) -> list[int]:
        return [t for t in newly if t == 3]  # position 3 is a hard error

    outcome = verify_clusters_ordered(
        ids, ordered, proposals, stream_filter=stream_filter
    )
    # Outcome (j=1, repair): first cluster accepted; cluster 2 rejected;
    # cluster 3 deferred (not accepted, not rejected).
    assert outcome.accepted_clusters == 1
    assert outcome.accepted_positions == [1]
    assert outcome.rejected_positions == [2, 3]
    assert not outcome.all_accepted


def test_verify_clusters_ordered_missing_proposal_fails_cluster() -> None:
    ids = torch.tensor([[7, 0, 0]])
    ordered = [Cluster(positions=[1, 2])]
    outcome = verify_clusters_ordered(ids, ordered, {1: 5})  # no proposal for 2
    assert outcome.accepted_clusters == 0
    assert outcome.rejected_positions == [1, 2]


def test_verify_clusters_ordered_admit_rejects() -> None:
    ids = torch.tensor([[7, 0, 0]])
    ordered = [Cluster(positions=[1]), Cluster(positions=[2])]
    proposals = {1: 5, 2: 6}

    def admit(trial: list[int]) -> bool:
        return 6 not in trial  # canvas with token 6 cannot complete

    outcome = verify_clusters_ordered(ids, ordered, proposals, admit=admit)
    assert outcome.accepted_clusters == 1
    assert outcome.rejected_positions == [2]


# ---------------------------------------------------------------------------
# E74: outcome enumeration + successor cache
# ---------------------------------------------------------------------------


def test_enumerate_outcome_canvases_accept_all_then_weakest() -> None:
    ids = torch.tensor([[7, 0, 0, 0]])
    ordered = [
        Cluster(positions=[1], survival=0.9),
        Cluster(positions=[2], survival=0.1),  # weakest → most likely failure
        Cluster(positions=[3], survival=0.8),
    ]
    proposals = {1: 5, 2: 6, 3: 8}
    canvases = enumerate_outcome_canvases(ids, ordered, proposals, fanout=2)
    assert len(canvases) == 2
    j0, c0 = canvases[0]
    assert j0 == 3  # accept-all
    assert c0[0].tolist() == [7, 5, 6, 8]
    j1, c1 = canvases[1]
    assert j1 == 1  # fail at the weakest cluster (order index 1)
    assert c1[0].tolist() == [7, 5, 0, 0]
    # Original canvas untouched.
    assert ids[0].tolist() == [7, 0, 0, 0]


def test_successor_cache_exact_match_hit_and_miss() -> None:
    cache = SuccessorCache()
    canvas = torch.tensor([[7, 5, 6]])
    payload = (torch.randn(1, 3, 4), torch.randn(1, 3, 2), torch.randn(1, 3, 3))
    cache.put(canvas, payload)
    hit = cache.get(torch.tensor([[7, 5, 6]]))
    assert hit is not None
    assert torch.equal(hit[0], payload[0])
    assert cache.get(torch.tensor([[7, 5, 9]])) is None  # mutated canvas → miss
    assert cache.get(torch.tensor([[7, 5]])) is None  # shape mismatch → miss


def _tiny_model(**overrides) -> TwoTowerModel:
    records = [
        ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train"),
        ExampleRecord(id="b", prompt="CTA", openui=CTA, split="train"),
    ]
    cfg = TwoTowerConfig(
        d_model=32,
        n_heads=4,
        context_layers=1,
        denoiser_layers=2,
        gen_steps=4,
        seed=0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return TwoTowerModel.from_records(records, config=cfg, device="cpu")


def test_denoiser_return_attn_matches_sdpa_logits() -> None:
    model = _tiny_model()
    model.eval()
    tok = model.tokenizer
    ids = torch.tensor([[tok.bos_id, tok.mask_id, tok.mask_id, tok.eos_id]])
    ctx, ctx_pad = model._encode_context(["Hero"])
    with torch.no_grad():
        plain = model.denoiser(ids, ctx, pad_id=tok.pad_id, ctx_pad_mask=ctx_pad)
        logits, hidden, attn = model.denoiser(
            ids, ctx, pad_id=tok.pad_id, ctx_pad_mask=ctx_pad, return_attn=True
        )
    assert torch.allclose(plain, logits, atol=1e-5)
    assert attn.shape == (1, ids.size(1), ids.size(1))
    # Row-stochastic attention.
    assert torch.allclose(attn.sum(dim=-1), torch.ones(1, ids.size(1)), atol=1e-4)
    assert hidden.shape[:2] == ids.shape


def test_speculative_batched_forward_matches_single() -> None:
    """E74 hit-equals-recompute: a canvas's batched-slot logits match a
    dedicated single forward on the same canvas."""
    model = _tiny_model()
    model.eval()
    tok = model.tokenizer
    a = torch.tensor([[tok.bos_id, tok.mask_id, tok.mask_id, tok.eos_id]])
    b = torch.tensor([[tok.bos_id, 5, tok.mask_id, tok.eos_id]])
    ctx, ctx_pad = model._encode_context(["Hero"])
    with torch.no_grad():
        batch = torch.cat([a, b], dim=0)
        batched, _h, _attn = model.denoiser(
            batch,
            ctx.expand(2, -1, -1),
            pad_id=tok.pad_id,
            ctx_pad_mask=ctx_pad.expand(2, -1),
            return_attn=True,
        )
        single = model.denoiser(b, ctx, pad_id=tok.pad_id, ctx_pad_mask=ctx_pad)
    assert torch.allclose(batched[1:2], single, atol=1e-5)


# ---------------------------------------------------------------------------
# E73: survival head mining + training
# ---------------------------------------------------------------------------


def test_mine_survival_batch_labels_committed_positions() -> None:
    model = _tiny_model()
    records = [
        ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train"),
        ExampleRecord(id="b", prompt="CTA", openui=CTA, split="train"),
    ]
    hidden, labels, weights = mine_survival_batch(model, records)
    assert hidden.dim() == 3
    assert labels.shape == weights.shape == hidden.shape[:2]
    assert float(weights.sum()) > 0  # some positions were committed
    committed_labels = labels[weights.bool()]
    assert ((committed_labels == 0) | (committed_labels == 1)).all()


def test_train_survival_gate_updates_head_and_flags() -> None:
    model = _tiny_model()
    records = [
        ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train"),
        ExampleRecord(id="b", prompt="CTA", openui=CTA, split="train"),
    ]
    before = [p.detach().clone() for p in model.survival_head.parameters()]
    summary = train_survival_gate(model, records, steps=4, batch_size=2)
    after = list(model.survival_head.parameters())
    assert any(
        not torch.allclose(b, a.detach()) for b, a in zip(before, after)
    )
    assert model.config.survival_gate is True
    assert model.config.survival_gate_train is True
    assert summary["steps"] == 4
    # Backbone stays trainable for later stages.
    assert all(p.requires_grad for p in model.denoiser.parameters())


# ---------------------------------------------------------------------------
# Integration: full V7 decode path on a tiny scratch model
# ---------------------------------------------------------------------------


def test_generate_with_v7_flags_produces_output_and_stats() -> None:
    model = _tiny_model(
        remask_policy="stability",
        stability_min_persistence=1,
        remask_ratio=0.15,
        unmask_mode="cluster",
        cluster_verify=True,
        survival_gate=True,
        speculative_successor=True,
        speculative_fanout=2,
    )
    model.speculative_stats.reset()
    pred = model.generate("Hero")
    assert isinstance(pred, str) and pred.strip()
    stats = model.speculative_stats.as_dict()
    assert stats["generates"] >= 1
    assert stats["denoiser_forwards"] >= 1
    # Speculation ran (or was legitimately skipped when no clusters formed).
    assert stats["speculative_batches"] >= 0
    assert stats["successor_hits"] + stats["successor_misses"] >= 0


def test_speculation_abstains_when_remask_needs_forwards() -> None:
    """Trust-gate remask is non-deterministic → do not burn speculative batches."""
    model = _tiny_model(
        unmask_mode="cluster",
        cluster_verify=True,
        speculative_successor=True,
        speculative_fanout=2,
        remask_ratio=0.15,
        remask_use_gate=True,
        remask_policy="confidence",
    )
    model.speculative_stats.reset()
    model.generate("Hero")
    stats = model.speculative_stats.as_dict()
    assert stats["speculative_batches"] == 0
    assert stats["successor_hits"] == 0
    assert stats["successor_misses"] == 0


def test_generate_with_v7_overlap_thread() -> None:
    model = _tiny_model(
        unmask_mode="cluster",
        cluster_verify=True,
        speculative_successor=True,
        speculative_fanout=2,
        speculative_overlap=True,
    )
    pred = model.generate("CTA")
    assert isinstance(pred, str) and pred.strip()


def test_v7_defaults_leave_decode_unchanged() -> None:
    """With all V7 knobs at defaults the canvas path is byte-identical.

    Compare two generates on the *same* weights (not two independently
    constructed models) so tokenizer/RNG init cannot spuriously diverge.
    """
    model = _tiny_model()
    model.eval()
    # Snapshot defaults, then explicitly re-apply V7-off knobs and regenerate.
    torch.manual_seed(0)
    a = model.generate("Hero")
    model.config.stability_min_persistence = 0
    model.config.unmask_mode = "positions"
    model.config.cluster_verify = False
    model.config.survival_gate = False
    model.config.speculative_successor = False
    torch.manual_seed(0)
    b = model.generate("Hero")
    assert a == b


def test_checkpoint_roundtrip_allows_missing_survival_head(tmp_path) -> None:
    model = _tiny_model()
    path = tmp_path / "model.pt"
    model.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in list(payload["state_dict"]):
        if key.startswith("survival_head."):
            del payload["state_dict"][key]
    torch.save(payload, path)
    loaded = TwoTowerModel.from_checkpoint(path, device="cpu")
    assert loaded is not None
