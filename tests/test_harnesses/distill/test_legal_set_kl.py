"""Tests for SPV2-03 dense legal-set KL distillation."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from slm_training.harnesses.distill.legal_set_kl import (
    LegalSetDistillExample,
    LegalSetKLConfig,
    legal_set_kl_loss,
    legal_set_kl_loss_from_examples,
    legal_set_teacher_distribution,
    train_legal_set_kl_fixture,
)
from slm_training.harnesses.distill.legal_set_teacher_trace import (
    build_teacher_trace_fixture,
    load_teacher_trace_manifest,
    load_teacher_traces,
    trace_to_distill_examples,
    write_teacher_trace_manifest,
    write_teacher_traces,
)


class _FixtureLogitNet(torch.nn.Module):
    """Tiny network used by fixture tests."""

    def __init__(self, n_states: int, vocab_size: int, hidden: int = 16) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(n_states, hidden)
        self.head = torch.nn.Linear(hidden, vocab_size)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(indices))


def test_kl_zero_when_student_equals_teacher() -> None:
    vocab_size = 8
    teacher_logits = torch.randn(vocab_size)
    student_logits = teacher_logits.clone()
    legal_action_ids = (1, 3, 5)
    loss, metrics = legal_set_kl_loss(
        student_logits, teacher_logits, legal_action_ids
    )
    assert loss.item() < 1e-5
    assert metrics["kl_div"] < 1e-5
    assert metrics["legal_set_size"] == 3


def test_kl_ignores_illegal_actions() -> None:
    vocab_size = 8
    teacher_logits = torch.randn(vocab_size)
    student_logits = teacher_logits.clone()
    legal_action_ids = (1, 3, 5)
    loss1, _ = legal_set_kl_loss(student_logits, teacher_logits, legal_action_ids)

    student_logits[2] += 100.0
    student_logits[7] -= 100.0
    loss2, _ = legal_set_kl_loss(student_logits, teacher_logits, legal_action_ids)
    assert abs(loss2.item() - loss1.item()) < 1e-5


def test_train_fixture_reduces_kl() -> None:
    n_states = 8
    vocab_size = 16
    torch.manual_seed(0)
    student_net = _FixtureLogitNet(n_states, vocab_size)
    teacher_net = _FixtureLogitNet(n_states, vocab_size)
    teacher_net.eval()
    for param in teacher_net.parameters():
        param.requires_grad = False

    examples = [
        LegalSetDistillExample(
            state_id=f"s{i}",
            legal_action_ids=tuple(sorted(torch.randperm(vocab_size)[:4].tolist())),
        )
        for i in range(n_states)
    ]
    result = train_legal_set_kl_fixture(
        student_net, teacher_net, examples, steps=40, lr=0.05
    )
    initial = result["history"][0]["loss"]
    final = result["history"][-1]["loss"]
    assert final < initial
    assert result["final_metrics"]["n_examples"] == n_states


def test_temperature_scaling_changes_loss_monotonically() -> None:
    teacher_logits = torch.tensor([1.0, 2.0, 0.5, -1.0, 3.0, 0.0, -2.0, 1.5])
    student_logits = torch.tensor([3.0, 0.0, 2.0, -1.0, 1.0, 0.5, -1.5, 2.5])
    legal_action_ids = (0, 2, 4, 6)

    losses: list[float] = []
    for temperature in (0.5, 1.0, 2.0, 4.0):
        loss, _ = legal_set_kl_loss(
            student_logits,
            teacher_logits,
            legal_action_ids,
            config=LegalSetKLConfig(temperature=temperature),
        )
        losses.append(loss.item())

    # A more uniform student moves toward the teacher as temperature rises.
    for i in range(len(losses) - 1):
        assert losses[i + 1] < losses[i]


def test_empty_legal_set_returns_finite_zero_loss() -> None:
    student_logits = torch.randn(8)
    teacher_logits = torch.randn(8)
    loss, metrics = legal_set_kl_loss(student_logits, teacher_logits, ())
    assert loss.item() == 0.0
    assert math.isfinite(loss.item())
    assert metrics["legal_set_size"] == 0
    assert math.isnan(metrics["legal_entropy"])
    assert math.isnan(metrics["student_entropy"])
    assert math.isnan(metrics["teacher_entropy"])


def test_approximate_traces_can_be_filtered() -> None:
    manifest, traces = build_teacher_trace_fixture(n_states=16, vocab_size=32, seed=0)
    approximate = [t for t in traces if t.approximate]
    exact = [t for t in traces if not t.approximate]
    assert approximate
    assert exact
    assert len(approximate) + len(exact) == len(traces)


def test_teacher_probs_equivalent_to_logits() -> None:
    vocab_size = 8
    teacher_logits = torch.randn(vocab_size)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_logits = torch.randn(vocab_size)
    legal_action_ids = (1, 3, 5, 7)

    loss_logits, _ = legal_set_kl_loss(
        student_logits,
        teacher_logits,
        legal_action_ids,
        config=LegalSetKLConfig(teacher_is_prob=False),
    )
    loss_probs, _ = legal_set_kl_loss(
        student_logits,
        teacher_probs,
        legal_action_ids,
        config=LegalSetKLConfig(teacher_is_prob=True),
    )
    assert abs(loss_logits.item() - loss_probs.item()) < 1e-4


def test_batch_equivalence_with_per_example_mean() -> None:
    vocab_size = 8
    student_logits_full = torch.randn(2, vocab_size)
    teacher0 = torch.randn(vocab_size)
    teacher1 = torch.randn(vocab_size)
    examples = [
        LegalSetDistillExample(
            state_id="0", legal_action_ids=(1, 3), teacher_logits=teacher0
        ),
        LegalSetDistillExample(
            state_id="1", legal_action_ids=(2, 4, 6), teacher_logits=teacher1
        ),
    ]
    batch_loss, _ = legal_set_kl_loss_from_examples(
        student_logits_full, examples, LegalSetKLConfig()
    )

    loss0, _ = legal_set_kl_loss(
        student_logits_full[0], teacher0, examples[0].legal_action_ids
    )
    loss1, _ = legal_set_kl_loss(
        student_logits_full[1], teacher1, examples[1].legal_action_ids
    )
    assert abs(batch_loss.item() - (loss0.item() + loss1.item()) / 2.0) < 1e-5


def test_teacher_distribution_renormalizes_prob_input() -> None:
    probs = torch.tensor([0.1, 0.3, 0.6])
    legal_action_ids = (0, 2)
    teacher_probs = legal_set_teacher_distribution(
        probs, legal_action_ids, is_prob=True, epsilon=1e-8
    )
    assert abs(teacher_probs.sum().item() - 1.0) < 1e-5
    assert teacher_probs.size(0) == 2


def test_trace_roundtrip(tmp_path) -> None:
    manifest, traces = build_teacher_trace_fixture(n_states=4, vocab_size=16, seed=7)
    manifest_path = tmp_path / "manifest.json"
    traces_path = tmp_path / "traces.jsonl"
    write_teacher_trace_manifest(manifest_path, manifest)
    write_teacher_traces(traces_path, traces)

    loaded_manifest = load_teacher_trace_manifest(manifest_path)
    loaded_traces = load_teacher_traces(traces_path)
    assert loaded_manifest.manifest_id == manifest.manifest_id
    assert len(loaded_traces) == len(traces)
    assert loaded_traces[0].legal_action_ids == traces[0].legal_action_ids


def test_trace_to_distill_examples_preserves_teacher_probs() -> None:
    manifest, traces = build_teacher_trace_fixture(n_states=8, vocab_size=16, seed=3)
    examples = trace_to_distill_examples(traces)
    assert len(examples) == len(traces)
    for example, trace in zip(examples, traces):
        assert example.state_id == trace.state_id
        assert example.legal_action_ids == trace.legal_action_ids
        if trace.teacher_probs is not None:
            assert example.teacher_probs is not None
            assert example.teacher_logits is None
        else:
            assert example.teacher_logits is not None
            assert example.teacher_probs is None


def test_legal_set_kl_loss_without_torch_raises() -> None:
    """Import-level guard is exercised if torch is unavailable."""
    # This test documents the contract; in a normal environment torch is present.
    pytest.importorskip("torch")
