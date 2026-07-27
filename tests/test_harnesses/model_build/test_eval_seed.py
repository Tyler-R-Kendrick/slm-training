"""Eval-time RNG seeding for deterministic constrained scoreboards."""

from __future__ import annotations

import random

from slm_training.harnesses.model_build.eval_runner import _seed_eval_rng


def test_seed_eval_rng_is_reproducible() -> None:
    _seed_eval_rng(123)
    first = [random.random() for _ in range(8)]
    _seed_eval_rng(123)
    second = [random.random() for _ in range(8)]
    assert first == second


def test_seed_eval_rng_changes_with_seed() -> None:
    _seed_eval_rng(1)
    a = random.random()
    _seed_eval_rng(2)
    b = random.random()
    assert a != b
