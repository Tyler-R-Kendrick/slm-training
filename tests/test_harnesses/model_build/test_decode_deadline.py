"""Cooperative decode deadline hard wall."""

from __future__ import annotations

import time

import pytest

from slm_training.models.decode_stats import (
    check_decode_deadline,
    clear_decode_deadline,
    decode_deadline_remaining,
    set_decode_deadline,
)


def test_check_decode_deadline_raises_after_budget() -> None:
    set_decode_deadline(0.05)
    try:
        time.sleep(0.08)
        with pytest.raises(TimeoutError, match="decode deadline exceeded"):
            check_decode_deadline()
    finally:
        clear_decode_deadline()


def test_check_decode_deadline_noop_when_unarmed() -> None:
    clear_decode_deadline()
    check_decode_deadline()  # must not raise
    assert decode_deadline_remaining() is None


def test_set_decode_deadline_remaining_positive() -> None:
    set_decode_deadline(1.0)
    try:
        remaining = decode_deadline_remaining()
        assert remaining is not None
        assert 0.0 < remaining <= 1.0
        check_decode_deadline()  # not expired yet
    finally:
        clear_decode_deadline()


def test_clear_decode_deadline_disarms() -> None:
    set_decode_deadline(0.01)
    time.sleep(0.03)
    clear_decode_deadline()
    check_decode_deadline()  # cleared — no raise
