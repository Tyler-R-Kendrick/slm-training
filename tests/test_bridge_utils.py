"""Tests for shared bridge subprocess helpers."""

from __future__ import annotations

from slm_training.bridge_utils import sanitized_node_env


def test_sanitized_node_env_clears_host_node_options(monkeypatch) -> None:
    monkeypatch.setenv("NODE_OPTIONS", "--import tsx")
    env = sanitized_node_env()
    assert env["NODE_OPTIONS"] == ""


def test_sanitized_node_env_preserves_other_variables(monkeypatch) -> None:
    monkeypatch.setenv("NODE_OPTIONS", "--max-old-space-size=8192")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    env = sanitized_node_env()
    assert env["PATH"] == "/usr/bin:/bin"
    assert env["NODE_OPTIONS"] == ""
