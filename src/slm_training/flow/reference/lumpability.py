"""Quotient-state lumpability analysis for finite CTMC generators."""

from __future__ import annotations

from typing import Any

import numpy as np

from slm_training.flow.reference.enumerate import StateGraph
from slm_training.flow.reference.generator import Generator

LUMPABLE = "lumpable"
NOT_LUMPABLE = "not_lumpable"
UNKNOWN_NUMERIC = "unknown_numeric"


def _partition_indices(
    graph: StateGraph, partition: dict[str, int]
) -> dict[int, list[int]]:
    """Map block id -> list of state indices."""
    blocks: dict[int, list[int]] = {}
    for i, state in enumerate(graph.states):
        block = partition.get(state.fingerprint, 0)
        blocks.setdefault(block, []).append(i)
    return blocks


def is_strongly_lumpable(
    generator: Generator,
    partition: dict[str, int],
    atol: float = 1e-9,
) -> tuple[bool, dict[str, Any]]:
    """Check strong lumpability: every pair of states in a block has identical
    aggregate outgoing rate to every other block.
    """
    n = generator.n_states
    blocks: dict[int, list[int]] = {}
    for i in range(n):
        state_fp = generator.index_state[i].fingerprint
        block = partition.get(state_fp)
        if block is None:
            return False, {"reason": "partition_missing_state", "state": state_fp}
        blocks.setdefault(block, []).append(i)

    block_ids = sorted(blocks)
    violations: list[dict[str, Any]] = []
    for block in block_ids:
        members = blocks[block]
        # Target block -> expected aggregate rate from first member.
        reference: dict[int, float] = {}
        first = members[0]
        for other_block in block_ids:
            if other_block == block:
                continue
            rate = sum(
                generator.Q[first, j]
                for j in blocks[other_block]
            )
            reference[other_block] = float(rate)
        for member in members[1:]:
            for other_block in block_ids:
                if other_block == block:
                    continue
                rate = sum(generator.Q[member, j] for j in blocks[other_block])
                if abs(rate - reference[other_block]) > atol:
                    violations.append(
                        {
                            "block": block,
                            "member": member,
                            "target_block": other_block,
                            "member_rate": rate,
                            "reference_rate": reference[other_block],
                        }
                    )
    ok = not violations
    return ok, {
        "n_blocks": len(blocks),
        "n_violations": len(violations),
        "violations": violations[:10],
    }


def is_ordinary_lumpable(
    generator: Generator,
    partition: dict[str, int],
    atol: float = 1e-9,
) -> tuple[bool, dict[str, Any]]:
    """Check ordinary lumpability: for every block B and every block C,
    the sum of incoming rates from C to x is identical for all x in B.
    """
    n = generator.n_states
    blocks: dict[int, list[int]] = {}
    for i in range(n):
        state_fp = generator.index_state[i].fingerprint
        block = partition.get(state_fp)
        if block is None:
            return False, {"reason": "partition_missing_state", "state": state_fp}
        blocks.setdefault(block, []).append(i)

    block_ids = sorted(blocks)
    violations: list[dict[str, Any]] = []
    for target_block in block_ids:
        members = blocks[target_block]
        reference: dict[int, float] = {}
        first = members[0]
        for source_block in block_ids:
            rate = sum(generator.Q[i, first] for i in blocks[source_block])
            reference[source_block] = float(rate)
        for member in members[1:]:
            for source_block in block_ids:
                rate = sum(generator.Q[i, member] for i in blocks[source_block])
                if abs(rate - reference[source_block]) > atol:
                    violations.append(
                        {
                            "target_block": target_block,
                            "member": member,
                            "source_block": source_block,
                            "member_rate": rate,
                            "reference_rate": reference[source_block],
                        }
                    )
    ok = not violations
    return ok, {
        "n_blocks": len(blocks),
        "n_violations": len(violations),
        "violations": violations[:10],
    }


def build_quotient_matrix(
    generator: Generator,
    partition: dict[str, int],
) -> np.ndarray:
    """Build the quotient generator assuming strong lumpability holds."""
    n = generator.n_states
    blocks: dict[int, list[int]] = {}
    for i in range(n):
        state_fp = generator.index_state[i].fingerprint
        block = partition.get(state_fp)
        if block is None:
            raise ValueError(f"missing partition for state {state_fp}")
        blocks.setdefault(block, []).append(i)

    block_ids = sorted(blocks)
    block_index = {b: i for i, b in enumerate(block_ids)}
    m = len(block_ids)
    Qq = np.zeros((m, m), dtype=float)
    for b, members in blocks.items():
        i = block_index[b]
        # Any member gives the same aggregate rates by strong lumpability.
        representative = members[0]
        for c, targets in blocks.items():
            j = block_index[c]
            Qq[i, j] = sum(generator.Q[representative, t] for t in targets)
    return Qq


def classify_partition(
    generator: Generator,
    partition: dict[str, int],
    atol: float = 1e-9,
) -> dict[str, Any]:
    """Return ``lumpable``/``not_lumpable``/``unknown_numeric`` with evidence."""
    strong_ok, strong_info = is_strongly_lumpable(generator, partition, atol=atol)
    ordinary_ok, ordinary_info = is_ordinary_lumpable(generator, partition, atol=atol)
    if strong_ok:
        status = LUMPABLE
    elif ordinary_ok:
        status = LUMPABLE
    else:
        status = NOT_LUMPABLE
    if not np.isfinite(generator.Q).all():
        status = UNKNOWN_NUMERIC
    return {
        "status": status,
        "strong_lumpable": strong_ok,
        "ordinary_lumpable": ordinary_ok,
        "strong_details": strong_info,
        "ordinary_details": ordinary_info,
    }
