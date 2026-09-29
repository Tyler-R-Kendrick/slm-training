"""Compatibility exports for the stdlib-only bounded process lifecycle owner.

Tracking observed descendants is lifecycle ownership, not isolation. Deliberate
namespace/double-fork escapes require the workload isolation backend.
"""

from slm_training.harness_core.bounded_process import (
    OwnedProcessTree as OwnedProcessTree,
    _process_start_identity as identity,
)

__all__ = ["OwnedProcessTree", "identity"]
