"""Exportable formal objects and multi-prover verification loop.

Closes the formal loop for VSS support certificates and Lean claim modules
using **independent checkers** (structural Python + certificate replay +
optional Lean kernel). A single kernel is never sufficient authority.
"""

from slm_training.formal.checkers import (
    CHECKER_LEAN_KERNEL,
    CHECKER_PYTHON_REFERENCE,
    CHECKER_PYTHON_REPLAY,
    CHECKER_PYTHON_STRUCTURAL,
    CheckerResult,
    run_checkers,
)
from slm_training.formal.loop import (
    FormalLoopReport,
    close_formal_loop,
    loop_requires_multi_backend,
)
from slm_training.formal.objects import (
    FORMAL_OBJECT_SCHEMA,
    FormalObjectKind,
    FormalObjectV1,
    export_lean_claim,
    export_support_certificate,
    lean_claim_catalog,
)

__all__ = [
    "CHECKER_LEAN_KERNEL",
    "CHECKER_PYTHON_REFERENCE",
    "CHECKER_PYTHON_REPLAY",
    "CHECKER_PYTHON_STRUCTURAL",
    "FORMAL_OBJECT_SCHEMA",
    "CheckerResult",
    "FormalLoopReport",
    "FormalObjectKind",
    "FormalObjectV1",
    "close_formal_loop",
    "export_lean_claim",
    "export_support_certificate",
    "lean_claim_catalog",
    "loop_requires_multi_backend",
    "run_checkers",
]
