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
from slm_training.formal.authority import (
    FORMAL_AUTHORITY_SCHEMA,
    FormalAuthorityV2,
    from_formal_object_v1,
    from_formal_preflight_v1,
    load_formal_authority,
    negotiate_schema,
    prefer_write_payload,
    to_formal_object_v1,
    to_formal_preflight_v1,
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
    "FORMAL_AUTHORITY_SCHEMA",
    "FORMAL_OBJECT_SCHEMA",
    "CheckerResult",
    "FormalAuthorityV2",
    "FormalLoopReport",
    "FormalObjectKind",
    "FormalObjectV1",
    "close_formal_loop",
    "export_lean_claim",
    "export_support_certificate",
    "from_formal_object_v1",
    "from_formal_preflight_v1",
    "lean_claim_catalog",
    "load_formal_authority",
    "loop_requires_multi_backend",
    "negotiate_schema",
    "prefer_write_payload",
    "run_checkers",
    "to_formal_object_v1",
    "to_formal_preflight_v1",
]
