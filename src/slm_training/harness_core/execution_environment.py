"""Canonical execution variables used in verification environment identities."""

import os


def verification_environment():
    """Normalize import paths before a controller changes its working directory."""
    return {
        key: os.pathsep.join(map(os.path.abspath, value.split(os.pathsep)))
        if key == "PYTHONPATH" else value
        for key, value in os.environ.items()
        if key in {"PATH", "NODE_OPTIONS", "ORT_DISABLE_TELEMETRY"}
        or key.startswith(("PYTHON", "SLM_", "OMP_", "MKL_", "OPENBLAS_", "NUMEXPR_"))
    }
