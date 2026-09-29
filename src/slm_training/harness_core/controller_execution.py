"""Explicit controller source/runtime pin, independent of workload authority."""

from pathlib import Path

from slm_training.harness_core.execution_release import runtime_source_identity


def validated_controller_environment(binding):
    from scripts.merge_verification_identity import digest, environment_identity

    if any(not isinstance(binding.get(key), str) or len(binding[key]) != 64
           or any(c not in "0123456789abcdef" for c in binding[key])
           for key in ("source_digest", "runtime_digest")):
        raise ValueError("configured_controller_requires_nonempty_identity_pins")
    root = Path(binding["cwd"]).resolve()
    environment = environment_identity()
    # Import-path relocation belongs to the workload operation's environment
    # fence. The configured controller runtime remains independently pinned.
    runtime_environment = {k: v for k, v in environment.items() if k != "execution_environment_sha256"}
    if (set(binding) != {"cwd", "source_digest", "runtime_digest"}
            or root != Path(__file__).resolve().parents[3]
            or runtime_source_identity(root) != binding["source_digest"]
            or digest(runtime_environment) != binding["runtime_digest"]):
        raise ValueError("configured_controller_source_or_runtime_changed")
    return root, environment


def validate_controller(binding):
    return validated_controller_environment(binding)[0]
