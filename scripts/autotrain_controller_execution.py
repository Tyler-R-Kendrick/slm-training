"""Configured controller code and live journal authority, separate from workload."""

from contextlib import contextmanager
from contextvars import ContextVar

from slm_training.harness_core.controller_execution import validate_controller

_PARENT = ContextVar("controller_execution_parent", default=None)
_HOST_CONFIGURATION = ContextVar("controller_host_configuration", default=None)
CONFIGURATION_KEYS = ("repair_config", "repair_config_digest", "delivery_config", "delivery_config_digest")


def configured_controller(args):
    root, digest, runtime = (getattr(args, name, None) for name in
                             ("controller_root", "controller_source_digest", "controller_runtime_digest"))
    if root is None and digest is None and runtime is None:
        from pathlib import Path

        if Path.cwd().resolve() != Path(__file__).resolve().parents[1]:
            raise ValueError("split_controller_requires_explicit_pins")
        return None
    if root is None or digest is None or runtime is None:
        raise ValueError("controller_requires_explicit_root_source_and_runtime_pins")
    binding = {"cwd": str(root.resolve()), "source_digest": digest, "runtime_digest": runtime}
    return binding


@contextmanager
def controller_authority_scope(runtime, binding, *, host_configuration=None):
    from slm_training.harness_core.controller_execution import validated_controller_environment

    environment = validated_controller_environment(binding)[1] if binding is not None else None
    with runtime._transaction():
        _validate_host_configuration(host_configuration)
        token = _PARENT.set((runtime, binding))
        config_token = _HOST_CONFIGURATION.set(host_configuration)
    try:
        yield environment
    finally:
        _HOST_CONFIGURATION.reset(config_token)
        _PARENT.reset(token)


@contextmanager
def journal_authority(journal):
    """A copied/hash-valid journal cannot substitute for the live issuer's store."""
    from slm_training.harness_core.checkpoint_publication import _SCOPE

    scope = _SCOPE.get()
    if scope is not None:
        if scope.publisher.store.root.resolve() != journal.root.resolve():
            raise ValueError("execution_transition_foreign_journal")
        with scope.publisher.publication(scope.lease) as state:
            yield state
        return
    parent = _PARENT.get()
    if parent is None or parent[0].store.root.resolve() != journal.root.resolve():
        raise ValueError("execution_transition_requires_live_controller")
    with parent[0]._transaction():
        if parent[1] is not None:
            validate_controller(parent[1])
        yield None


def parent_controller():
    parent = _PARENT.get()
    return None if parent is None else parent[1]


def record_launch(runtime, request, lease):
    """Bind configured code to this existing request/lease without changing either."""
    from slm_training.harness_core.activity_contract import contract_digest

    binding = parent_controller() or request.get("controller_execution")
    if binding is None:
        return None
    validate_controller(binding)
    receipt = {"request_digest": contract_digest(request),
               "lease": lease.model_dump(mode="json"), "controller_execution": binding}
    configuration = _HOST_CONFIGURATION.get()
    if configuration is not None:
        _validate_host_configuration(configuration)
        receipt["host_configuration"] = dict(configuration)
    with runtime.publication(lease):
        runtime.store.append_event("operation_controller_launch_verified",
            experiment_id=lease.activity_id, detail=receipt)
    return receipt


def _operation_controller_binding(request):
    """Only a live delegated child can consume a controller launch receipt."""
    from pathlib import Path
    from slm_training.autoresearch.runtime.activity_publication import DelegatedPublisher
    from slm_training.autoresearch.storage import CampaignStore
    from slm_training.harness_core.activity_contract import ActivityLease, contract_digest

    receipt = request.get("controller_launch")
    if receipt is None:
        binding = request.get("controller_execution") or parent_controller()
    else:
        journal = CampaignStore("runtime", Path(request["root"]) / "loops" / request["loop_id"])
        lease = ActivityLease.model_validate(request["lease"])
        logical = {k: v for k, v in request.items() if k not in {"lease", "parent_event", "controller_launch"}}
        with DelegatedPublisher(journal, request["source_digest"]).publication(lease) as state:
            if (receipt["request_digest"] != contract_digest(logical)
                    or state.spec.input_digest != receipt["request_digest"]
                    or receipt["lease"] != lease.model_dump(mode="json")
                    or not any(e["event_type"] == "operation_controller_launch_verified"
                        and e["experiment_id"] == lease.activity_id and e["detail"] == receipt
                        for e in journal.verify_event_chain())):
                raise ValueError("controller_launch_receipt_not_bound_to_live_request")
        binding = receipt["controller_execution"]
        if request.get("controller_execution", binding) != binding:
            raise ValueError("controller_launch_changed_successor_pin")
    _validate_host_configuration(receipt.get("host_configuration") if receipt is not None
                                 else _HOST_CONFIGURATION.get())
    if binding is None:
        if Path(request["cwd"]).resolve() != Path(__file__).resolve().parents[1]:
            raise ValueError("split_controller_requires_explicit_pins")
    return binding


def operation_controller(request):
    binding = _operation_controller_binding(request)
    if binding is not None:
        validate_controller(binding)
    return binding


def operation_environment(request):
    """One fresh environment measurement for adjacent controller/workload checks."""
    from slm_training.harness_core.controller_execution import validated_controller_environment
    from scripts.merge_verification_identity import environment_identity

    binding = _operation_controller_binding(request)
    return (validated_controller_environment(binding)[1]
            if binding is not None else environment_identity())


def configured_host_configuration(args):
    """Pin explicit host options; credentials remain inside their configured files."""
    import hashlib

    values = {}
    for key in ("repair_config", "delivery_config"):
        path = getattr(args, key, None)
        values[key] = str(path.resolve()) if path is not None else None
        values[key + "_digest"] = hashlib.sha256(path.read_bytes()).hexdigest() if path is not None else None
    _validate_host_configuration(values)
    return values


def _validate_host_configuration(values):
    import hashlib
    from pathlib import Path

    if values is None:
        return
    if set(values) != set(CONFIGURATION_KEYS):
        raise ValueError("controller_host_configuration_keys_changed")
    for key in ("repair_config", "delivery_config"):
        path, expected = values[key], values[key + "_digest"]
        if path is None:
            if expected is not None:
                raise ValueError("controller_host_configuration_missing_path")
        elif (not isinstance(path, str) or not Path(path).is_absolute()
              or Path(path).is_symlink() or not Path(path).is_file()
              or hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected):
            raise ValueError("controller_host_configuration_changed")


def authorize_configuration_rebinding(runtime, request):
    """Host config can change execution authority, never the retained driver grant."""
    parent, configured = _PARENT.get(), _HOST_CONFIGURATION.get()
    if (parent is None or parent[0] is not runtime or parent[1] is None
            or configured is None
            or {key: request.get(key) for key in CONFIGURATION_KEYS} != configured):
        raise ValueError("driver_configuration_change_requires_live_host_binding")
    with runtime._transaction():
        _validate_host_configuration(configured)


def operation_host_configuration(request):
    """Authenticated execution view; never alter the logical request or its digest."""
    _operation_controller_binding(request)
    values = (request.get("controller_launch") or {}).get("host_configuration")
    if values is None:
        values = _HOST_CONFIGURATION.get()
    return dict(values) if values is not None else {key: request.get(key) for key in CONFIGURATION_KEYS}
