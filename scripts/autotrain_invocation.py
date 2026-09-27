"""Host invocation deadline, distinct from explicit operator cancellation."""

from contextlib import contextmanager
import math
import os
import signal
import time

from slm_training.levers import INTERRUPT_AFTER_SECONDS, KILL_GRACE_SECONDS


def invocation_deadline(args):
    limit = args._supervisor_started + INTERRUPT_AFTER_SECONDS
    supplied = getattr(args, "invocation_deadline", None)
    if supplied is not None:
        if not math.isfinite(supplied) or supplied <= 0:
            raise ValueError("invalid parent invocation deadline")
        limit = min(limit, supplied)
    return limit


@contextmanager
def invocation_scope(runtime, args):
    # Only trusted host CLI supplies this monotonic deadline. Signal alone is
    # never evidence of budget expiry. SIGINT/TERM always retain cancel priority.
    runtime.invocation_expired = False
    explicit = [False]
    deadline = invocation_deadline(args)
    runtime.invocation_work_deadline = deadline - 3 * KILL_GRACE_SECONDS
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1)}
    old = os.environ.get("AUTOTRAIN_SUPERVISOR_WORK_DEADLINE")
    os.environ["AUTOTRAIN_SUPERVISOR_WORK_DEADLINE"] = str(deadline - 3 * KILL_GRACE_SECONDS)

    def stop(signum, _frame):
        expiry = (signum == signal.SIGUSR1 and getattr(args, "invocation_deadline", None) is not None
                  and time.monotonic() >= deadline)
        explicit[0] = explicit[0] or not expiry
        runtime.invocation_expired = expiry and not explicit[0]
        runtime.cancel_event.set()

    for s in previous:
        signal.signal(s, stop)
    try:
        yield
    finally:
        if runtime.cancel_event.is_set():
            with runtime._transaction():
                runtime.store.append_event("supervisor_stop_requested", detail={
                    "epoch": runtime.epoch, "deadline": deadline,
                    "cause": "bounded_expiry" if runtime.invocation_expired else "explicit_cancel"})
            if not runtime.invocation_expired:
                runtime.cancel_all(reason="explicit supervisor stop")
        for s, handler in previous.items():
            signal.signal(s, handler)
        if old is None:
            os.environ.pop("AUTOTRAIN_SUPERVISOR_WORK_DEADLINE", None)
        else:
            os.environ["AUTOTRAIN_SUPERVISOR_WORK_DEADLINE"] = old
