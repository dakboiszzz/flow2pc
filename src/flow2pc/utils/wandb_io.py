"""Logging-only workarounds for W&B artifact preflight and finalization failures.

No model imports, seeding, sampling or optimizer operations belong here.
"""

import json
import os
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from uuid import uuid4


@contextmanager
def startup_lock():
    """Serialize SDK startup, without holding a lock during training."""
    path = os.environ.get("FLOW2PC_WANDB_INIT_LOCK")
    if not path:
        yield
        return
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def guard_artifact_uploads(run, communication_errors, record):
    """Keep SDK preflight best-effort while retaining the real artifact enqueue.

    W&B 0.30.0 catches WandbApiFailedError in _assert_can_log_artifact,
    but PublicApi authentication wraps that timeout as AuthenticationError
    (a CommError). Extend the intended guard; do not bypass log_artifact.
    A second _public_api call attaches the save handle after enqueue. Catch
    communication errors from that path too; report deferred artifacts without
    preventing scalar/media logging. Type/name errors still propagate.
    No SDK files are changed on disk.
    """
    # Offline/disabled runs never use the failing online API. Leave them
    # completely untouched, including the SDK's frozen attribute namespace.
    if getattr(run, "offline", False) or getattr(run, "disabled", False):
        return
    original = getattr(run, "_assert_can_log_artifact", None)
    original_log = getattr(run, "log_artifact", None)
    if not callable(original_log) or getattr(original_log, "_flow2pc_guarded", False):
        return
    unavailable = False
    upload_error = None

    def checked(artifact):
        nonlocal unavailable
        if unavailable:
            return
        try:
            # Native and migrated controls use the same lock for API startup.
            # No model/training operation occurs inside this section.
            with startup_lock():
                return original(artifact)
        except communication_errors as error:
            unavailable = True
            record("artifact preflight", error)
            # Returning lets the original log_artifact continue to stage and
            # publish the artifact through the SDK's regular upload path.

    def logged(artifact, *args, **kwargs):
        nonlocal upload_error
        name = getattr(artifact, "name", str(artifact))
        if upload_error is not None:
            record(f"artifact upload deferred: {name}", upload_error)
            return None
        try:
            return original_log(artifact, *args, **kwargs)
        except communication_errors as error:
            upload_error = error
            record(f"artifact upload: {name}", error)
            return None

    if callable(original):
        run._assert_can_log_artifact = checked
    # W&B Run rejects new attributes. Keep the idempotence marker on our
    # ordinary Python function and replace only existing SDK methods.
    logged._flow2pc_guarded = True
    run.log_artifact = logged


class _LocalRun:
    """Native callback sink when even the offline SDK cannot start.

    The original callback still writes history/best metrics and the original
    trainer still writes checkpoints, trajectories and images to the run folder.
    """

    def __init__(self):
        self.id = "local-" + uuid4().hex[:12]
        self.url, self.offline, self.disabled, self.summary = None, True, True, {}

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def log_artifact(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass


class _RunSink:
    """Guard native scalar/summary calls without catching trainer exceptions."""

    def __init__(self, run, communication_errors, record):
        self._run, self._errors, self._record = run, communication_errors, record
        self._unavailable = False
        sink = self

        class Summary:
            def update(self, values):
                sink._call("summary", run.summary.update, values)

            def __setitem__(self, name, value):
                self.update({name: value})

        self.summary = Summary()

    def __getattr__(self, name):
        return getattr(self._run, name)

    def _call(self, operation, function, *args, **kwargs):
        if self._unavailable:
            return None
        try:
            return function(*args, **kwargs)
        except self._errors as error:
            self._unavailable = True
            self._record(operation, error)
            return None

    def log(self, *args, **kwargs):
        return self._call("scalar/media logging", self._run.log, *args, **kwargs)

    def define_metric(self, *args, **kwargs):
        return self._call("metric definition", self._run.define_metric, *args, **kwargs)

    def log_artifact(self, *args, **kwargs):
        return self._call("artifact upload", self._run.log_artifact, *args, **kwargs)


@contextmanager
def native_wandb_io(wandb, directory, mode, *, mode_changed=None):
    """Adapt only native run startup/finalization; preserve training exceptions."""
    directory = Path(directory)
    errors = []
    pending = False
    effective_mode = mode
    local_only = False
    original_init = wandb.init
    communication_errors = (wandb.errors.CommError, TimeoutError, ConnectionError)

    def write_status():
        value = {
            "status": ("local_only" if local_only else
                       "disabled" if effective_mode == "disabled" else
                       "pending_sync" if pending or effective_mode == "offline" else
                       "online_with_warnings" if errors else "online"),
            "requested_mode": mode, "effective_mode": effective_mode,
            "offline": effective_mode == "offline", "errors": errors,
        }
        target = directory / "tracking_status.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(target)

    def record(operation, error):
        nonlocal pending
        errors.append({"operation": operation, "error_type": type(error).__name__, "error": str(error)})
        pending |= operation != "artifact preflight"
        print(f"W&B {operation} warning; local results retained: {error}", file=sys.stderr, flush=True)
        write_status()

    @contextmanager
    def protected_init(*args, **kwargs):
        nonlocal effective_mode, local_only
        kwargs.setdefault("settings", {"init_timeout": 30})
        lock = startup_lock() if mode != "disabled" else nullcontext()
        with lock:
            try:
                run = original_init(*args, **kwargs)
            except communication_errors as error:
                record(f"{mode} startup", error)
                run = None
                if mode == "online":
                    effective_mode = "offline"
                    fallback = {k: v for k, v in kwargs.items() if k != "resume"}
                    fallback["mode"] = "offline"
                    try:
                        run = original_init(*args, **fallback)
                    except communication_errors as offline_error:
                        record("offline startup", offline_error)
                if run is None:
                    local_only, effective_mode = True, "disabled"
                    run = _LocalRun()
        if getattr(run, "disabled", False):
            effective_mode = "disabled"
        elif getattr(run, "offline", False):
            effective_mode = "offline"
        if mode_changed is not None:
            mode_changed(effective_mode)
        guard_artifact_uploads(run, communication_errors, record)
        write_status()
        failed = True
        try:
            yield _RunSink(run, communication_errors, record)
            failed = False
        finally:
            try:
                run.finish(exit_code=1 if failed else 0)
            except communication_errors as error:
                record("finish", error)
            write_status()

    wandb.init = protected_init
    try:
        yield
    finally:
        wandb.init = original_init
