"""Shared local metrics and explicit W&B upload state; no model dependencies."""

import json
import math
import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
from uuid import uuid4

from flow2pc.utils.checkpoint import atomic_json
from flow2pc.utils.wandb_io import guard_artifact_uploads


class LocalLogger:
    def __init__(self, directory, config, append=False):
        self.directory = Path(directory)
        self.mode = config["logging"]["mode"]
        self.resilient = (config.get("training", {}).get("protocol") == "marginal"
                          or config.get("model", {}).get("name") == "otpfm")
        self.tracking_errors = []
        self.offline = False
        self._finish_run = None
        self.handle = (directory / "metrics.jsonl").open("a" if append else "w")
        self.run = None
        try:
            if config["logging"]["mode"] != "disabled":
                import wandb

                metadata_path = directory / "wandb_run.json"
                tracking = dict(config["logging"])
                self.offline = tracking["mode"] == "offline"
                if metadata_path.is_file():
                    metadata = json.loads(metadata_path.read_text())
                    if metadata["project"] != tracking.get("project", "flow2pc") or (
                        tracking.get("entity") is not None
                        and metadata.get("entity") != tracking["entity"]
                    ):
                        raise ValueError(
                            "W&B resume project/entity differs from the saved run"
                        )
                    tracking["name"] = metadata["name"]
                    if metadata.get("entity") is not None:
                        tracking["entity"] = metadata["entity"]
                else:
                    metadata = {
                        "id": uuid4().hex[:16],
                        "name": tracking.get("name", directory.name),
                        "project": tracking.get("project", "flow2pc"),
                        "entity": tracking.get("entity"),
                    }
                    atomic_json(metadata_path, metadata)
                    tracking["name"] = metadata["name"]
                if tracking["mode"] == "online":
                    tracking["resume"] = "allow"
                # Reuse the old sweep's startup lock when many W&B SDK workers
                # start together. Training runs concurrently after initialization.
                with ExitStack() as stack:
                    startup_lock = os.environ.get("FLOW2PC_WANDB_INIT_LOCK")
                    if startup_lock:
                        import fcntl

                        path = Path(startup_lock)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        handle = stack.enter_context(path.open("a"))
                        fcntl.flock(handle, fcntl.LOCK_EX)
                    init_options = ({"settings": {"init_timeout": 30}}
                                    if config.get("model", {}).get("name") == "otpfm" else {})
                    try:
                        self.run = wandb.init(
                            **tracking, id=metadata["id"], config=config, dir=str(directory), **init_options
                        )
                    except (wandb.errors.CommError, TimeoutError, ConnectionError) as error:
                        if not self.resilient:
                            raise
                        self.tracking_failed("online startup", error)
                        # Keep full SDK history/media/artifacts locally when the
                        # remote service is unavailable. Sync it later explicitly.
                        fallback = {k: v for k, v in tracking.items() if k != "resume"}
                        fallback["mode"] = "offline"
                        self.offline = True
                        try:
                            self.run = wandb.init(**fallback, id=metadata["id"], config=config, dir=str(directory), **init_options)
                        except (wandb.errors.CommError, TimeoutError, ConnectionError) as offline_error:
                            self.tracking_failed("offline startup", offline_error)
                self._finish_run = self.run
                if config.get("model", {}).get("name") == "otpfm" and self.run is not None:
                    guard_artifact_uploads(
                        self.run, (wandb.errors.CommError, TimeoutError, ConnectionError),
                        self.tracking_failed,
                    )
                atomic_json(
                    metadata_path,
                    {**metadata, "url": self.run.url if self.run is not None else None,
                     "entity": self.run.entity if self.run is not None else metadata.get("entity")},
                )
                self.write_tracking_status()
        except BaseException:
            self.handle.close()
            raise

    def write_tracking_status(self):
        if self.resilient:
            atomic_json(self.directory / "tracking_status.json", {
                "status": ("disabled" if self.mode == "disabled" else
                           "local_only" if self._finish_run is None else
                           "pending_sync" if self.tracking_errors or self.offline else "online"),
                "requested_mode": self.mode,
                "effective_mode": ("disabled" if self.mode == "disabled" else
                                   "local" if self._finish_run is None else
                                   "offline" if self.offline else "online"),
                "offline": self.offline, "errors": self.tracking_errors,
            })

    def tracking_failed(self, operation, error):
        self.tracking_errors.append({"operation": operation, "error": str(error)})
        self.write_tracking_status()

    @contextmanager
    def remote(self, operation):
        import wandb

        try:
            yield
        except (wandb.errors.CommError, TimeoutError, ConnectionError) as error:
            if not self.resilient:
                raise
            self.tracking_failed(operation, error)
            # Further local writes must not repeatedly wait on a failed API.
            self.run = None

    def log(self, row):
        if any(
            isinstance(v, (int, float)) and not math.isfinite(v) for v in row.values()
        ):
            raise ValueError("Non-finite ALI-CFM training metric")
        self.handle.write(json.dumps(row, allow_nan=False) + "\n")
        self.handle.flush()
        if self.run is not None:
            with self.remote("scalar logging"):
                self.run.log(row)

    def close(self, failed=False):
        self.handle.close()
        if self._finish_run is not None:
            with self.remote("finish"):
                self._finish_run.finish(exit_code=1 if failed else 0)
        self.write_tracking_status()

