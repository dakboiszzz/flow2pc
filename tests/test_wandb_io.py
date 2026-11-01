"""Logging fault regressions; no W&B service or model execution is required."""

import json
from types import SimpleNamespace

import pytest

from flow2pc.utils.wandb_io import guard_artifact_uploads, native_wandb_io


class CommunicationError(Exception):
    pass


class AuthenticationError(CommunicationError):
    pass


class FakeRun:
    def __init__(self):
        self.preflight_calls = 0
        self.published = []
        self.exit_codes = []
        self.summary = {}
        self.id, self.url, self.offline, self.disabled = "fixture", "https://fixture", False, False

    def log(self, *args, **kwargs):
        pass

    def define_metric(self, *args, **kwargs):
        pass

    def _assert_can_log_artifact(self, artifact):
        self.preflight_calls += 1
        raise AuthenticationError("service process is busy")

    def log_artifact(self, artifact):
        self._assert_can_log_artifact(artifact)
        self.published.append(artifact)

    def finish(self, *, exit_code):
        self.exit_codes.append(exit_code)


class FrozenRun(FakeRun):
    """Mirror the installed W&B 0.30.0 rule for adding Run attributes."""

    def __init__(self):
        super().__init__()
        self._frozen = True

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False) and not hasattr(self, name):
            raise Exception(f"Attribute {name} is not supported on Run object.")
        super().__setattr__(name, value)


def test_online_guard_supports_frozen_sdk_run_and_is_idempotent(monkeypatch):
    monkeypatch.delenv("FLOW2PC_WANDB_INIT_LOCK", raising=False)
    run, errors = FrozenRun(), []
    guard_artifact_uploads(run, (CommunicationError,), lambda *args: errors.append(args))
    first = run.log_artifact
    guard_artifact_uploads(run, (CommunicationError,), lambda *args: errors.append(args))
    assert run.log_artifact is first
    run.log_artifact("source")
    assert run.published == ["source"]
    assert len(errors) == 1
    assert not hasattr(run, "_flow2pc_artifacts_guarded")


def test_offline_frozen_run_is_not_modified():
    run = FrozenRun()
    run.offline = True
    before = dict(vars(run))
    guard_artifact_uploads(run, (CommunicationError,), lambda *args: None)
    assert vars(run) == before


def test_failed_preflight_still_queues_artifacts_and_does_not_retry_authentication(monkeypatch):
    monkeypatch.delenv("FLOW2PC_WANDB_INIT_LOCK", raising=False)
    run, errors = FakeRun(), []
    guard_artifact_uploads(run, (CommunicationError,), lambda *args: errors.append(args))
    run.log_artifact("source")
    run.log_artifact("checkpoints-and-images")
    assert run.published == ["source", "checkpoints-and-images"]
    assert run.preflight_calls == 1
    assert len(errors) == 1


def test_artifact_validation_errors_still_propagate(monkeypatch):
    monkeypatch.delenv("FLOW2PC_WANDB_INIT_LOCK", raising=False)
    run = FakeRun()
    def wrong_type(artifact):
        raise ValueError("artifact type mismatch")
    run._assert_can_log_artifact = wrong_type
    guard_artifact_uploads(run, (CommunicationError,), lambda *args: None)
    with pytest.raises(ValueError, match="type mismatch"):
        run.log_artifact("source")
    assert not run.published


def test_upload_handle_failure_is_nonfatal_and_later_artifacts_are_marked_deferred(monkeypatch):
    monkeypatch.delenv("FLOW2PC_WANDB_INIT_LOCK", raising=False)
    run, errors = FakeRun(), []
    def failed_save_handle(artifact):
        run.published.append(artifact)
        raise AuthenticationError("save handle authentication timed out after enqueue")
    run.log_artifact = failed_save_handle
    guard_artifact_uploads(run, (CommunicationError,), lambda *args: errors.append(args))
    assert run.log_artifact("source") is None
    assert run.log_artifact("outputs") is None
    assert run.published == ["source"]
    assert errors[0][0] == "artifact upload: source"
    assert errors[1][0] == "artifact upload deferred: outputs"


def test_native_finish_timeout_preserves_local_completion_and_restores_init(tmp_path, monkeypatch):
    monkeypatch.delenv("FLOW2PC_WANDB_INIT_LOCK", raising=False)
    run = FakeRun()
    def timeout(*, exit_code):
        run.exit_codes.append(exit_code)
        raise CommunicationError("finish timeout")
    run.finish = timeout
    original_init = lambda **kwargs: run
    sdk = SimpleNamespace(init=original_init, errors=SimpleNamespace(CommError=CommunicationError))
    with native_wandb_io(sdk, tmp_path, "online"):
        with sdk.init(project="flow2pc") as active:
            active.log_artifact("source")
            (tmp_path / "trained.txt").write_text("local computation completed")
        # Mirrors native run_one writing result.json after its run context.
        (tmp_path / "result.json").write_text('{"complete": true}')
    assert sdk.init is original_init
    assert run.exit_codes == [0]
    assert (tmp_path / "result.json").is_file()
    assert json.loads((tmp_path / "tracking_status.json").read_text())["status"] == "pending_sync"


def test_native_training_exception_is_not_suppressed(tmp_path, monkeypatch):
    monkeypatch.delenv("FLOW2PC_WANDB_INIT_LOCK", raising=False)
    run = FakeRun()
    original_init = lambda **kwargs: run
    sdk = SimpleNamespace(init=original_init, errors=SimpleNamespace(CommError=CommunicationError))
    with pytest.raises(RuntimeError, match="training failed"):
        with native_wandb_io(sdk, tmp_path, "online"):
            with sdk.init(project="flow2pc"):
                raise RuntimeError("training failed")
    assert run.exit_codes == [1]
    assert sdk.init is original_init
    assert not (tmp_path / "result.json").exists()


def test_native_online_outage_falls_back_before_strict_online_check(tmp_path, monkeypatch):
    monkeypatch.delenv("FLOW2PC_WANDB_INIT_LOCK", raising=False)
    calls, modes, run = [], [], FakeRun()
    def original_init(**kwargs):
        calls.append(kwargs["mode"])
        if kwargs["mode"] == "online":
            raise AuthenticationError("W&B unavailable")
        run.offline, run.url = True, None
        return run
    sdk = SimpleNamespace(init=original_init, errors=SimpleNamespace(CommError=CommunicationError))
    with native_wandb_io(sdk, tmp_path, "online", mode_changed=modes.append):
        with sdk.init(project="flow2pc", mode="online") as active:
            # Native run_one performs this check immediately after startup.
            assert not (modes[-1] == "online" and (active.disabled or active.offline or not active.url))
            (tmp_path / "training_started.txt").write_text("started")
    assert calls == ["online", "offline"]
    assert modes == ["offline"]
    assert (tmp_path / "training_started.txt").is_file()
    status = json.loads((tmp_path / "tracking_status.json").read_text())
    assert status["effective_mode"] == "offline"
    assert status["status"] == "pending_sync"


def test_native_offline_sdk_failure_still_allows_local_computation(tmp_path, monkeypatch):
    monkeypatch.delenv("FLOW2PC_WANDB_INIT_LOCK", raising=False)
    calls, modes = [], []
    def original_init(**kwargs):
        calls.append(kwargs["mode"])
        raise CommunicationError("SDK startup unavailable")
    sdk = SimpleNamespace(init=original_init, errors=SimpleNamespace(CommError=CommunicationError))
    with native_wandb_io(sdk, tmp_path, "online", mode_changed=modes.append):
        with sdk.init(project="flow2pc", mode="online") as active:
            active.summary.update({"status": "training"})
            active.log({"epoch": 0, "train/loss": 1.0})
            (tmp_path / "history.jsonl").write_text('{"epoch": 0, "train/loss": 1.0}\n')
    assert calls == ["online", "offline"]
    assert modes == ["disabled"]
    assert (tmp_path / "history.jsonl").is_file()
    assert json.loads((tmp_path / "tracking_status.json").read_text())["status"] == "local_only"


def test_native_scalar_logging_outage_does_not_abort_local_history(tmp_path, monkeypatch):
    monkeypatch.delenv("FLOW2PC_WANDB_INIT_LOCK", raising=False)
    run = FakeRun()
    calls = []
    def failed_log(values):
        calls.append(values)
        raise CommunicationError("service stopped")
    run.log = failed_log
    def original_init(**kwargs):
        return run
    sdk = SimpleNamespace(init=original_init, errors=SimpleNamespace(CommError=CommunicationError))
    with native_wandb_io(sdk, tmp_path, "online"):
        with sdk.init(project="flow2pc", mode="online") as active:
            active.log({"epoch": 0})
            active.log({"epoch": 1})
            (tmp_path / "history.jsonl").write_text('{"epoch": 0}\n{"epoch": 1}\n')
    assert calls == [{"epoch": 0}]
    assert (tmp_path / "history.jsonl").is_file()
    assert json.loads((tmp_path / "tracking_status.json").read_text())["status"] == "pending_sync"
