"""MFM snapshot training orchestration and a bound, weights-only checkpoint pair."""

import copy
import json
import math
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

from flow2pc.data.mfm_catalog import FORMAT, digest, profile
from flow2pc.utils.mfm_source import fingerprint
from flow2pc.utils.paths import MFM_PRETRAIN_ROOT, REPO_ROOT


DEFAULTS = {
    "hidden_dims_geopath": [64, 64, 64], "hidden_dims_flow": [64, 64, 64],
    "activation_geopath": "selu", "activation_flow": "selu", "time_geopath": True,
    "sigma": 0.1, "optimal_transport_method": "exact", "velocity_metric": "land",
    "gamma_current": 0.125, "rho": 0.001, "alpha_metric": 2.0,
    "geopath_optimizer": "adam", "geopath_lr": 1e-4, "geopath_weight_decay": 1e-5,
    "flow_optimizer": "adamw", "flow_lr": 1e-3, "flow_weight_decay": 1e-5,
}
TRAINING = {"epochs": 1000, "batch_size": 128, "device": "cuda:0",
            "patience_geopath": 5, "patience_flow": 5, "validate_every": 1}


def path(value):
    result = Path(value).expanduser()
    return (REPO_ROOT / result).resolve() if not result.is_absolute() else result.resolve()


def prepare(config):
    """Metadata-only resolution; no model imports, data materialization or writes."""
    config = copy.deepcopy(config)
    allowed = {"stage", "seed", "model", "data", "training", "output", "wandb"}
    if set(config) - allowed or config.get("stage") != "pretrain":
        raise ValueError("Invalid snapshot pretraining configuration")
    for key in ("model", "data", "output"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"{key} must be a mapping")
    if set(config["model"]) - {"name", "params"} or config["model"]["name"] != "mfm":
        raise ValueError("Snapshot training requires model.name=mfm")
    params = config["model"].get("params", {})
    if not isinstance(params, dict):
        raise ValueError("model.params must be a mapping")
    if set(params) - {"backend", "source_root", "native"} or params.get("backend") != "snapshots":
        raise ValueError("Use MFM backend=snapshots with source_root/native options")
    data = config["data"]
    if set(data) - {"name", "root", "params"} or data["name"] != "mfm_snapshots":
        raise ValueError("Snapshot training requires data.name=mfm_snapshots")
    if set(data.get("params", {})) != {"profile"}:
        raise ValueError("data.params requires only profile")
    spec = profile(data["params"]["profile"])
    seed = config.setdefault("seed", 42)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid seed")
    native = params.get("native", {})
    if not isinstance(native, dict):
        raise ValueError("model.params.native must be a mapping")
    if set(native) - DEFAULTS.keys():
        raise ValueError(f"Unknown native options: {set(native) - DEFAULTS.keys()}")
    native = {**DEFAULTS, **native}
    if native["velocity_metric"] != "land" or native["optimal_transport_method"] not in ("exact", "None"):
        raise ValueError("Snapshot recipes use original LAND and exact OT or independent coupling")
    for key in ("hidden_dims_geopath", "hidden_dims_flow"):
        if not isinstance(native[key], list) or not native[key] or any(type(x) is not int or x < 1 for x in native[key]):
            raise ValueError(f"Invalid {key}")
    for key in ("gamma_current", "rho", "alpha_metric", "geopath_lr", "flow_lr"):
        if type(native[key]) not in (int, float) or not math.isfinite(native[key]) or native[key] <= 0:
            raise ValueError(f"Invalid {key}")
    for key in ("sigma", "geopath_weight_decay", "flow_weight_decay"):
        if type(native[key]) not in (int, float) or not math.isfinite(native[key]) or native[key] < 0:
            raise ValueError(f"Invalid {key}")
    if type(native["time_geopath"]) is not bool:
        raise ValueError("time_geopath must be boolean")
    for key in ("geopath_optimizer", "flow_optimizer"):
        if native[key] not in ("adam", "adamw"):
            raise ValueError(f"Invalid {key}")
    settings = config.get("training", {})
    if not isinstance(settings, dict):
        raise ValueError("training must be a mapping")
    if set(settings) - TRAINING.keys():
        raise ValueError("Unknown snapshot training options")
    settings = {**TRAINING, **settings}
    for key in TRAINING.keys() - {"device"}:
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"training.{key} must be positive")
    if settings["validate_every"] > settings["epochs"]:
        raise ValueError("validate_every must not exceed the teacher epoch budget")
    device = settings["device"]
    if not isinstance(device, str) or not (device in ("cpu", "cuda", "auto") or (device.startswith("cuda:") and device[5:].isdigit())):
        raise ValueError("Use cpu, auto, cuda or cuda:N")
    wandb = config.setdefault("wandb", {"mode": "disabled"})
    if not isinstance(wandb, dict):
        raise ValueError("wandb must be a mapping")
    if set(wandb) - {"mode", "project", "run_name"} or wandb.get("mode", "disabled") not in ("disabled", "offline", "online"):
        raise ValueError("Invalid wandb configuration")
    if set(config["output"]) != {"dir"}:
        raise ValueError("output requires dir")
    source = path(params.get("source_root", str(MFM_PRETRAIN_ROOT)))
    if not (source / "mfm/flow_matchers/models/mfm.py").is_file():
        raise FileNotFoundError(f"Original MFM source missing: {source}")
    config["model"]["params"] = {"backend": "snapshots", "source_root": str(source), "native": native}
    config["data"]["root"] = str(path(data["root"]))
    config["output"]["dir"] = str(path(config["output"]["dir"]))
    config["training"] = settings
    return {"method": "mfm", "backend": "snapshots", "config": config, "dim": spec["dim"]}


def source_identity(source):
    source = Path(source)
    files = {str(p.relative_to(source)): fingerprint(p) for p in sorted((source / "mfm").rglob("*.py"))}
    revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                              text=True, capture_output=True, check=False)
    adapters = {name: fingerprint(Path(__file__).with_name(name))
                for name in ("snapshots.py", "snapshot_native.py", "model.py", "adapter.py")}
    return {"files": files, "adapter_files": adapters,
            "revision": revision.stdout.strip() if revision.returncode == 0 else None}


def run(resolved):
    import torch
    import yaml
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, WandbLogger
    from torchcfm.optimal_transport import OTPlanSampler

    from flow2pc.data.mfm_snapshots import load
    from flow2pc.pretrain.models.mfm.model import make_network
    from flow2pc.pretrain.models.mfm.snapshot_native import data_module, native_classes
    from flow2pc.utils.checkpoint import atomic_json, atomic_path
    from flow2pc.utils.mfm_source import source_module
    from flow2pc.utils.seed import seed_everything

    config = resolved["config"]
    params, training = config["model"]["params"], config["training"]
    source, output = params["source_root"], Path(config["output"]["dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty run: {output}")
    data = load({"data_dir": config["data"]["root"]}, {})
    if data.source_info["profile"] != config["data"]["params"]["profile"]:
        raise ValueError("Prepared profile does not match config")
    if data.source_info["split_seed"] != config["seed"]:
        raise ValueError("Prepared split seed does not match config")
    seed_everything(config["seed"])
    provenance = source_identity(source)
    matcher_cls, geo_cls, flow_cls = native_classes(source)
    dm = data_module(data, training["batch_size"])
    device = training["device"]
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    accelerator = "gpu" if device.startswith("cuda") else "cpu"
    devices = [int(device.split(":")[1]) if ":" in device else 0] if accelerator == "gpu" else 1
    args = SimpleNamespace(**params["native"], dim=data.dim, data_type="snapshots",
                           data_name=data.source_info["profile"], seed_current=config["seed"],
                           mfm=True, whiten=False, working_dir=str(output), accelerator=accelerator,
                           n_centers=100, kappa=1.0, metric_epochs=50, metric_patience=5, metric_lr=0.01)
    geo_net = make_network(vars(args), geopath=True, repo_path=source)
    flow_net = make_network(vars(args), repo_path=source)
    matcher = matcher_cls(geopath_net=geo_net, sigma=args.sigma, alpha=1.0)
    ot = OTPlanSampler(method="exact") if args.optimal_transport_method == "exact" else None
    metric = source_module("geo_metrics.metric_factory", source).DataManifoldMetric(args, [], dm)
    grid = [float(data.times[i]) for i in data.observed_indices]
    geo = geo_cls(matcher, args, skipped_time_points=[], ot_sampler=ot, data_manifold_metric=metric)
    # Exclude live modules from Lightning hyperparameters; exported artifacts are weights-only.
    geo.hparams.clear()
    geo.save_hyperparameters({"args": vars(args)})
    geo.snapshot_times = grid
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    atomic_json(output / "status.json", {"status": "running"})
    logger = CSVLogger(str(output), name="logs")
    tracking = config.get("wandb", {})
    if tracking.get("mode", "disabled") != "disabled":
        logger = WandbLogger(project=tracking.get("project", "flow2pc"),
                             name=tracking.get("run_name"), save_dir=str(output),
                             offline=tracking["mode"] == "offline")

    def trainer(phase, monitor, patience):
        callback = ModelCheckpoint(dirpath=output / "native" / phase, filename="best",
                                   monitor=monitor, mode="min", save_top_k=1)
        engine = Trainer(max_epochs=training["epochs"], accelerator=accelerator, devices=devices,
                         logger=logger, callbacks=[callback, EarlyStopping(monitor=monitor, patience=patience)],
                         default_root_dir=str(output), num_sanity_val_steps=0,
                         check_val_every_n_epoch=1 if phase == "geopath" else training["validate_every"],
                         inference_mode=False)
        return engine, callback

    try:
        engine, geo_callback = trainer("geopath", "GeoPathNet/val_loss_geopath", training["patience_geopath"])
        engine.fit(geo, datamodule=dm)
        if not geo_callback.best_model_path:
            raise ValueError("Geopath training produced no validation-selected checkpoint")
        selected_geo = torch.load(geo_callback.best_model_path, map_location="cpu", weights_only=True)
        geo.load_state_dict(selected_geo["state_dict"], strict=True)
        geo.geopath_net.eval().requires_grad_(False)
        # The second phase uses exactly the selected geopath, with no further updates.
        matcher.geopath_net = geo.geopath_net.cpu()
        flow = flow_cls(matcher, flow_net, skipped_time_points=[], ot_sampler=ot, args=args)
        flow.save_hyperparameters({"args": vars(args)})
        flow.snapshot_times = grid
        # The native matcher is not an nn.Module; register its network for device movement.
        flow.add_module("frozen_geopath", matcher.geopath_net)
        engine, flow_callback = trainer("flow", "FlowNet/val_loss_cfm", training["patience_flow"])
        engine.fit(flow, datamodule=dm)
        if not flow_callback.best_model_path:
            raise ValueError("Flow training produced no validation-selected checkpoint")
        selected_flow = torch.load(flow_callback.best_model_path, map_location="cpu", weights_only=True)
        if source_identity(source) != provenance:
            raise ValueError("MFM source changed during training")
        binding = {"format": FORMAT, "pair_id": uuid.uuid4().hex, "args": vars(args),
                   "snapshot_identity": data.source_info["identity"], "profile": data.source_info["profile"],
                   "prepared_path": config["data"]["root"], "source": provenance,
                   "training": training, "effective_batch_size": dm.batch_size,
                   "model_times": data.source_info["model_times"], "observed_indices": data.observed_indices}
        files = {}
        for role, selected, prefix in (("geopath", selected_geo, "geopath_net."), ("flow", selected_flow, "flow_net.")):
            weights = {k: v.cpu() for k, v in selected["state_dict"].items() if k.startswith(prefix)}
            if not weights:
                raise ValueError(f"Selected {role} checkpoint has no weights")
            destination = output / "checkpoints" / f"{role}.pt"
            payload = {"binding": binding, "role": role, "state_dict": weights}
            with atomic_path(destination) as temporary:
                torch.save(payload, temporary)
            files[role] = {"path": str(destination.resolve()), "sha256": fingerprint(destination)}
        atomic_json(output / "teacher_pair.json", {"binding": binding, "binding_identity": digest(binding), "files": files})
        atomic_json(output / "status.json", {"status": "complete", "teacher_pair": "teacher_pair.json"})
    except BaseException as error:
        atomic_json(output / "status.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        if isinstance(logger, WandbLogger):
            logger.experiment.finish()
    return output


def load_teacher(config, device):
    import torch

    from flow2pc.pretrain.models.mfm.adapter import MetricPathSampler, MetricVelocity
    from flow2pc.pretrain.models.mfm.base import LoadedTeacher
    from flow2pc.pretrain.models.mfm.model import network_from_checkpoint

    flow_path, geo_path = path(config["checkpoint_path"]), path(config["geopath_checkpoint"])
    pair = json.loads((flow_path.parent.parent / "teacher_pair.json").read_text())
    binding = pair["binding"]
    if binding.get("format") != FORMAT or digest(binding) != pair["binding_identity"]:
        raise ValueError("Invalid MFM checkpoint pair manifest")
    payloads = {}
    for role, checkpoint in (("flow", flow_path), ("geopath", geo_path)):
        if fingerprint(checkpoint) != pair["files"][role]["sha256"]:
            raise ValueError(f"MFM {role} checkpoint pair checksum mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("binding") != binding or payload.get("role") != role:
            raise ValueError("MFM flow/geopath pair binding mismatch")
        payloads[role] = payload
    source = path(config.get("repo_path", str(MFM_PRETRAIN_ROOT)))
    if source_identity(source) != binding["source"]:
        raise ValueError("MFM source differs from the trained checkpoint")
    args = binding["args"]
    flow = network_from_checkpoint(payloads["flow"], args, repo_path=source).to(device)
    geo = network_from_checkpoint(payloads["geopath"], args, geopath=True, repo_path=source).to(device)
    model = MetricVelocity(flow, (args["dim"],)).eval().requires_grad_(False)
    metadata = {**binding, "repo_path": str(source), "sample_shape": [args["dim"]],
                "flow_sha256": fingerprint(flow_path), "geopath_sha256": fingerprint(geo_path)}
    return LoadedTeacher(model, metadata, path_sampler=MetricPathSampler(
        geo, sigma=args["sigma"], ot_method=args["optimal_transport_method"]))
