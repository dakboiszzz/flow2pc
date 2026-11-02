"""Load and validate the original paired MFM flow/geopath checkpoints."""

from pathlib import Path

import torch

from flow2pc.pretrain.models.mfm.adapter import MetricPathSampler, MetricVelocity
from flow2pc.pretrain.models.mfm.base import LoadedTeacher
from flow2pc.pretrain.models.mfm.model import network_from_checkpoint
from flow2pc.utils.mfm_source import fingerprint, sample_shape, source_module, validate_dataset
from flow2pc.utils.paths import MFM_ROOT as DEFAULT_REPO


def checkpoint_args(checkpoint):
    for section in ("datamodule_hyper_parameters", "hyper_parameters"):
        args = checkpoint.get(section, {}).get("args")
        if args is not None:
            return dict(args) if isinstance(args, dict) else dict(vars(args))
    raise ValueError(
        "MFM checkpoint needs args in datamodule_hyper_parameters or hyper_parameters"
    )

def _read(path, repo_path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Place the MFM checkpoint at {path} before running distillation"
        )
    # Upstream geopath checkpoints pickle modules as Lightning hyperparameters.
    source_module("networks.flow_networks.mlp", repo_path)
    return torch.load(path, map_location="cpu", weights_only=False)

def build(config):
    checkpoint = _read(config["checkpoint_path"], config.get("repo_path"))
    args = checkpoint_args(checkpoint)
    return MetricVelocity(
        network_from_checkpoint(checkpoint, args, repo_path=config.get("repo_path")),
        sample_shape(args),
    )

def load(config, device):
    repo = config.get("repo_path")
    flow_path = Path(config["checkpoint_path"])
    geo_path = Path(config["geopath_checkpoint"])
    # Fail before opening either file when a required checkpoint is absent.
    for path in (flow_path, geo_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Place the MFM checkpoint at {path} before running distillation"
            )
    flow = _read(flow_path, repo)
    args = checkpoint_args(flow)
    if not args.get("mfm", True):
        raise ValueError("metric_flow expects an MFM checkpoint, not a CFM run")
    if config.get("dataset"):
        validate_dataset(args, config["dataset"])
    geo = _read(geo_path, repo)
    geo_args = checkpoint_args(geo)
    for key in (
        "data_type",
        "dim",
        "whiten",
        "seed_current",
        "t_exclude_current",
        "time_geopath",
        "sigma",
        "optimal_transport_method",
        "gamma_current",
    ):
        if args.get(key) != geo_args.get(key):
            raise ValueError(f"Flow/geopath metadata mismatch: {key}")
    if args["data_type"] in ("scrna", "image", "lidar") and args.get(
        "data_name"
    ) != geo_args.get("data_name"):
        raise ValueError("Flow/geopath metadata mismatch: data_name")
    model = MetricVelocity(
        network_from_checkpoint(flow, args, repo_path=repo), sample_shape(args)
    ).to(device)
    geopath = network_from_checkpoint(geo, geo_args, geopath=True, repo_path=repo).to(
        device
    )
    metadata = {
        "args": args,
        "sample_shape": list(sample_shape(args)),
        "flow_sha256": fingerprint(flow_path),
        "geopath_sha256": fingerprint(geo_path),
        "repo_path": str(Path(repo or DEFAULT_REPO).resolve()),
    }
    sampler = MetricPathSampler(
        geopath,
        sigma=args.get("sigma", 0.1),
        ot_method=args.get("optimal_transport_method", "exact"),
    )
    return LoadedTeacher(model, metadata, path_sampler=sampler)


def load_teacher(config, device):
    """Load the selected MFM teacher and freeze its velocity model."""
    if config["name"] not in ("mfm", "metric_flow"):
        raise ValueError(f"Unsupported MFM teacher: {config['name']}")
    result = load(config, device)
    result.model.eval().requires_grad_(False)
    return result
