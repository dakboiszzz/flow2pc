"""Resolve MMFM pretraining without importing upstream code or opening datasets."""

from copy import deepcopy
import math
from pathlib import Path

from flow2pc.data.metric_profiles import vector_profile
from flow2pc.data.mfm_catalog import profile as snapshot_profile
from flow2pc.pretrain.models.mmfm.source import DEFAULT_SOURCE
from flow2pc.utils.paths import MFM_PRETRAIN_ROOT, REPO_ROOT


def mapping(value, label, allowed):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f"Invalid {label}; allowed keys: {sorted(allowed)}")
    return value


def path(value):
    value = Path(value).expanduser()
    return str(value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve())


def prepare(raw):
    c = deepcopy(raw)
    mapping(c, "MMFM config", {"stage", "model", "data", "training", "seed", "output", "evaluation", "wandb"})
    if c.get("stage") != "pretrain" or c.get("model", {}).get("name") != "mmfm":
        raise ValueError("MMFM pretraining requires stage: pretrain and model.name: mmfm")
    mapping(c["model"], "model", {"name", "params"})
    params = mapping(c["model"].get("params", {}), "model.params", {
        "source_root", "x_latent_dim", "time_embed_dim", "num_out_layers", "activation",
        "conditional_model", "interpolation", "flow_variance", "coupling",
    })
    source_root = path(params.get("source_root", DEFAULT_SOURCE))
    seed = c.get("seed", 42)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    data = mapping(c.get("data", {}), "data", {"name", "root", "params"})
    name = data.get("name")
    options = mapping(data.get("params", {}), "data.params", {
        "source_root", "whiten", "heldout_index", "train_fraction", "profile",
    })
    if name == "mfm_snapshots":
        spec = snapshot_profile(options.get("profile"))
        if set(options) - {"profile"}:
            raise ValueError("Prepared snapshots accept only data.params.profile")
        profile = {"repo_path": str(MFM_PRETRAIN_ROOT), "args": {"dim": spec["dim"], "whiten": False}, "profile": options["profile"]}
    else:
        profile = vector_profile(name, seed, **options)
    profile["repo_path"] = path(profile["repo_path"])
    if type(profile["args"]["whiten"]) is not bool:
        raise ValueError("data.params.whiten must be boolean")
    model_config = {
        "data_dim": profile["args"]["dim"], "conditional_model": False,
        "x_latent_dim": params.get("x_latent_dim", 64),
        "time_embed_dim": params.get("time_embed_dim", 64),
        "num_out_layers": params.get("num_out_layers", 3),
        "activation": params.get("activation", "LeakyReLU"),
        "normalization": None,
    }
    if params.get("conditional_model", False) is not False:
        raise ValueError("This integration supports unconditional MMFM only")
    for key in ("x_latent_dim", "time_embed_dim", "num_out_layers"):
        if type(model_config[key]) is not int or model_config[key] < 2:
            raise ValueError(f"model.params.{key} must be an integer >= 2")
    if model_config["activation"] not in ("SELU", "ReLU", "LeakyReLU"):
        raise ValueError("Unsupported native MMFM activation")
    matcher = {
        "coupling": params.get("coupling", "minibatch_ot"),
        "interpolation": params.get("interpolation", "cubic"),
        "flow_variance": params.get("flow_variance", "adaptive2-4-0.01"),
    }
    if matcher["coupling"] != "minibatch_ot" or matcher["interpolation"] not in ("cubic", "linear", "lagrange"):
        raise ValueError("Use minibatch_ot and a native cubic/linear/lagrange interpolant")
    sigma = matcher["flow_variance"]
    if isinstance(sigma, (int, float)) and not isinstance(sigma, bool):
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("flow_variance must be finite and nonnegative")
    elif isinstance(sigma, str):
        pieces = sigma.split("-")
        if len(pieces) != 3 or pieces[0] not in ("adaptive1", "adaptive2", "adaptive3", "adaptive4"):
            raise ValueError("Invalid native MMFM adaptive variance")
        maximum, minimum = float(pieces[1]), float(pieces[2])
        if not (math.isfinite(maximum) and math.isfinite(minimum) and maximum > 0 and minimum >= 0):
            raise ValueError("Invalid adaptive variance coefficients")
    else:
        raise ValueError("Invalid flow_variance")
    defaults = {
        "epochs": 300, "batch_size": 128, "lr": 0.01, "gradient_clip": 1.0,
        "device": "auto", "steps_per_epoch": None, "validation_batches": 4,
        "resume_checkpoint": None,
    }
    training = {**defaults, **mapping(c.get("training", {}), "training", defaults)}
    for key in ("epochs", "batch_size", "validation_batches"):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    count = training["steps_per_epoch"]
    if count is not None and (type(count) is not int or count < 1):
        raise ValueError("steps_per_epoch must be null or a positive integer")
    for key in ("lr", "gradient_clip"):
        if not math.isfinite(float(training[key])) or float(training[key]) <= 0:
            raise ValueError(f"training.{key} must be positive and finite")
    if training["resume_checkpoint"]:
        training["resume_checkpoint"] = path(training["resume_checkpoint"])
    output = mapping(c.get("output", {}), "output", {"dir"})
    if not output.get("dir"):
        raise ValueError("output.dir is required")
    evaluation = {"enabled": True, "samples": 128, "projections": 32, "ode_substeps": 16, "seed": 1234,
                  **mapping(c.get("evaluation", {}), "evaluation", {"enabled", "samples", "projections", "ode_substeps", "seed"})}
    for key in ("samples", "projections", "ode_substeps"):
        if type(evaluation[key]) is not int or evaluation[key] < 1:
            raise ValueError(f"evaluation.{key} must be a positive integer")
    wandb = {"mode": "disabled", **mapping(c.get("wandb", {}), "wandb", {"mode", "project", "group", "run_name"})}
    if wandb["mode"] not in ("disabled", "online", "offline"):
        raise ValueError("Invalid wandb.mode")
    return {
        "method": "mmfm", "source_root": source_root, "model_config": model_config,
        "matcher": matcher, "seed": seed, "dataset_profile": profile,
        "dataset_config": {
            "name": name, "data_dir": path(data.get("root", MFM_PRETRAIN_ROOT / "data")),
            "seed": seed, "train_fraction": options.get("train_fraction", 0.9),
            "batch_size": training["batch_size"],
        },
        "training": training, "output_dir": path(output["dir"]),
        "evaluation": evaluation, "wandb": wandb,
    }
