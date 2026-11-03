"""Original MFM schema validation; does not load models or data."""
from copy import deepcopy
import math

from flow2pc.utils.combinations import validate_combination


def checkpoint_training_config(checkpoint, *, resume=False):
    """Normalize legacy metadata in memory; never rewrite checkpoint artifacts."""
    config = deepcopy(checkpoint["training_config"])
    training = config["training"]
    legacy_weights = [
        container["source_nll_weight"]
        for container in (checkpoint, training)
        if "source_nll_weight" in container
    ]
    if resume and any(float(weight) != 0.0 for weight in legacy_weights):
        raise ValueError("Cannot resume a legacy checkpoint with nonzero source NLL weight")
    training.pop("source_nll_weight", None)
    return config


def validate_config(config):
    """Validate configuration only; never open data, source models, or weights."""
    validate_combination(config.dataset["name"], config.teacher["name"])
    if config.task != "distill":
        raise ValueError("metric_flow supports pretrained-teacher distillation only")
    required = config.teacher.get("checkpoint_path")
    if not required or not config.teacher.get("options", {}).get("geopath_checkpoint"):
        raise ValueError(
            "metric_flow config must specify checkpoint_path and options.geopath_checkpoint"
        )
    structure = config.student.get("pc_structure", "hclt")
    if structure not in ("hclt", "gaussian"):
        raise ValueError("MFM supports hclt and gaussian students")
    if config.student.get("topology", "learned") not in ("learned", "balanced"):
        raise ValueError("topology must be learned or balanced")
    allowed = {
        "teacher": {"name", "checkpoint_path", "options"},
        "dataset": {"name", "data_dir", "options"},
        "student": {
            "pc_structure",
            "num_latents",
            "num_components",
            "time_order",
            "time_embedding_dim",
            "topology",
            "topology_samples",
        },
        "training": {
            "steps",
            "batch_size",
            "lr",
            "seed",
            "device",
            "output_dir",
            "resume_checkpoint",
            "gradient_clip",
            "log_every",
            "validate_every",
            "checkpoint_every",
        },
        "evaluation": {
            "enabled",
            "samples",
            "projections",
            "ode_substeps",
            "seed",
            "decode_images",
            "vae_path",
        },
        "wandb": {"mode", "project", "group", "run_name"},
    }
    for group, names in allowed.items():
        unknown = set(getattr(config, group)) - names
        if unknown:
            raise ValueError(f"Unknown metric {group} options: {sorted(unknown)}")
    counts = {
        "steps": 20000,
        "batch_size": 128,
        "log_every": 100,
        "validate_every": 1000,
        "checkpoint_every": 5000,
    }
    for key, default in counts.items():
        if int(config.training.get(key, default)) < 1:
            raise ValueError(f"{key} must be positive")
    for key, default in (("lr", 0.001), ("gradient_clip", 1.0)):
        value = float(config.training.get(key, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key, default in (("samples", 128), ("projections", 32), ("ode_substeps", 8)):
        if int(config.evaluation.get(key, default)) < 1:
            raise ValueError(f"evaluation.{key} must be positive")
    for key, default in (
        ("num_latents", 64),
        ("num_components", 64),
        ("time_order", 4),
        ("topology_samples", 2048),
    ):
        if int(config.student.get(key, default)) < (
            2 if key in ("num_latents", "topology_samples") else 1
        ):
            raise ValueError(f"Invalid student.{key}")
    embedding = int(config.student.get("time_embedding_dim", 64))
    if embedding < 2 or embedding % 2:
        raise ValueError("time_embedding_dim must be positive and even")
    if config.wandb.get("mode", "disabled") not in ("disabled", "offline", "online"):
        raise ValueError("wandb.mode must be disabled, offline, or online")
    if config.evaluation.get("decode_images") and config.dataset["name"] != "afhq":
        raise ValueError("decode_images requires the afhq dataset")

