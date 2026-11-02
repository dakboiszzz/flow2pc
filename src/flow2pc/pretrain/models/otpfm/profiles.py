"""Dataset profiles for the common OTP-FM pipeline; add native adapters here."""

import importlib
from dataclasses import dataclass

from flow2pc.data.preprocessing import scaler_from_statistics

from ._source.factory import create_model, create_potentials


@dataclass(frozen=True)
class DatasetProfile:
    native_name: str
    data_module: str
    trainer_module: str
    trainer_class: str
    alignment_function: str
    loader_function: str
    source_subdir: str
    metric_time_rule: str = "native nearest stored time"


@dataclass(frozen=True)
class MarginalProfile:
    """Shared-data adaptation; native transport profiles retain their own path."""

    source_subdir: str
    protocol: str = "shared_ode"


PROFILES = {}


def register_profile(name, profile):
    if name in PROFILES:
        raise ValueError(f"OTP-FM dataset profile already registered: {name}")
    PROFILES[name] = profile


def get_profile(name):
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"OTP-FM has no native adapter for {name!r}; registered profiles: {sorted(PROFILES)}") from None


for name, native, cls, function in (
    ("gom", "gulfofmexico", "GoMTrainer", "gom"),
    ("beijing", "beijingair", "BeijingTrainer", "beijing"),
):
    register_profile(name, DatasetProfile(
        native, f".{function}_data", f".{function}_trainer", cls,
        f"compute_{function}_ot_alignments", f"create_{function}_dataloaders", name,
        "native nearby-time minimum for holdouts" if name == "gom" else "native nearest stored time",
    ))


for name, subdir in (
    ("cite", "single_cell"), ("multi", "single_cell"), ("eb", "single_cell"),
    ("st", "spatial_transcriptomics"), ("cell_tracking", "cell_tracking"),
    ("knot", "knot"),
):
    register_profile(name, MarginalProfile(subdir))


def setup_native(dataset, config, device):
    """Preserve native coupling -> DataLoader -> potential -> network order.

    Raw loading and scaling are shared data responsibilities. Pairings remain
    method-owned, copied original OT chains, never observed trajectory IDs.
    """
    profile = get_profile(dataset.metadata.name)
    data = importlib.import_module(profile.data_module, __package__ + "._source")
    trainers = importlib.import_module(profile.trainer_module, __package__ + "._source")
    marginals = [cloud.numpy() for cloud in dataset.clouds]
    alignments = None
    if config.get("ot_coupling", True):
        # The original helper silently falls back if POT is missing. An exact
        # migration must fail rather than change the coupling experiment.
        import ot  # noqa: F401

        alignments = getattr(data, profile.alignment_function)(
            marginals, config["train_times"], config.get("ot_method", "emd")
        )
    train_loader, val_loader = getattr(data, profile.loader_function)(
        marginals, batch_size=config["batch_size"], holdout_times=config["holdout_times"],
        val_split=config.get("val_split", 0.0), ot_alignments=alignments,
    )
    potentials = create_potentials(config)
    model = create_model(config, dataset.dim, potentials, device)
    return {
        "model": model, "potentials": potentials, "train_loader": train_loader,
        "val_loader": val_loader, "trainer_class": getattr(trainers, profile.trainer_class),
        "marginals": {i: cloud for i, cloud in enumerate(dataset.clouds)},
        "scaler": scaler_from_statistics(dataset.preprocessing),
        "train_times": config["train_times"], "holdout_times": config["holdout_times"],
        "dim": dataset.dim,
    }
