"""Checkpoint-independent profiles for the three MFM vector datasets."""

from copy import deepcopy

from flow2pc.utils.paths import MFM_PRETRAIN_ROOT


def vector_profile(name, seed=42, **options):
    if name not in ("arch", "sphere", "lidar"):
        raise ValueError("MMFM currently supports arch, sphere and lidar")
    allowed = {"source_root", "whiten", "heldout_index", "train_fraction"}
    if set(options) - allowed:
        raise ValueError(f"Unknown dataset options: {sorted(set(options) - allowed)}")
    heldout = options.get("heldout_index", None if name == "lidar" else 1)
    if heldout is not None and (type(heldout) is not int or heldout != 1 or name == "lidar"):
        raise ValueError("Only interior index 1 of arch/sphere may be held out")
    fraction = float(options.get("train_fraction", 0.9))
    if not 0 < fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    return {
        "repo_path": str(options.get("source_root", MFM_PRETRAIN_ROOT)),
        "args": {
            "data_type": name, "data_name": name,
            "dim": 2 if name == "arch" else 3,
            "seed_current": seed, "batch_size": 128,
            "split_ratios": [0.9, 0.1],
            "whiten": options.get("whiten", name == "lidar"),
            "t_exclude_current": heldout,
        },
    }


def dataset_identity(data):
    """Coordinates and splits that must be identical across teacher and student."""
    return deepcopy({
        "source_info": data.source_info,
        "normalization": data.normalization,
        "split_indices": data.split_indices,
        "times": data.times.tolist(),
        "heldout_indices": data.heldout_indices,
    })
