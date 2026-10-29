"""Load existing trusted ALI checkpoints using explicit data/time metadata.

No teacher retraining, scaler refitting or architecture-name guessing occurs.
Layer sizes come from weights; activations, time layout and preprocessing
come from the declared original checkpoint contract.
"""

from pathlib import Path

import torch
import yaml
from torch import nn

from flow2pc.data.registry import datasets
from flow2pc.pretrain.models.alicfm.adapter import FrozenALICFM
from flow2pc.pretrain.models.alicfm.checkpoint_adapter import ALICFMTeacherWrapper
from flow2pc.pretrain.models.alicfm.profiles import profile_for
from flow2pc.utils.checkpoint import sha256


def extract_state(saved, keys):
    if not isinstance(saved, dict):
        raise ValueError("Legacy import requires a state-dict checkpoint")
    for key in keys:
        value = saved.get(key)
        if (
            isinstance(value, dict)
            and value
            and all(isinstance(v, torch.Tensor) for v in value.values())
        ):
            return value
    if saved and all(isinstance(v, torch.Tensor) for v in saved.values()):
        return saved
    raise ValueError(f"No tensor state dictionary under {keys}")


class ImportedInterpolant(nn.Module):
    def __init__(self, state, activation):
        super().__init__()
        for prefix in ("interpolant_net.", "net."):
            if state and all(key.startswith(prefix) for key in state):
                state = {key[len(prefix) :]: value for key, value in state.items()}
                break
        self.net = ALICFMTeacherWrapper._build_mlp_from_state_dict(state, activation)
        self.net.load_state_dict(state, strict=True)

    def forward(self, x0, x1, t, training=False):
        return (
            (1 - t) * x0 + t * x1 + t * (1 - t) * self.net(torch.cat([x0, x1, t], -1))
        )


def preprocessing_from_checkpoint(saved, meta, name, dim):
    if "preprocessing" in meta:
        return dict(meta["preprocessing"])
    contract = saved.get("data_contract", {})
    original = contract.get("preprocessing", {})
    low_high = saved.get("min_max")
    if original.get("normalise"):
        low_high = (original["min_max"]["min"], original["min_max"]["max"])
    shift, scale = (
        torch.zeros(dim, dtype=torch.float64),
        torch.ones(dim, dtype=torch.float64),
    )
    if original.get("whiten"):
        shift = torch.tensor(original["whitening"]["mean"], dtype=torch.float64)
        scale = torch.tensor(original["whitening"]["scale"], dtype=torch.float64)
    if low_high is not None:
        lo, hi = [torch.as_tensor(x, dtype=torch.float64) for x in low_high]
        shift, scale = shift + scale * lo, scale * (hi - lo)
    elif not original and not meta.get("identity_preprocessing", False):
        raise ValueError(
            "Legacy checkpoint omits scaling. Supply its actual preprocessing, or explicitly declare identity_preprocessing: true"
        )
    features = {
        "cite": "obsm/X_pca",
        "multi": "obsm/X_pca",
        "eb": "obsm/X_phate",
        "st": "aligned_tumor_y_x",
        "cell_tracking": "cell4_mask_xy",
        "knot": "generated_xy",
    }
    return {
        "kind": "affine",
        "fit_split": "legacy_checkpoint",
        "fit_time_indices": original.get("time_order", "legacy_training_scope"),
        "normalise": low_high is not None,
        "whiten": bool(original.get("whiten", False)),
        "shift": shift.float().tolist(),
        "scale": scale.float().tolist(),
        "feature_key": features[name],
        "pca_recomputed": False,
    }


def load(checkpoint, metadata, device="cpu", generator_checkpoint=None):
    metadata = Path(metadata).resolve()
    meta = yaml.safe_load(metadata.read_text())
    if (
        not isinstance(meta, dict)
        or "data" not in meta
        or "time_input" not in meta
        or meta.get("velocity_time_units") != "model"
    ):
        raise ValueError(
            "Legacy metadata must declare data, time_input and velocity_time_units: model"
        )
    preprocessing_path = None
    if "preprocessing_file" in meta:
        if "preprocessing" in meta:
            raise ValueError("Declare preprocessing or preprocessing_file, not both")
        preprocessing_path = Path(meta["preprocessing_file"]).expanduser()
        if not preprocessing_path.is_absolute():
            preprocessing_path = metadata.parent / preprocessing_path
        meta["preprocessing"] = yaml.safe_load(preprocessing_path.read_text())
        if not isinstance(meta["preprocessing"], dict):
            raise ValueError("Saved preprocessing must be a mapping")
    # Existing local experiment checkpoints may contain NumPy metadata; this is
    # the explicit legacy loader, never the new weights-only artifact loader.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    generator_path = (
        Path(generator_checkpoint) if generator_checkpoint else Path(checkpoint)
    )
    generator_saved = (
        saved
        if not generator_checkpoint
        else torch.load(generator_path, map_location="cpu", weights_only=False)
    )
    field_state = extract_state(
        saved,
        (
            "cfm_model",
            "ot_cfm_model",
            "velocity_state_dict",
            "velocity_net",
            "state_dict",
        ),
    )
    generator_state = extract_state(
        generator_saved, ("interpolant", "generator_state_dict", "state_dict")
    )
    field = ALICFMTeacherWrapper._build_mlp_from_state_dict(
        field_state, meta.get("field_activation", "selu")
    )
    field.load_state_dict(field_state, strict=True)
    interpolant = ImportedInterpolant(
        generator_state, meta.get("generator_activation", "elu")
    )
    data = dict(meta["data"])
    params = dict(data.get("params", {}))
    original = saved.get("data_contract", {})
    if data["name"] in ("cite", "multi"):
        params.setdefault(
            "dim",
            original.get("preprocessing", {}).get(
                "n_pca_components_used", meta.get("dim", 50)
            ),
        )
    if "time_index" in original:
        params.setdefault("heldout_index", original["time_index"])
    dim = [module for module in field.modules() if isinstance(module, nn.Linear)][
        -1
    ].out_features
    time_columns = {
        "global": 1,
        "local": 1,
        "local_interval": 2,
        "local_interval_global": 3,
    }
    if meta["time_input"] not in time_columns:
        raise ValueError("Unknown legacy time_input layout")
    field_input = next(
        module for module in field.modules() if isinstance(module, nn.Linear)
    ).in_features
    generator_input = next(
        module for module in interpolant.modules() if isinstance(module, nn.Linear)
    ).in_features
    generator_output = [
        module for module in interpolant.modules() if isinstance(module, nn.Linear)
    ][-1].out_features
    if (
        field_input != dim + time_columns[meta["time_input"]]
        or generator_input != 2 * dim + 1
        or generator_output != dim
    ):
        raise ValueError(
            "Legacy state shapes do not match the explicitly declared feature/time layout"
        )
    if meta.get("cfm_coordinates", "model") not in ("model", "raw"):
        raise ValueError("Legacy cfm_coordinates must be model or raw")
    preprocessing = preprocessing_from_checkpoint(saved, meta, data["name"], dim)
    root = Path(data["root"]).expanduser()
    if not root.is_absolute():
        root = metadata.parent / root
    dataset = datasets.get(data["name"])(
        name=data["name"],
        root=root,
        seed=data.get("split_seed", 2026),
        params=params,
        preprocessing=preprocessing,
    )
    if dataset.dim != dim:
        raise ValueError(
            "Legacy teacher feature dimension differs from data representation"
        )
    if (
        "model_times" in meta
        and list(dataset.metadata.model_times) != meta["model_times"]
    ):
        raise ValueError("Declared legacy model times differ from dataset metadata")
    profile = profile_for(data["name"])
    model_config = {
        "dim": dim,
        "profile": profile,
        "time_input": meta["time_input"],
        "cfm_coordinates": meta.get("cfm_coordinates", "model"),
        "num_marginals": len(dataset.metadata.model_times),
    }
    manifest = {
        "method": "alicfm",
        "format": "legacy",
        "model_config": model_config,
        "dataset": dataset.contract(),
        "distillation_sampler": meta.get(
            "distillation_sampler",
            {
                "single_cell": "mmot_middle",
                "st": "independent",
                "knot": "independent",
                "cell_tracking": "border_ot",
            }[profile],
        ),
        "legacy_provenance": {
            "checkpoint_sha256": sha256(checkpoint),
            "generator_sha256": sha256(generator_path),
            "metadata_sha256": sha256(metadata),
            "preprocessing_file_sha256": sha256(preprocessing_path)
            if preprocessing_path
            else None,
        },
    }
    return FrozenALICFM(
        field.to(device), interpolant.to(device), manifest, preprocessing
    )
