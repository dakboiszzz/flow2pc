"""Hash-verified native OTP normalization and observed-only OT chains."""

import json
from pathlib import Path

import numpy as np
import ot
from sklearn.preprocessing import StandardScaler
import torch

from repc.single_cell import sha256_file


def load_data(entry, config, split_seed=2026, validation_fraction=.05):
    root = Path(entry["data_root"])
    manifest_path = root / "manifest.json"
    if sha256_file(manifest_path) != entry["data_manifest_sha256"]:
        raise ValueError("OTP data manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text())
    filename = "dingling_pm25_13marginals.npy" if entry["dataset"] == "beijing" else "GoMvortex_data.npy"
    path = root / filename
    if sha256_file(path) != manifest["files_sha256"][filename]:
        raise ValueError("OTP input checksum mismatch")
    # Beijing's hash-verified native object array requires pickle; GoM does not.
    raw = np.load(path, allow_pickle=entry["dataset"] == "beijing")
    raw = [values.reshape(-1, entry["dim"]).astype(np.float32) for values in raw]
    if any(not np.isfinite(values).all() or len(values) < 2 for values in raw):
        raise ValueError("invalid raw marginals")
    scaler = StandardScaler().fit(np.concatenate(raw))
    clouds = [torch.from_numpy(scaler.transform(values).astype(np.float32)) for values in raw]
    observed, holdouts = entry["observed_indices"], entry["holdout_indices"]
    if config["train_times"] != observed or config["holdout_times"] != holdouts:
        raise ValueError("saved config and manifest disagree about observed times")
    if set(observed) & set(holdouts) or sorted(observed + holdouts) != list(range(len(clouds))):
        raise ValueError("invalid observed/held-out partition")
    if len(config["tks"]) != len(observed) - 2:
        raise ValueError("potential count differs from observed tuple ordering")
    if not 0 < validation_fraction < .5:
        raise ValueError("validation fraction must be in (0,.5)")
    order = torch.randperm(len(clouds[0]), generator=torch.Generator().manual_seed(split_seed))
    nv = max(1, round(len(order) * validation_fraction))
    contract = {"dataset": entry["dataset"], "dim": entry["dim"],
                "raw_file": str(path), "raw_sha256": manifest["files_sha256"][filename],
                "data_manifest_sha256": entry["data_manifest_sha256"],
                "counts": [len(values) for values in clouds],
                "times": entry["normalized_times"], "observed_indices": observed,
                "holdout_indices": holdouts, "potential_times": config["tks"],
                "normalization": {"type": "StandardScaler", "fit_scope": "all_original_marginals_including_holdouts",
                                  "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                                  "variance": scaler.var_.tolist(), "input_dtype": "float32"},
                "split_seed": split_seed, "validation_fraction": validation_fraction,
                "split_unit": "source_row; intermediate observations may be shared across tuples"}
    return {"clouds": clouds, "contract": contract, "train_rows": order[nv:],
            "validation_rows": order[:nv]}


def couple_observed(clouds, observed):
    """Same POT float32 squared costs and row-argmax chains as native loaders."""
    indices = torch.empty(len(clouds[observed[0]]), len(observed), dtype=torch.long)
    indices[:, 0] = torch.arange(len(indices))
    for column, (left, right) in enumerate(zip(observed[:-1], observed[1:])):
        source, target = clouds[left].numpy(), clouds[right].numpy()
        cost = ot.dist(source, target, metric="sqeuclidean")
        plan, info = ot.emd(np.ones(len(source)) / len(source), np.ones(len(target)) / len(target),
                            cost, numItermax=10_000_000, log=True)
        if info.get("warning"):
            raise RuntimeError(f"observed OT coupling failed: {info['warning']}")
        mapping = torch.from_numpy(plan.argmax(1))
        indices[:, column + 1] = mapping[indices[:, column]]
    tuples = torch.stack([clouds[index][indices[:, column]]
                          for column, index in enumerate(observed)], 1)
    return tuples, indices


def physical(values, contract):
    normalization = contract["normalization"]
    return (values.double() * values.new_tensor(normalization["scale"], dtype=torch.float64)
            + values.new_tensor(normalization["mean"], dtype=torch.float64))
