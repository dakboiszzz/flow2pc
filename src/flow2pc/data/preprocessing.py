"""Serializable all-marginal preprocessing with the native float32 operation order.

The protocol is explicit: held-out-time observations contribute to fitted
statistics, as in the supplied native experiments. No teacher code is imported.
"""

import numpy as np
import torch

NATIVE_KIND = "all_marginals_sequential_float32"


def scaler_from_statistics(preprocessing):
    from sklearn.preprocessing import StandardScaler

    statistics = preprocessing["whitening"]
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(statistics["mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(statistics["scale"], dtype=np.float64)
    scaler.var_ = np.asarray(statistics["variance"], dtype=np.float64)
    scaler.n_features_in_ = len(scaler.mean_)
    scaler.n_samples_seen_ = statistics["n_samples"]
    return scaler


def fit_all_marginals(values, *, normalise, whiten, feature_key, time_indices):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Native preprocessing requires finite [N,D] coordinates")
    statistics = None
    mean, std = np.zeros(values.shape[1]), np.ones(values.shape[1])
    if whiten:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        values = scaler.fit_transform(values)
        mean, std = scaler.mean_, scaler.scale_
        statistics = {
            "mean": mean.tolist(),
            "scale": std.tolist(),
            "variance": scaler.var_.tolist(),
            "n_samples": int(scaler.n_samples_seen_),
        }
    coordinates = torch.as_tensor(values, dtype=torch.float32)
    lo, hi = torch.zeros(values.shape[1]), torch.ones(values.shape[1])
    if normalise:
        lo, hi = coordinates.min(0).values, coordinates.max(0).values
        if (hi <= lo).any():
            raise ValueError("Native min-max normalization has a nonpositive range")
    return {
        "kind": NATIVE_KIND,
        "fit_split": "all",
        "fit_time_indices": list(time_indices),
        "heldout_time_contributes_to_statistics": True,
        "normalise": normalise,
        "whiten": whiten,
        "whitening": statistics,
        "min_max": {"min": lo.tolist(), "max": hi.tolist()},
        # Affine summary for velocity unit conversions; forward data transforms
        # must use the sequential operations below, not this fused summary.
        "shift": (mean + std * lo.numpy()).astype(np.float32).tolist(),
        "scale": (std * (hi - lo).numpy()).astype(np.float32).tolist(),
        "metric_shift": mean.astype(np.float32).tolist(),
        "metric_scale": std.astype(np.float32).tolist(),
        "feature_key": feature_key,
        "pca_recomputed": False,
    }


def transform(values, preprocessing):
    values = torch.as_tensor(values, dtype=torch.float32)
    if preprocessing.get("kind") != NATIVE_KIND:
        return (values - torch.tensor(preprocessing["shift"])) / torch.tensor(
            preprocessing["scale"]
        )
    # Held-out marginals intentionally have empty train/val splits. Plotting
    # combines those splits with test rows; sklearn rejects empty transforms.
    if values.ndim == 2 and values.shape[0] == 0:
        return values
    if preprocessing["whiten"]:
        scaler = scaler_from_statistics(preprocessing)
        values = torch.from_numpy(scaler.transform(values.numpy()))
    if preprocessing["normalise"]:
        lo = torch.tensor(preprocessing["min_max"]["min"], dtype=torch.float32)
        hi = torch.tensor(preprocessing["min_max"]["max"], dtype=torch.float32)
        values = (values - lo) / (hi - lo)
    return values


def inverse_minmax(values, preprocessing):
    if preprocessing["normalise"]:
        lo = values.new_tensor(preprocessing["min_max"]["min"])
        hi = values.new_tensor(preprocessing["min_max"]["max"])
        return values * (hi - lo) + lo
    return values


def inverse(values, preprocessing, *, retain_whitening=False):
    result = inverse_minmax(values.float(), preprocessing)
    if preprocessing["whiten"] and not retain_whitening:
        scaler = scaler_from_statistics(preprocessing)
        result = torch.from_numpy(scaler.inverse_transform(result.cpu().numpy()))
    return result.double()
