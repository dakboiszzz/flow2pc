"""Prepared unpaired marginals, shared by MFM teacher and PC student.

No teacher imports. Source arrays are never transformed in place.
"""

import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from flow2pc.data.mfm_catalog import FORMAT, digest, prepared_path, profile
from flow2pc.utils.mfm_source import fingerprint, source_module


def read_source(name, root, seed, source_root):
    spec = profile(name)
    family = spec["family"]
    sources = Path(root) / "sources"
    files, provenance, fixed, metric_reference = [], {}, None, None
    if family in ("cite", "multi", "eb"):
        import h5py

        from flow2pc.data.real.single_cell import read_days

        filename = {"cite": "op_cite_inputs_0.h5ad", "multi": "op_train_multi_targets_0.h5ad", "eb": "ebdata_v3.h5ad"}[family]
        path = sources / "single_cell" / filename
        with h5py.File(path, "r") as handle:
            matrix = handle[f"obsm/{spec['feature']}"]
            if matrix.shape[1] < spec["dim"]:
                raise ValueError("Source has insufficient feature dimensions")
            values = matrix[:, :spec["dim"]]
            if family == "eb":
                node = handle["obs/sample_labels"]
                codes = node["codes"][:]
                categories = [x.decode() if isinstance(x, bytes) else str(x) for x in node["categories"][:]]
                if (codes < 0).any() or (codes >= len(categories)).any():
                    raise ValueError("Missing or invalid EB category code")
                labels = sorted(categories)
                clouds = [values[codes == categories.index(label)] for label in labels]
                physical = None
                units = "day_range_categories; no midpoint inferred"
            else:
                days = read_days(handle)
                labels = [2, 3, 4, 7]
                if set(np.unique(days)) != set(labels):
                    raise ValueError("Expected CITE/Multi days [2,3,4,7]")
                clouds = [values[days == day] for day in labels]
                physical, units = labels, "day"
        files = [path]
        provenance = {"feature": spec["feature"], "pca_recomputed": False}
        times = np.linspace(0, 1, len(clouds))
    elif family == "gaussian":
        path = sources / "gaussian" / f"{spec['recipe']}.npz"
        with np.load(path, allow_pickle=False) as archive:
            clouds = list(archive["samples"])
            times = archive["times"].copy()
            fixed = (archive["train_indices"].copy(), archive["val_indices"].copy())
            provenance = {"means": archive["means"].tolist(), "stds": archive["stds"].tolist()}
        labels = physical = times.tolist()
        units, files = "synthetic_model_time", [path]
    elif family in ("arch", "sphere"):
        from flow2pc.data.metric_common import seeded

        module = source_module("dataloaders.trajectory_data", source_root)
        with seeded(seed):
            values, categories, order = getattr(module, f"generate_{family}_data")()
        clouds = [values[categories == label] for label in order]
        times = np.linspace(0, 1, len(clouds))
        labels, physical, units = order.tolist(), None, "synthetic_snapshot_index"
        files = [Path(module.__file__)]
        provenance = {"generator_seed": seed, "generator": f"generate_{family}_data"}
    elif family == "lidar":
        import torch

        from flow2pc.data.metric_common import seeded

        module = source_module("dataloaders.lidar_data", source_root)
        args = SimpleNamespace(dim=3, batch_size=128, split_ratios=[0.9, 0.1],
                               whiten=True, data_path=str(sources / "lidar"))
        with seeded(seed):
            native = module.LidarDataModule(args)
        clouds = [torch.cat([train.dataset, val.dataset]).float().numpy()
                  for train, val in zip(native.train_dataloaders, native.val_dataloaders, strict=True)]
        # The terrain is conditioning geometry, not an observed endpoint or test
        # marginal. Preserve the original metric reference pools as well as x0/x1.
        scene = native.dataset.float().numpy()
        metric_reference = [scene[:len(scene) // 2], scene[len(scene) // 2:]]
        normalization = {"mean": native.scaler.mean_.tolist(), "scale": native.scaler.scale_.tolist(),
                         "fit_scope": "full_ground_scene_as_in_upstream"}
        labels, files = [0, 1], [sources / "lidar", Path(module.__file__)]
        times, physical, units = np.array([0., 1.]), None, "generated_transport_parameter"
        provenance = {"generator_seed": seed, "upstream_normalization": normalization,
                      "exception": "terrain geometry and scaler fitted to the full ground scene"}
    else:
        from flow2pc.data.real.transport import TransportMarginals
        from flow2pc.data.synthetic.knot import Knot
        from flow2pc.data.temporal import CellTracking, SpatialTranscriptomics

        cls = {"gom": TransportMarginals, "beijing": TransportMarginals,
               "st": SpatialTranscriptomics, "cell_tracking": CellTracking, "knot": Knot}[family]
        params = {}
        if family in ("gom", "beijing"):
            # Only call the raw reader; its native_all constructor is not used.
            params = {"normalise": False, "whiten": True, "preprocessing_protocol": "native_all"}
        elif family == "knot":
            params = {"size": 120, "samples_per_time": 10, "noise_std": 0.01}
        reader = SimpleNamespace(root=sources, name=family, params=params, seed=seed)
        clouds, physical, labels, provenance = cls.read_source(reader)
        files = [sources / p for p in provenance.get("files", {})]
        if family == "knot":
            import inspect

            files.append(Path(inspect.getfile(Knot)))
        units = provenance.get("physical_time_units", "observation_index")
        # Do not carry the transport reader's legacy normalization-scope claim.
        provenance = {k: v for k, v in provenance.items() if k not in ("files", "preprocessing_scope")}
        times = np.linspace(0, 1, len(clouds))
    return clouds, {
        "labels": labels, "physical_times": physical, "physical_time_units": units,
        "model_times": np.asarray(times, dtype=np.float64).tolist(),
        "time_transform": "saved_explicit_grid" if family == "gaussian" else "ordinal_snapshot_index_to_unit_interval",
        "source_files": [{"path": str(p.resolve()), "sha256": fingerprint(p)} for p in files],
        "provenance": provenance, "has_trajectory_ids": False,
    }, fixed, metric_reference


def split_and_normalize(clouds, spec, seed, fixed=None):
    clouds = [np.asarray(x, dtype=np.float32) for x in clouds]
    heldout = spec["holdouts"]
    if len(clouds) < 2 or any(i <= 0 or i >= len(clouds) - 1 for i in heldout):
        raise ValueError("Holdouts must be interior marginal indices")
    rng = np.random.default_rng(seed)
    splits = []
    for i, values in enumerate(clouds):
        if values.ndim != 2 or values.shape[1] != spec["dim"] or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("Expected finite [N,dim] marginals with at least two samples")
        if i in heldout:
            train, val = np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        elif fixed is not None:
            train, val = (np.asarray(x) for x in fixed)
            combined = np.concatenate([train, val])
            if (combined.dtype.kind not in "iu" or not len(train) or not len(val)
                    or len(combined) != len(values)
                    or not np.array_equal(np.sort(combined), np.arange(len(values)))):
                raise ValueError("Invalid exported Gaussian split")
        else:
            order = rng.permutation(len(values))
            cut = max(1, min(len(values) - 1, int(0.9 * len(values))))
            train, val = order[:cut], order[cut:]
        splits.append({"train": train.astype(np.int64), "validation": val.astype(np.int64)})
    mean, scale = np.zeros(spec["dim"]), np.ones(spec["dim"])
    if spec["standardize"]:
        training = np.concatenate([x[s["train"]] for x, s in zip(clouds, splits, strict=True)]).astype(np.float64)
        mean, scale = training.mean(0), training.std(0).clip(1e-12)
    normalized = [((x - mean) / scale).astype(np.float32) for x in clouds]
    if any(not np.isfinite(x).all() for x in normalized):
        raise ValueError("Nonfinite normalized coordinates")
    normalization = {"mean": mean.tolist(), "scale": scale.tolist(),
                     "fit_scope": "observed_training_rows_only" if spec["standardize"] else "identity",
                     "fit_time_indices": [i for i in range(len(clouds)) if i not in heldout]}
    return normalized, splits, normalization


def prepare_dataset(name, root, seed, source_root):
    spec = profile(name)
    destination = prepared_path(root, name, seed)
    if destination.exists():
        manifest = read_manifest(destination)
        if manifest["profile"] != name or manifest["spec"] != spec or manifest["split_seed"] != seed:
            raise ValueError("Prepared profile differs; choose a fresh data root")
        for record in manifest["source_files"]:
            if fingerprint(record["path"]) != record["sha256"]:
                raise ValueError("Source changed; preserve prepared data and use a fresh data root")
        return destination
    clouds, info, fixed, metric_reference = read_source(name, root, seed, source_root)
    clouds, splits, normalization = split_and_normalize(clouds, spec, seed, fixed)
    if spec["family"] == "lidar":
        # Native loader has already applied this scaler. Record it for inverse
        # transforms without applying it a second time to the saved coordinates.
        normalization = {**info["provenance"]["upstream_normalization"],
                         "applied_by": "native_lidar_loader; full_ground_scene_exception"}
    grid = np.asarray(info["model_times"])
    if grid.shape != (len(clouds),) or not np.isfinite(grid).all() or grid[0] != 0 or grid[-1] != 1 or (np.diff(grid) <= 0).any():
        raise ValueError("Model times must increase strictly from zero to one")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".preparing-", dir=destination.parent))
    try:
        arrays = {}
        if metric_reference is not None:
            for i, values in enumerate(metric_reference):
                if values.ndim != 2 or values.shape[1] != spec["dim"] or not len(values) or not np.isfinite(values).all():
                    raise ValueError("Invalid terrain metric reference")
                filename = f"metric_{i}.npy"
                np.save(temporary / filename, values, allow_pickle=False)
                arrays[filename] = fingerprint(temporary / filename)
        for i, (cloud, split) in enumerate(zip(clouds, splits, strict=True)):
            for key, values in {"values": cloud, **split}.items():
                filename = f"{i}_{key}.npy"
                np.save(temporary / filename, values, allow_pickle=False)
                arrays[filename] = fingerprint(temporary / filename)
        manifest = {"format": FORMAT, "profile": name, "spec": spec, **info,
                    "split_seed": seed, "split_protocol": "exported_90_10" if fixed else "independent_rows_90_10_observed_only",
                    "normalization": normalization, "heldout_indices": spec["holdouts"],
                    "marginal_sizes": [len(x) for x in clouds], "arrays": arrays,
                    "metric_reference_count": len(metric_reference) if metric_reference is not None else 0,
                    "metric_reference_scope": "full_ground_scene_native_geometry_exception" if metric_reference is not None else "observed_training_rows_only",
                    "initial_law": "empirical_earliest_training_marginal"}
        manifest["identity"] = digest(manifest)
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def read_manifest(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("format") != FORMAT or manifest.get("identity") != digest({k: v for k, v in manifest.items() if k != "identity"}):
        raise ValueError("Invalid snapshot manifest identity/version")
    count = len(manifest["marginal_sizes"])
    grid = np.asarray(manifest["model_times"])
    if (count < 2 or grid.shape != (count,) or not np.isfinite(grid).all()
            or grid[0] != 0 or grid[-1] != 1 or (np.diff(grid) <= 0).any()
            or any(i <= 0 or i >= count - 1 for i in manifest["heldout_indices"])):
        raise ValueError("Invalid saved time grid/holdouts")
    expected_arrays = {f"{i}_{key}.npy" for i in range(count) for key in ("values", "train", "validation")}
    if manifest["metric_reference_count"] != (2 if manifest["spec"]["family"] == "lidar" else 0):
        raise ValueError("Unexpected metric reference geometry")
    expected_arrays.update(f"metric_{i}.npy" for i in range(manifest["metric_reference_count"]))
    if set(manifest["arrays"]) != expected_arrays:
        raise ValueError("Incomplete prepared arrays")
    for filename, expected in manifest["arrays"].items():
        if Path(filename).name != filename or fingerprint(directory / filename) != expected:
            raise ValueError(f"Snapshot array changed: {filename}")
    return manifest


def load(config, context):
    import torch

    from flow2pc.data.metric_base import MetricDatasetBundle

    metadata = context.get("teacher_metadata", {})
    directory = config.get("data_dir") or metadata.get("prepared_path")
    if not directory:
        raise ValueError("Prepared snapshot directory is required; run prepare first")
    directory = Path(directory).expanduser().resolve()
    manifest = read_manifest(directory)
    expected = metadata.get("snapshot_identity")
    if expected is not None and manifest["identity"] != expected:
        raise ValueError("Teacher and prepared dataset identity differ")
    marginals, training, validation, splits = [], [], [], {}
    for i in range(len(manifest["marginal_sizes"])):
        values = torch.from_numpy(np.load(directory / f"{i}_values.npy", allow_pickle=False))
        indices = {k: np.load(directory / f"{i}_{k}.npy", allow_pickle=False) for k in ("train", "validation")}
        if values.ndim != 2 or values.shape != (manifest["marginal_sizes"][i], manifest["spec"]["dim"]) or not torch.isfinite(values).all():
            raise ValueError("Invalid saved marginal shape/values")
        combined = np.concatenate(list(indices.values()))
        if combined.dtype.kind not in "iu" or any(x.ndim != 1 for x in indices.values()):
            raise ValueError("Invalid saved split indices")
        if i in manifest["heldout_indices"]:
            if len(combined):
                raise ValueError("Temporal test holdout entered train/validation")
        elif not all(len(x) for x in indices.values()) or not np.array_equal(np.sort(combined), np.arange(len(values))):
            raise ValueError("Saved train/validation splits overlap or do not partition observations")
        marginals.append(values)
        training.append(values[indices["train"]])
        # Test references retained separately; never used for validation sampling.
        validation.append(values[indices["validation"]])
        splits[str(i)] = {k: v.tolist() for k, v in indices.items()}
    info = {**manifest, "snapshots_sha256": manifest["identity"], "prepared_path": str(directory)}
    result = MetricDatasetBundle(
        None, (manifest["spec"]["dim"],), "metric", training=training,
        validation=validation, times=torch.tensor(manifest["model_times"], dtype=torch.float64),
        source_info=info, marginals=marginals, train_marginals=training,
        validation_marginals=validation, heldout_indices=manifest["heldout_indices"],
        normalization=manifest["normalization"], split_indices=splits,
    )
    result.test_marginals = [x if i in result.heldout_indices else x[:0] for i, x in enumerate(marginals)]
    result.metric_marginals = (
        [torch.from_numpy(np.load(directory / f"metric_{i}.npy", allow_pickle=False))
         for i in range(manifest["metric_reference_count"])]
        if manifest["metric_reference_count"] else training
    )
    return result
