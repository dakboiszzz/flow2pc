"""Contracts for snapshot preparation. Run only on explicit test authorization."""

import json
import numpy as np
import pytest

from flow2pc.data.mfm_catalog import PROFILES, digest, link_sources, source_links
from flow2pc.data.mfm_snapshots import load, prepare_dataset, read_source, split_and_normalize


def test_catalog_contains_twenty_non_image_profiles():
    assert len(PROFILES) == 20
    assert "afhq" not in PROFILES
    assert {"eb_pca5", "eb_pca50", "eb_phate2"} <= PROFILES.keys()


def test_links_preserve_sources_and_reject_conflicts(tmp_path):
    bundle, native = tmp_path / "bundle", tmp_path / "native"
    for key, target in source_links(bundle, native).items():
        if key == "lidar":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"source las fixture")
        else:
            target.mkdir(parents=True, exist_ok=True)
            (target / "keep").write_bytes(b"original")
    root = tmp_path / "prepared"
    first = link_sources(root, bundle=bundle, native_data=native)
    assert link_sources(root, bundle=bundle, native_data=native) == first
    assert (root / "sources/single_cell").is_symlink()
    assert (root / "sources/single_cell/keep").read_bytes() == b"original"
    (root / "sources/gom").unlink()
    (root / "sources/gom").write_bytes(b"unrelated file")
    with pytest.raises(FileExistsError):
        link_sources(root, bundle=bundle, native_data=native)
    assert (root / "sources/gom").read_bytes() == b"unrelated file"


def test_holdouts_and_validation_never_fit_normalization():
    spec = {"dim": 1, "holdouts": [1, 3], "standardize": True}
    clouds = [np.arange(20, dtype=np.float32)[:, None] + i for i in range(5)]
    _, splits, expected = split_and_normalize(clouds, spec, 42)
    modified = [x.copy() for x in clouds]
    for i in range(5):
        if i in spec["holdouts"]:
            modified[i] += 1e6
            assert not len(splits[i]["train"]) and not len(splits[i]["validation"])
        else:
            modified[i][splits[i]["validation"]] += 1e6
            assert not set(splits[i]["train"]) & set(splits[i]["validation"])
    _, _, actual = split_and_normalize(modified, spec, 42)
    assert actual == expected


def test_gaussian_preserves_nonuniform_times_and_exported_split(tmp_path):
    root = tmp_path / "sources/gaussian"
    root.mkdir(parents=True)
    samples = np.arange(80, dtype=np.float32).reshape(4, 20, 1)
    np.savez(root / "quickstart.npz", samples=samples, times=[0., .33, .67, 1.],
             means=[.5, 1.5, .4, .8], stds=[.1] * 4,
             train_indices=np.arange(18), val_indices=np.arange(18, 20))
    prepared = prepare_dataset("gaussian_quickstart", tmp_path, 42, tmp_path)
    dataset = load({"data_dir": str(prepared)}, {})
    assert dataset.times.tolist() == [0., .33, .67, 1.]
    assert dataset.split_indices["0"] == {"train": list(range(18)), "validation": [18, 19]}
    assert dataset.train_marginals[0].numpy().tolist() == samples[0, :18].tolist()
    assert prepare_dataset("gaussian_quickstart", tmp_path, 42, tmp_path) == prepared
    with pytest.raises(ValueError, match="identity differ"):
        load({"data_dir": str(prepared)}, {"teacher_metadata": {"snapshot_identity": "wrong"}})
    (prepared / "0_values.npy").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="array changed"):
        load({"data_dir": str(prepared)}, {})


@pytest.mark.parametrize("name,dim,key", [("eb_pca5", 5, "X_pca"), ("eb_pca50", 50, "X_pca"), ("eb_phate2", 2, "X_phate")])
def test_eb_h5ad_representation_is_explicit(tmp_path, name, dim, key):
    h5py = pytest.importorskip("h5py")
    directory = tmp_path / "sources/single_cell"
    directory.mkdir(parents=True)
    with h5py.File(directory / "ebdata_v3.h5ad", "w") as handle:
        handle["obsm/X_pca"] = np.ones((20, 50)) * 7
        handle["obsm/X_phate"] = np.ones((20, 2)) * 3
        handle["obs/sample_labels/categories"] = np.array([b"Day 00-03", b"Day 06-09", b"Day 12-15", b"Day 18-21", b"Day 24-27"])
        handle["obs/sample_labels/codes"] = np.repeat(np.arange(5), 4)
    clouds, info, _, metric = read_source(name, tmp_path, 42, tmp_path)
    assert metric is None
    assert clouds[0].shape == (4, dim)
    assert info["provenance"]["feature"] == key
    assert info["physical_times"] is None
    assert np.all(clouds[0] == (7 if key == "X_pca" else 3))


def test_holdout_is_separate_from_validation(tmp_path, monkeypatch):
    import flow2pc.data.mfm_snapshots as snapshots

    clouds = [np.full((20, 2), i, dtype=np.float32) for i in range(9)]
    monkeypatch.setattr(snapshots, "read_source", lambda *args: (
        clouds, {"model_times": np.linspace(0, 1, 9).tolist(), "source_files": []}, None, None,
    ))
    prepared = prepare_dataset("gom", tmp_path, 42, tmp_path)
    dataset = load({"data_dir": str(prepared)}, {})
    assert dataset.observed_indices == [0, 2, 4, 6, 8]
    for i in (1, 3, 5, 7):
        assert len(dataset.train_marginals[i]) == len(dataset.validation_marginals[i]) == 0
        assert len(dataset.test_marginals[i]) == 20
    manifest = json.loads((prepared / "manifest.json").read_text())
    assert manifest["identity"] == digest({k: v for k, v in manifest.items() if k != "identity"})


def test_lidar_retains_geometry_separately_from_endpoints(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch

    import flow2pc.data.mfm_snapshots as snapshots

    (tmp_path / "sources").mkdir()
    (tmp_path / "sources/lidar").write_bytes(b"LAS fixture")
    module_path = tmp_path / "lidar_data.py"
    module_path.write_text("# upstream fixture\n")
    terrain = torch.arange(150, dtype=torch.float32).reshape(50, 3)

    def module(args):
        return SimpleNamespace(
            dataset=terrain,
            train_dataloaders=[SimpleNamespace(dataset=torch.zeros(18, 3)) for _ in range(2)],
            val_dataloaders=[SimpleNamespace(dataset=torch.ones(2, 3)) for _ in range(2)],
            scaler=SimpleNamespace(mean_=np.array([1., 2., 3.]), scale_=np.array([2., 2., 2.])),
        )

    monkeypatch.setattr(snapshots, "source_module", lambda *args: SimpleNamespace(
        LidarDataModule=module, __file__=str(module_path),
    ))
    prepared = prepare_dataset("lidar", tmp_path, 42, tmp_path)
    data = load({"data_dir": str(prepared)}, {})
    torch.testing.assert_close(torch.cat(data.metric_marginals), terrain)
    assert data.normalization["mean"] == [1., 2., 3.]
    # Coordinates from the native loader must not be standardized again.
    assert set(data.marginals[0].flatten().tolist()) == {0., 1.}
    assert len(data.train_marginals[0]) == 18
