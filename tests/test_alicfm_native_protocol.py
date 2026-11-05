"""Native real-data migration contracts; execute only at the user's request."""

import ast
import copy
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml
from sklearn.preprocessing import StandardScaler

from flow2pc.data.preprocessing import inverse, transform
from flow2pc.data.real.single_cell import SingleCell
from flow2pc.data.restore import restore_dataset
from flow2pc.pretrain.models.alicfm.trainer import ALICFM

ROOT = Path(__file__).resolve().parents[1]
TAGS = ("cite5", "cite50", "cite100", "multi5", "multi50", "multi100", "st", "cell_tracking")


def config(tag):
    return yaml.safe_load((ROOT / f"configs/pretrain/alicfm/alicfm_{tag}.yaml").read_text())


@pytest.mark.parametrize("whiten", [False, True])
def test_native_population_order_scaler_and_float32_operations(tmp_path, whiten):
    values = np.random.default_rng(12).normal(size=(80, 3)).astype(np.float32)
    values[20:40] += 100  # Native statistics include the held-out time.
    directory = tmp_path / "single_cell"
    directory.mkdir()
    with h5py.File(directory / "op_cite_inputs_0.h5ad", "w") as f:
        f.create_dataset("obsm/X_pca", data=values)
        f.create_dataset("obs/day", data=np.repeat([2, 3, 4, 7], 20))
    params = {
        "dim": 3, "heldout_index": 1, "normalise": True, "whiten": whiten,
        "preprocessing_protocol": "native_all",
    }
    dataset = SingleCell(name="cite", root=tmp_path, seed=42, params=params)
    native_values = StandardScaler().fit_transform(values) if whiten else values
    clouds = [torch.as_tensor(x, dtype=torch.float32) for x in np.split(native_values, 4)]
    lo = torch.stack([x.min(0).values for x in clouds]).min(0).values
    hi = torch.stack([x.max(0).values for x in clouds]).max(0).values
    for index, cloud in enumerate(clouds):
        actual = dataset.get_marginal(index, "all").materialize()
        torch.testing.assert_close(actual, (cloud - lo) / (hi - lo), rtol=0, atol=0)
        metric = inverse(actual, dataset.preprocessing, retain_whitening=True)
        torch.testing.assert_close(metric, (actual * (hi - lo) + lo).double(), rtol=0, atol=0)
    assert dataset.preprocessing["fit_time_indices"] == [0, 1, 2, 3]
    assert dataset.preprocessing["heldout_time_contributes_to_statistics"]
    assert len(dataset.get_marginal(1, "train")) == 0
    restored = restore_dataset(dataset.contract(), dataset.preprocessing)
    assert restored.contract() == dataset.contract()
    torch.testing.assert_close(
        restored.get_marginal(0, "all").materialize(),
        dataset.get_marginal(0, "all").materialize(), rtol=0, atol=0,
    )


def test_old_affine_preprocessing_still_uses_the_old_operations():
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    p = {"kind": "affine", "shift": [0.1, 0.2], "scale": [0.3, 0.4]}
    torch.testing.assert_close(
        transform(values, p),
        (values - torch.tensor(p["shift"])) / torch.tensor(p["scale"]), rtol=0, atol=0,
    )


@pytest.mark.parametrize("tag", TAGS)
def test_all_eight_teacher_configs_select_the_explicit_native_protocol(tag):
    cfg = config(tag)
    ALICFM().validate(cfg)
    assert cfg["training"]["protocol"] == "native"
    assert cfg["data"]["params"]["preprocessing_protocol"] == "native_all"
    assert cfg["logging"]["project"] == "flow2pc"
    broken = copy.deepcopy(cfg)
    broken["data"]["params"].pop("preprocessing_protocol")
    with pytest.raises(ValueError, match="native_all"):
        ALICFM().validate(broken)


def test_tracking_selects_the_repository_recipe_not_the_selected_checkpoint():
    cfg = config("cell_tracking")
    assert cfg["model"]["params"]["cfm_coordinates"] == "raw"
    assert cfg["model"]["params"]["t_smooth"] == 0.001
    assert cfg["model"]["params"]["correct_coeff"] == 1.0
    assert cfg["training"]["num_ali_train_steps"] == 70000
    assert cfg["training"]["num_cfm_train_steps"] == 50000
    assert cfg["training"]["samples_per_marginal"] == 10


def test_st_plot_methods_are_copied_from_the_matching_native_revision():
    upstream = ROOT.parent / "ali-cfm"
    if not (upstream / ".git").exists():
        pytest.skip("Native Git source is unavailable")
    source = subprocess.check_output(
        ["git", "-C", str(upstream), "show", "501e9db:ali_cfm/spatial_transcriptomics/utils.py"],
        text=True,
    )
    original = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "Plotter")
    migrated = ast.parse((ROOT / "src/flow2pc/pretrain/models/alicfm/_source/st_plot.py").read_text())
    copied = next(n for n in migrated.body if isinstance(n, ast.ClassDef))
    for method in copied.body:
        expected = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == method.name)
        assert ast.dump(method) == ast.dump(expected)
