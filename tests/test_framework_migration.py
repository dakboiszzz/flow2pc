"""Contract and source-parity tests, with small H5AD-shaped fixtures only."""

import ast
import copy
import importlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch
import yaml

from flow2pc.data.base import TemporalMetadata
from flow2pc.data.real.single_cell import SingleCell
from flow2pc.pretrain.models.alicfm import model
from flow2pc.pretrain.models.alicfm._source import training_funcs
from flow2pc.pretrain.models.alicfm.trainer import MethodConfig
from flow2pc.pretrain.runner import load_teacher, run
from flow2pc.utils.checkpoint import load_teacher_artifact
from flow2pc.utils.config import load_config, resolve_config
from flow2pc.utils.registry import Registry
from flow2pc.utils.seed import seed_everything

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / "ali-cfm" / "ali_cfm"
SOURCE = ROOT / "src/flow2pc/pretrain/models/alicfm/_source"


@pytest.fixture
def data_root(tmp_path):
    directory = tmp_path / "data/single_cell"
    directory.mkdir(parents=True)
    values = np.random.default_rng(11).normal(size=(40, 2)).astype(np.float32)
    values[10:20] += 100_000  # The held-out day must not affect fitted scaling.
    for filename in ("op_cite_inputs_0.h5ad", "op_train_multi_targets_0.h5ad"):
        with h5py.File(directory / filename, "w") as handle:
            handle.create_dataset("obsm/X_pca", data=values)
            handle.create_dataset("obs/day", data=np.repeat([2, 3, 4, 7], 10))
    return directory.parent


@pytest.fixture
def config(tmp_path, data_root):
    raw = yaml.safe_load((ROOT / "configs/pretrain/alicfm_cite50.yaml").read_text())
    raw["data"].update(
        root=str(data_root),
        params={
            "dim": 2,
            "heldout_index": 1,
            "normalise": True,
            "whiten": False,
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
        },
    )
    raw["output"]["dir"] = str(tmp_path / "run")
    raw["logging"] = {"mode": "disabled"}
    raw["model"]["params"].update(
        cfm_dim=8,
        interpolant_dim=8,
        disc_dim=8,
        t_smooth=0.0,
        reg_metric="l2",
        correct_coeff=0.01,
        num_t_steps=1,
        finite_difference_h=0.01,
    )
    raw["training"].update(
        device="cpu",
        batch_size=4,
        num_ali_pretrain_steps=1,
        num_ali_train_steps=1,
        num_cfm_train_steps=1,
    )
    return resolve_config(raw, base=tmp_path)


def dataset_for(config):
    return SingleCell(
        name=config["data"]["name"],
        root=config["data"]["root"],
        seed=config["seed"],
        params=config["data"]["params"],
    )


def native_modules():
    if not (UPSTREAM / "nets.py").exists():
        pytest.skip("Supplied upstream checkout is unavailable for native parity")
    namespace = "_flow2pc_native_parity"
    package = types.ModuleType(namespace)
    package.__path__ = [str(UPSTREAM)]
    sys.modules[namespace] = package
    native_net = importlib.import_module(namespace + ".nets")
    utility = importlib.util.spec_from_file_location(
        namespace + ".training_utils", UPSTREAM / "training/training_utils.py"
    )
    native_utility = importlib.util.module_from_spec(utility)
    sys.modules[utility.name] = native_utility
    utility.loader.exec_module(native_utility)
    # The two unused imports transitively load old hardcoded data/metrics code.
    # Remove only these imports to execute the native loop bodies in isolation.
    tree = ast.parse((UPSTREAM / "training/training_funcs.py").read_text())
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module in ("ali_cfm.loggin_and_metrics", "ali_cfm.data_utils")
        )
    ]
    native_stages = types.ModuleType(namespace + ".training_funcs")
    native_stages.__package__ = namespace
    exec(compile(tree, "native_training_funcs.py", "exec"), native_stages.__dict__)  # noqa: S102
    return native_net, native_utility, native_stages


def test_metadata_preserves_irregular_physical_days(config):
    data = dataset_for(config)
    assert data.metadata.physical_times == (2.0, 3.0, 4.0, 7.0)
    assert data.metadata.model_times == (0.0, 1 / 3, 2 / 3, 1.0)
    assert not data.metadata.has_trajectory_ids
    for day in [2.0, 2.5, 3.0, 3.9, 4.0, 5.5, 7.0]:
        assert data.metadata.to_physical_time(
            data.metadata.to_model_time(day)
        ) == pytest.approx(day)
    assert data.metadata.to_model_time(4) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        data.metadata.to_model_time(10)
    with pytest.raises(ValueError):
        TemporalMetadata("bad", 2, (2.0, 1.0), (0.0, 1.0), "identity", "v1", 1)


def test_loader_splits_and_train_only_scaling(config):
    first, second = dataset_for(config), dataset_for(config)
    for i in range(4):
        for split in ("train", "val", "test"):
            np.testing.assert_array_equal(first.rows[i][split], second.rows[i][split])
        sets = [set(first.rows[i][k]) for k in ("train", "val", "test")]
        assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert len(first.get_marginal(1, "train")) == 0
    assert len(first.get_marginal(1, "val")) == 0
    assert len(first.get_marginal(1, "test")) == 10
    train = torch.cat(
        [first.get_marginal(i, "train").materialize() for i in first.observed]
    )
    torch.testing.assert_close(train.min(0).values, torch.zeros(2))
    torch.testing.assert_close(train.max(0).values, torch.ones(2))
    assert first.get_marginal(1, "test")[0].min() > 1000
    assert first.preprocessing["fit_time_indices"] == [0, 2, 3]
    for i in first.observed:
        marginal = first.get_marginal(i, "train")
        assert len(marginal[0]) == 2
        assert len(marginal.materialize()) == len(marginal)
    changed = copy.deepcopy(config)
    changed["seed"] += 1
    assert (
        first.contract()["split_indices_sha256"]
        != dataset_for(changed).contract()["split_indices_sha256"]
    )


def test_config_roots_and_missing_environment(config, tmp_path, monkeypatch):
    raw = copy.deepcopy(config)
    raw["data"]["root"] = "${MY_MIGRATION_DATA}"
    monkeypatch.delenv("MY_MIGRATION_DATA", raising=False)
    with pytest.raises(ValueError, match="MY_MIGRATION_DATA"):
        resolve_config(raw, base=tmp_path)
    monkeypatch.setenv("MY_MIGRATION_DATA", str(tmp_path / "explicit"))
    monkeypatch.setenv("FLOW2PC_DATA_ROOT", str(tmp_path / "fallback"))
    assert resolve_config(raw, base=tmp_path)["data"]["root"] == str(
        tmp_path / "explicit"
    )
    raw["data"].pop("root")
    assert resolve_config(raw, base=tmp_path)["data"]["root"] == str(
        tmp_path / "fallback"
    )
    raw["output"]["dir"] = "local-output"
    assert resolve_config(raw, base=tmp_path)["output"]["dir"] == str(
        tmp_path / "local-output"
    )


@pytest.mark.parametrize(
    "field,value", [("stage", "unknown"), ("seed", True), ("seed", -1)]
)
def test_invalid_config_keys(config, tmp_path, field, value):
    config[field] = value
    with pytest.raises(ValueError):
        resolve_config(config, base=tmp_path)


def test_registry_unknown_unavailable_and_duplicate():
    registry = Registry("teacher")
    registry.register("unavailable", None, "Awaiting actual source")
    with pytest.raises(ValueError, match="Unknown teacher"):
        registry.get("wrong")
    with pytest.raises(NotImplementedError, match="Awaiting actual source"):
        registry.get("unavailable")
    with pytest.raises(ValueError, match="Duplicate"):
        registry.register("unavailable", None, "again")


def test_discovery_is_lazy_and_boundaries_hold():
    code = """
import sys
from flow2pc.pretrain.registry import teachers
from flow2pc.data.registry import datasets
assert teachers.describe()['alicfm']['status'] == 'implemented'
assert teachers.describe()['otpfm']['status'] == 'unavailable'
assert datasets.describe()['cite']['status'] == 'implemented'
assert not any(k == 'torch' or k.startswith('torch.') for k in sys.modules)
assert 'h5py' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, cwd="/tmp")
    for folder, forbidden in [
        ("pretrain", ("flow2pc.distill", "flow2pc.pc", "repc")),
        ("data", ("flow2pc.pretrain", "repc")),
    ]:
        for file in (ROOT / "src/flow2pc" / folder).rglob("*.py"):
            for node in ast.walk(ast.parse(file.read_text())):
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith(forbidden), file
                elif isinstance(node, ast.Import):
                    assert all(not n.name.startswith(forbidden) for n in node.names), (
                        file
                    )


def test_native_network_and_derivative_parity():
    native, _, _ = native_modules()
    torch.manual_seed(7)
    original = native.TrainableInterpolant(2, 8, t_smooth=0.01)
    migrated = model.TrainableInterpolant(2, 8, t_smooth=0.01)
    migrated.load_state_dict(original.state_dict(), strict=True)
    x0, x1, t = torch.randn(4, 2), torch.randn(4, 2), torch.rand(4, 1)
    torch.testing.assert_close(
        original(x0, x1, t, training=False),
        migrated(x0, x1, t, training=False),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        original.dI_dt(x0, x1, t), migrated.dI_dt(x0, x1, t), rtol=0, atol=0
    )
    torch.manual_seed(19)
    expected = original.compute_length_reg_term(
        {0: x0, 1: x1}, metric="l2", num_t_steps=1
    )
    expected.backward()
    torch.manual_seed(19)
    actual = migrated.compute_length_reg_term(
        {0: x0, 1: x1}, metric="l2", num_t_steps=1
    )
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for p, q in zip(original.parameters(), migrated.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    for name, args in [
        ("MLP", {"dim": 2, "w": 8, "time_varying": True}),
        ("Discriminator", {"in_dim": 3, "w": 8, "apply_sigmoid": False}),
    ]:
        left, right = getattr(native, name)(**args), getattr(model, name)(**args)
        right.load_state_dict(left.state_dict(), strict=True)
        values = torch.randn(4, 3)
        torch.testing.assert_close(left(values), right(values), rtol=0, atol=0)


class NormalizeLogger(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.args.args and node.args.args[-1].arg == "logger":
            node.args.args.pop()
            node.args.defaults.pop()
        if node.args.kwonlyargs and node.args.kwonlyargs[-1].arg == "logger":
            node.args.kwonlyargs.pop()
            node.args.kw_defaults.pop()
        node.body = [
            n
            for n in node.body
            if not (
                isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "logger" for t in n.targets)
            )
        ]
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == "logger":
            node.id = "wandb"
        return node


def test_migrated_algorithm_bodies_match_original():
    if not UPSTREAM.exists():
        pytest.skip("No native checkout")
    original = ast.parse((UPSTREAM / "training/training_funcs.py").read_text())
    migrated = ast.parse((SOURCE / "training_funcs.py").read_text())
    left = [n for n in original.body if isinstance(n, ast.FunctionDef)]
    right = [
        NormalizeLogger().visit(n)
        for n in migrated.body
        if isinstance(n, ast.FunctionDef)
    ]
    assert [ast.dump(n) for n in left] == [ast.dump(n) for n in right]


def test_native_staged_optimizer_update_parity(config):
    from torchcfm.conditional_flow_matching import OTPlanSampler

    native, _, native_stages = native_modules()
    data = dataset_for(config)
    clouds = [data.get_marginal(i, "train").materialize() for i in range(4)]
    cfg = MethodConfig(**config["model"]["params"], **config["training"], dim=2)
    results = []
    for nets, stages in [(native, native_stages), (model, training_funcs)]:
        seed_everything(19)
        generator = nets.TrainableInterpolant(2, 8, t_smooth=0.0)
        discriminator = nets.Discriminator(3, 8, apply_sigmoid=False)
        field = nets.MLP(2, w=8, time_varying=True)
        before = [
            copy.deepcopy(m.state_dict()) for m in (generator, discriminator, field)
        ]
        op = [
            torch.optim.Adam(m.parameters(), lr=0.001)
            for m in (generator, generator, discriminator, field)
        ]
        sampler = OTPlanSampler("exact", reg=0.1)
        rows = []
        logger = types.SimpleNamespace(log=rows.append)
        with patch.object(stages, "wandb", logger):
            stages.pretain_interpolant(
                generator, op[0], sampler, clouds, data.observed, "pre", cfg
            )
            stages.train_interpolant_with_gan(
                generator,
                discriminator,
                sampler,
                clouds,
                op[1],
                op[2],
                "gan",
                data.observed,
                19,
                None,
                cfg,
                plot_freaquency=0,
            )
            stages.train_ot_cfm(
                field,
                op[3],
                generator,
                sampler,
                clouds,
                "cfm",
                cfg,
                data.observed,
                None,
            )
        for module, old in zip((generator, discriminator, field), before, strict=True):
            assert any(
                not torch.equal(value, old[name])
                for name, value in module.state_dict().items()
            )
        results.append(
            ([m.state_dict() for m in (generator, discriminator, field)], rows)
        )
    assert results[0][1] == results[1][1]
    for old, new in zip(results[0][0], results[1][0], strict=True):
        for key in old:
            torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)


@pytest.fixture
def completed(config):
    checkpoint = run(config)
    return checkpoint, config


def test_teacher_train_save_reload_and_independent_inference(completed):
    checkpoint, config = completed
    payload, manifest = load_teacher_artifact(checkpoint)
    assert manifest["dataset"]["physical_times"] == [2.0, 3.0, 4.0, 7.0]
    assert manifest["dataset"]["heldout_index"] == 1
    assert manifest["dataset"]["preprocessing_id"]
    assert manifest["training_protocol"].startswith("one_fold")
    teacher = load_teacher(checkpoint)
    assert not teacher.training and not any(
        p.requires_grad for p in teacher.parameters()
    )
    native = model.MLP(2, w=8, time_varying=True)
    native.load_state_dict(payload["cfm_model"])
    x, t = torch.randn(4, 2), torch.tensor([0.0, 0.3, 0.8, 1.0])
    torch.testing.assert_close(
        teacher.velocity(x, t), native(torch.cat([x, t[:, None]], -1))
    )
    torch.testing.assert_close(
        teacher.velocity(x, 0.3), teacher.velocity(x, torch.full((4,), 0.3))
    )
    a = teacher.interpolate(x, x + 1, t)
    torch.testing.assert_close(a, teacher.interpolate(x, x + 1, t), rtol=0, atol=0)
    assert not a.requires_grad
    code = "from flow2pc.pretrain.runner import load_teacher; import torch; t=load_teacher(__import__('sys').argv[1]); assert t.velocity(torch.zeros(2,2),0.7).shape==(2,2)"
    subprocess.run(
        [sys.executable, "-c", code, str(checkpoint)], cwd="/tmp", check=True
    )
    with pytest.raises(FileExistsError):
        run(config)
    assert (
        len((checkpoint.parent.parent / "metrics.jsonl").read_text().splitlines()) == 3
    )


@pytest.mark.parametrize(
    "case", ["shape", "time_shape", "time_range", "dtype", "nan", "context"]
)
def test_frozen_adapter_rejects_invalid_queries(completed, case):
    checkpoint, _ = completed
    teacher = load_teacher(checkpoint)
    x, t, context = torch.zeros(2, 2), 0.5, None
    if case == "shape":
        x = torch.zeros(2, 3)
    if case == "time_shape":
        t = torch.zeros(2, 1)
    if case == "time_range":
        t = 2.0
    if case == "dtype":
        x = x.double()
    if case == "nan":
        x[0, 0] = float("nan")
    if case == "context":
        context = {"ignored": True}
    with pytest.raises(ValueError):
        teacher.velocity(x, t, context=context)


@pytest.mark.parametrize("artifact", ["weights", "preprocessing", "time", "dim"])
def test_checkpoint_rejects_tampering(completed, artifact):
    checkpoint, _ = completed
    manifest_path = checkpoint.parent.parent / "manifest.json"
    if artifact == "weights":
        with checkpoint.open("ab") as handle:
            handle.write(b"tamper")
    elif artifact == "preprocessing":
        path = checkpoint.parent.parent / "preprocessing.json"
        path.write_text(path.read_text() + " ")
    else:
        manifest = json.loads(manifest_path.read_text())
        if artifact == "time":
            manifest["dataset"]["model_times"] = [0.0, 0.9, 0.5, 1.0]
        else:
            manifest["model_config"]["dim"] = 3
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_teacher(checkpoint)


def test_cli_validates_without_outputs_and_rejects_unavailable(config, tmp_path):
    file = tmp_path / "config.yaml"
    file.write_text(yaml.safe_dump(config))
    for command in [
        [sys.executable, str(ROOT / "main.py")],
        [sys.executable, "-m", "flow2pc.cli"],
    ]:
        result = subprocess.run(
            [*command, "--config", str(file), "--validate-only"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(result.stdout)["dataset"]["dim"] == 2
    assert not Path(config["output"]["dir"]).exists()
    config["model"]["name"] = "mfm"
    file.write_text(yaml.safe_dump(config))
    result = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--config", str(file)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2 and "Not yet migrated" in result.stderr
    assert not Path(config["output"]["dir"]).exists()


def test_distill_contract_does_not_silently_train(tmp_path):
    file = tmp_path / "distill.yaml"
    file.write_text(
        yaml.safe_dump(
            {
                "stage": "distill",
                "teacher": {"checkpoint": "teacher.pt"},
                "student": {"name": "hclt"},
                "distillation": {
                    "sampler": {"name": "teacher_path"},
                    "steps": 1,
                    "batch_size": 2,
                    "lr": 0.001,
                },
                "output": {"dir": "result"},
            }
        )
    )
    result = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--config", str(file)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2 and "separate migration phase" in result.stderr
    assert not (tmp_path / "result").exists()
    assert load_config(file)["teacher"]["checkpoint"] == str(tmp_path / "teacher.pt")


def test_pc_exports_preserve_existing_implementation():
    from flow2pc import pc
    from repc.hclt_pc import TimeDependentHCLTPC

    assert pc.TimeDependentHCLTPC is TimeDependentHCLTPC


def test_packaged_source_has_recorded_integrity():
    from flow2pc.utils.checkpoint import sha256

    provenance = json.loads((SOURCE.parent / "SOURCE_PROVENANCE.json").read_text())
    for file in provenance["files"]:
        assert sha256(SOURCE / file["migrated_file"]) == file["migrated_sha256"]
        upstream = UPSTREAM.parent / file["upstream_file"]
        if upstream.exists():
            assert sha256(upstream) == file["upstream_sha256"]
